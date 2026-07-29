"""Adapter presenting a locally-trained causal LM through the HuggingFace-shaped
interface that `LMEngine` expects.

Why this exists
---------------
`LMEngine` (ADR-001) deliberately drives the model by hand instead of `.generate()`, and
in doing so it depends on four things that are HuggingFace conventions rather than
anything intrinsic to a transformer:

    model.eval()
    model(input_ids=..., past_key_values=..., use_cache=True) -> .logits, .past_key_values
    model.config.n_positions
    model.config.eos_token_id

A from-scratch `TransformerLM` has none of them: its `forward(x)` maps
`(batch, seq) -> (batch, seq, vocab)` and it carries no config object and no KV cache.
This adapter supplies exactly those four things and nothing else, so `LMEngine`,
`InferenceEngine`, and `RequestReceiver` run against a locally-trained model unmodified.

Deliberately NOT a real KV cache
--------------------------------
`past_key_values` here is not a cache of keys and values -- it is the token prefix, and
every decode step re-runs the full forward pass over it. That makes decoding O(n^2) in
sequence length where a real cache would be O(n).

This is the documented trade-off from the sprint plan ("recompute-based decode, no real
KV-cache"), not an oversight. The reason is that a genuine cache is a change to
`TransformerLM`'s attention module -- every block would have to accept and return
per-layer (k, v) tensors, and the Triton FlashAttention kernel would need an incremental
path -- which is a substantially larger piece of work than the serving integration
itself. Recompute keeps the two projects' seam small and honest: the serving layer is
genuinely running the locally-trained weights, and the cost of not having a cache shows
up as latency rather than as a correctness fudge.

Tokenizer note: the model is trained with tiktoken's pretrained "gpt2" encoding, which is
token-for-token identical to HuggingFace's `gpt2` tokenizer (same 50257-entry BPE, same
`<|endoftext|>` = 50256). So `RequestReceiver`'s existing `AutoTokenizer` path needs no
change either -- point it at "gpt2".
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

GPT2_EOS_TOKEN_ID = 50256


@dataclass
class AdapterConfig:
    """The subset of a HuggingFace `PretrainedConfig` that `LMEngine` actually reads."""

    n_positions: int
    eos_token_id: int
    vocab_size: Optional[int] = None


class CausalLMOutput(NamedTuple):
    """Stands in for HF's `CausalLMOutputWithPast`, with only the fields LMEngine uses.

    `past_key_values` is the running token prefix, not layer-wise key/value tensors --
    see the module docstring.
    """

    logits: torch.Tensor
    past_key_values: torch.Tensor


class TransformerLMAdapter(nn.Module):
    """Wrap a from-scratch causal LM so `LMEngine` can drive it unchanged.

    Args:
        model: any module mapping `(batch, seq)` token ids to `(batch, seq, vocab)`
            logits. Typed loosely on purpose: this keeps `inferenceLM` free of any
            import of the training project, so the two repos stay decoupled and the
            adapter is testable against a stub.
        context_length: the model's maximum position count; surfaced as
            `config.n_positions`, which `LMEngine.inference()` validates `max_length`
            against.
        eos_token_id: surfaced as `config.eos_token_id` for `stopping_criteria()`.
            Defaults to GPT-2's, matching the tiktoken encoding used in training.
        vocab_size: recorded on the config for introspection; unused by LMEngine.
    """

    def __init__(
        self,
        model: nn.Module,
        context_length: int,
        eos_token_id: int = GPT2_EOS_TOKEN_ID,
        vocab_size: Optional[int] = None,
        use_kv_cache: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.use_kv_cache = use_kv_cache
        if use_kv_cache and not hasattr(model, "forward_with_cache"):
            raise ValueError(
                "use_kv_cache=True requires the wrapped model to expose forward_with_cache("
                "input_ids, past_kv) -> (logits, past_kv). Leave it False to use the "
                "recompute path, which works with any (batch, seq) -> logits module."
            )
        self.config = AdapterConfig(
            n_positions=context_length,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> CausalLMOutput:
        """One forward pass in HF's calling convention.

        On prefill `past_key_values` is None and `input_ids` is the whole prompt. On
        decode `input_ids` is the single newest token and `past_key_values` is the
        prefix returned by the previous step; the two are concatenated and the model is
        re-run over the result.

        Returns logits for the full prefix (LMEngine slices `[:, -1, :]` itself) and the
        new prefix to hand back on the next step.
        """
        if self.use_kv_cache:
            # LMEngine treats past_key_values as opaque, so in this mode it carries the
            # model's real per-layer (K, V) state instead of a token prefix. Each step is
            # then O(prefix) rather than O(prefix^2) cumulative.
            logits, new_past = self.model.forward_with_cache(
                input_ids.to(self.device), past_kv=past_key_values
            )
            return CausalLMOutput(logits=logits, past_key_values=new_past)

        if past_key_values is None:
            prefix = input_ids
        else:
            prefix = torch.cat([past_key_values, input_ids], dim=1)

        # Keep the window inside the model's positional range. LMEngine already refuses
        # a max_length above n_positions, so this only bites if a caller drives the
        # adapter directly -- truncating from the left is the least surprising behaviour.
        if prefix.shape[1] > self.config.n_positions:
            prefix = prefix[:, -self.config.n_positions:]

        prefix = prefix.to(self.device)
        logits = self.model(prefix)
        return CausalLMOutput(logits=logits, past_key_values=prefix)
