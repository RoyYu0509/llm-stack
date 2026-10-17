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
from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

GPT2_EOS_TOKEN_ID = 50256


@dataclass
class AdapterConfig:
    """The subset of a HuggingFace `PretrainedConfig` that `LMEngine` actually reads."""

    n_positions: int
    eos_token_id: int
    vocab_size: Optional[int] = None

# 一个 NamedTuple, 方便 LMEngine 直接解包成 logits, past_key_values
class CausalLMOutput(NamedTuple):
    """Stands in for HF's `CausalLMOutputWithPast`, with only the fields LMEngine uses.

    `past_key_values` is the running token prefix, not layer-wise key/value tensors --
    see the module docstring.
    """

    logits: torch.Tensor
    past_key_values: torch.Tensor


class TransformerLMAdapter(nn.Module):
    """
    Wrap a from-scratch causal LM so `LMEngine` can drive it unchanged.

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
        decode_backend: str = "ragged",
        decode_slots: Optional[int] = None,
        decode_max_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.use_kv_cache = use_kv_cache
        # "ragged": RaggedDecodeBatch (left-padded rectangle, re-cat'ed every step) -- the original path.
        # "prealloc": PreallocDecodeBatch (slots allocated once, K/V appended in place, FA-2 kv-cache kernel).
        self.last_token_logits_only = False       # True: prefill returns (1, 1, vocab) logits instead of (1, S, vocab)
        self.decode_backend = decode_backend
        self.decode_slots = decode_slots          # prealloc: max requests decoding at once
        self.decode_max_len = decode_max_len      # prealloc: prompt + generated tokens a slot can hold
        self.compile_decode = False               # prealloc: run the decode forward as one torch.compile'd graph
        self.compile_backend = "inductor"         # "aot_eager" = trace only (CPU tests); anything torch.compile accepts
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

    @property
    def supports_batched_decode(self) -> bool:
        return self.use_kv_cache and getattr(self.model, "supports_ragged_decode", False)

    def new_decode_batch(self):
        if not self.supports_batched_decode:
            raise RuntimeError("batched decode needs use_kv_cache=True and a ragged-capable model")

        if self.decode_backend == "prealloc":
            from inferenceLM.engine.prealloc_batch import PreallocDecodeBatch

            return PreallocDecodeBatch(
                self.model,
                n_slots=self.decode_slots or 64,
                max_len=self.decode_max_len or self.config.n_positions,
                dtype=next(self.model.parameters()).dtype,
                device=self.device,
                compile_decode=self.compile_decode,
                compile_backend=self.compile_backend,
            )

        from inferenceLM.engine.ragged_batch import RaggedDecodeBatch

        return RaggedDecodeBatch(self.model)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        last_index: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        """
        One forward pass in HF's calling convention.

        On prefill `past_key_values` is None and `input_ids` is the whole prompt. On
        decode `input_ids` is the single newest token and `past_key_values` is the
        prefix returned by the previous step; the two are concatenated and the model is
        re-run over the result.

        Returns logits for the full prefix (LMEngine slices `[:, -1, :]` itself) and the
        new prefix to hand back on the next step.
        
        Example.
            # Prefill(第一次调用,past_key_values=None)
            ------------------------------------------
            input_ids:        (1, 5)          # 整个 prompt
            prefix = input_ids                # (1, 5),没有 past 可以 cat
            logits = self.model(prefix)       # (1, 5, vocab_size)
            返回 past_key_values = prefix     # (1, 5)

            - LMEngine 自己做 logits[:, -1, :] 取最后一个位置采样下一个 token。

            # Decode step 1
            ------------------------------------------
            input_ids:         (1, 1)                        # 上一步采样出的新 token
            past_key_values:   (1, 5)                        # 上一步返回的 prefix
            prefix = cat([past_key_values, input_ids], dim=1) # (1, 6)
            6 <= n_positions(8),不截断
            logits = self.model(prefix)                       # (1, 6, vocab_size)
            返回 past_key_values = prefix                      # (1, 6)

            # 继续到 decode step 4,prefix 长度超过 n_positions
            ------------------------------------------
            past_key_values:  (1, 8)     # 上一步
            input_ids:        (1, 1)
            prefix = cat(...)            # (1, 9)
            9 > n_positions(8) → 截断
            prefix = prefix[:, -8:]      # (1, 8),丢掉最左边(最老)的 1 个 token
            logits = self.model(prefix)  # (1, 8, vocab_size)
            返回 past_key_values = prefix # (1, 8)
        """
        # If: 开启 KV Cache, 走正常 model 内置的 KV Cache 逻辑
        if self.use_kv_cache:
            # LMEngine treats past_key_values as opaque, so in this mode it carries the
            # model's real per-layer (K, V) state instead of a token prefix. Each step is
            # then O(prefix) rather than O(prefix^2) cumulative.
            kw = {"last_only": True} if (self.last_token_logits_only and past_key_values is None) else {}
            if last_index is not None:   # batched prefill of right-padded prompts (see LMEngine.prefill_batch)
                kw = {"last_index": last_index}
            logits, new_past = self.model.forward_with_cache(
                input_ids.to(self.device), past_kv=past_key_values, **kw
            )
            return CausalLMOutput(logits=logits, past_key_values=new_past)
        # Else: 没 KV Cache, 每一步都要完整的传入目前为止生成的 token id 序列本身
        else:
            if last_index is not None:
                raise NotImplementedError("batched prefill needs use_kv_cache=True")
            if past_key_values is None:
                prefix = input_ids
            else:
                prefix = torch.cat([past_key_values, input_ids.to(past_key_values.device)], dim=1)

            # Keep the window inside the model's positional range. LMEngine already refuses
            # a max_length above n_positions, so this only bites if a caller drives the
            # adapter directly -- truncating from the left is the least surprising behaviour.
            if prefix.shape[1] > self.config.n_positions:
                prefix = prefix[:, -self.config.n_positions:]

            # 最终的结果
            prefix = prefix.to(self.device)
            logits = self.model(prefix)
            return CausalLMOutput(logits=logits, past_key_values=prefix)

