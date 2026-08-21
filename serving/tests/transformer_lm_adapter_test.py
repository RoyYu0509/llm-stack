import pytest
import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from inferenceLM.data.tokenized_data import TokenizedData
from inferenceLM.engine.lm_engine import LMEngine
from inferenceLM.engine.transformer_lm_adapter import (
    GPT2_EOS_TOKEN_ID,
    TransformerLMAdapter,
)


class _LogitsOnly(nn.Module):
    """Reduce a HuggingFace causal LM to the bare `(batch, seq) -> (batch, seq, vocab)`
    signature that a from-scratch `TransformerLM` has.

    This is the point of the test: GPT-2 stripped of its config, its cache and its output
    dataclass looks exactly like our own model, so driving it through the adapter and
    comparing against HuggingFace's own cached `generate()` isolates the adapter's
    recompute logic. If recompute-decode is wrong, the two token streams diverge.
    """

    def __init__(self, hf_model: GPT2LMHeadModel) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.hf_model(input_ids=input_ids).logits


@pytest.fixture(scope="module")
def hf_model():
    return GPT2LMHeadModel.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def adapted(hf_model):
    return TransformerLMAdapter(
        _LogitsOnly(hf_model),
        context_length=hf_model.config.n_positions,
        eos_token_id=GPT2_EOS_TOKEN_ID,
        vocab_size=hf_model.config.vocab_size,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt, max_length",
    [
        ("Hello, how are you? My name is Yifan and ", 30),
        ("Yifan Yu is ", 25),
    ],
    ids=["short_seq", "random_prompt"],
)
async def test_adapter_recompute_decode_matches_huggingface(
    hf_model, tokenizer, adapted, prompt, max_length
):
    """Greedy decoding without a KV cache must be token-identical to greedy decoding
    with one. Recompute is slower, never different."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    hf_output = hf_model.generate(
        input_ids,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        max_length=max_length,
    )
    hf_new_tokens = hf_output[0].tolist()[input_ids.shape[1]:]

    tokenized = TokenizedData(request_id="adapter_test", tokens=input_ids[0].tolist())
    adapter_tokens = await LMEngine(adapted).inference(
        tokenized, do_sample=False, max_length=max_length
    )

    assert adapter_tokens == hf_new_tokens


def test_adapter_exposes_the_config_fields_lm_engine_reads(adapted):
    """LMEngine reaches into `model.config` for exactly these two fields; a missing one
    is an AttributeError at request time rather than at construction time."""
    assert adapted.config.n_positions == 1024
    assert adapted.config.eos_token_id == GPT2_EOS_TOKEN_ID


@pytest.mark.asyncio
async def test_adapter_rejects_max_length_beyond_context(adapted, tokenizer):
    input_ids = tokenizer("Hello ", return_tensors="pt").input_ids
    tokenized = TokenizedData(request_id="too_long", tokens=input_ids[0].tolist())
    with pytest.raises(RuntimeError):
        await LMEngine(adapted).inference(tokenized, do_sample=False, max_length=1025)


def test_adapter_truncates_prefix_to_context_length(hf_model):
    """A caller driving the adapter directly (no LMEngine max_length guard) must not be
    able to push a longer-than-context prefix into the model."""
    adapted = TransformerLMAdapter(_LogitsOnly(hf_model), context_length=16)
    prefix = torch.zeros((1, 16), dtype=torch.long)
    new_token = torch.zeros((1, 1), dtype=torch.long)

    out = adapted(input_ids=new_token, past_key_values=prefix)

    assert out.past_key_values.shape[1] == 16
    assert out.logits.shape[1] == 16


def test_adapter_prefill_then_decode_grows_the_prefix(hf_model):
    adapted = TransformerLMAdapter(_LogitsOnly(hf_model), context_length=1024)
    prompt = torch.zeros((1, 5), dtype=torch.long)

    prefill = adapted(input_ids=prompt, past_key_values=None)
    assert prefill.past_key_values.shape[1] == 5

    step = adapted(input_ids=torch.zeros((1, 1), dtype=torch.long),
                   past_key_values=prefill.past_key_values)
    assert step.past_key_values.shape[1] == 6
