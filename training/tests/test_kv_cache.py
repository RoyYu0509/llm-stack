"""Tests for the inference KV cache (`TransformerLM.forward_with_cache`).

The cache is an optimization, so every test here asserts the same thing from a different
angle: it must change speed and nothing else. `forward` is the trusted reference because it
is the code the model was trained with.

All of these run on CPU with a small model, so they are part of the ordinary suite rather
than something that needs a rented GPU.
"""

import pytest
import torch

from cs336_basics.lm import TransformerLM


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return TransformerLM(
        vocab_size=257, context_length=64, num_layers=3, d_model=64,
        heads_num=4, d_ff=176, theta=10000.0, device="cpu", dtype=torch.float32,
    ).eval()


@torch.no_grad()
def _greedy_recompute(model, prompt, n_new):
    ids, out = prompt.clone(), []
    for _ in range(n_new):
        nxt = torch.argmax(model(ids)[:, -1, :], dim=-1, keepdim=True)
        out.append(int(nxt.item()))
        ids = torch.cat([ids, nxt], dim=1)
    return out


@torch.no_grad()
def _greedy_cached(model, prompt, n_new):
    logits, past = model.forward_with_cache(prompt, past_kv=None)
    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    out = [int(nxt.item())]
    for _ in range(n_new - 1):
        logits, past = model.forward_with_cache(nxt, past_kv=past)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        out.append(int(nxt.item()))
    return out


@torch.no_grad()
def test_prefill_matches_plain_forward(model):
    """With an empty cache, forward_with_cache is just forward."""
    ids = torch.randint(0, 257, (1, 12))
    ref = model(ids)
    cached, past = model.forward_with_cache(ids, past_kv=None)
    assert torch.allclose(ref, cached, atol=1e-5)
    assert len(past) == len(model.tf_layers)


@torch.no_grad()
def test_greedy_decoding_is_token_identical(model):
    """The property that actually matters: small logit differences are tolerable, a
    different argmax is not."""
    prompt = torch.randint(0, 257, (1, 8))
    assert _greedy_cached(model, prompt, 20) == _greedy_recompute(model, prompt, 20)


@torch.no_grad()
def test_incremental_logits_match_full_forward_at_each_step(model):
    """Step-by-step, not just at the end: a cache bug that only shows up deep into decoding
    (a stale key, a mis-rotated position) would pass an end-state check on a short run."""
    ids = torch.randint(0, 257, (1, 6))
    logits, past = model.forward_with_cache(ids, past_kv=None)
    for _ in range(10):
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        logits, past = model.forward_with_cache(nxt, past_kv=past)
        reference = model(ids)[:, -1, :]
        assert torch.allclose(reference, logits[:, -1, :], atol=1e-4)


@torch.no_grad()
def test_cache_grows_by_one_key_per_step(model):
    ids = torch.randint(0, 257, (1, 5))
    _, past = model.forward_with_cache(ids, past_kv=None)
    assert past[0][0].shape[-2] == 5
    _, past = model.forward_with_cache(torch.randint(0, 257, (1, 1)), past_kv=past)
    assert past[0][0].shape[-2] == 6


@torch.no_grad()
def test_rejects_decoding_past_the_context_length(model):
    """Positions beyond the context length have no RoPE table entry; fail loudly rather
    than index off the end or silently wrap."""
    full = torch.randint(0, 257, (1, 64))  # exactly context_length

    # Filling the context exactly is allowed...
    _, past = model.forward_with_cache(full, past_kv=None)
    assert past[0][0].shape[-2] == 64

    # ...and the very next token is the one that must be refused.
    with pytest.raises(RuntimeError):
        model.forward_with_cache(torch.randint(0, 257, (1, 1)), past_kv=past)


@torch.no_grad()
def test_batched_decoding_matches_per_sequence_decoding(model):
    """Cache bookkeeping must be per-sequence; a batch must not leak keys across rows."""
    a = torch.randint(0, 257, (1, 7))
    b = torch.randint(0, 257, (1, 7))
    batched = torch.cat([a, b], dim=0)

    logits, past = model.forward_with_cache(batched, past_kv=None)
    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    logits, _ = model.forward_with_cache(nxt, past_kv=past)
    batched_next = torch.argmax(logits[:, -1, :], dim=-1)

    singles = []
    for seq in (a, b):
        lg, pk = model.forward_with_cache(seq, past_kv=None)
        n = torch.argmax(lg[:, -1, :], dim=-1, keepdim=True)
        lg, _ = model.forward_with_cache(n, past_kv=pk)
        singles.append(int(torch.argmax(lg[:, -1, :], dim=-1).item()))

    assert [int(t) for t in batched_next] == singles
