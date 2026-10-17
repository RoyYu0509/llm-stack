"""PreallocDecodeBatch: pre-allocated KV slots, in-place append, per-row lengths.

Reference answer = `model(whole sequence so far)` -- the training forward, no cache, no batch (same rule as
batched_decode_test.py). CPU + fp32 + the pure-PyTorch `reference` attention, so the BOOKKEEPING is what is
tested (slots, lengths, positions, order, reuse); the FlashAttention kernel itself is checked against the
same reference on the GPU box (benchmarks/check_fa2_decode.py).
"""
import asyncio
import time
from typing import Dict, List

import pytest
import torch

TransformerLM = pytest.importorskip("cs336_basics.lm", reason="needs the training package").TransformerLM

from inferenceLM.data.request import RequestData  # noqa: E402
from inferenceLM.data.request_status import RequestStatus  # noqa: E402
from inferenceLM.data.tokenized_data import TokenizedData  # noqa: E402
from inferenceLM.engine.inference_engine import InferenceEngine  # noqa: E402
from inferenceLM.engine.prealloc_batch import PreallocDecodeBatch  # noqa: E402
from inferenceLM.engine.transformer_lm_adapter import TransformerLMAdapter  # noqa: E402

VOCAB, CTX = 64, 64
ATOL = 1e-4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = TransformerLM(vocab_size=VOCAB, context_length=CTX, num_layers=2, d_model=32, heads_num=4, d_ff=64,
                      theta=10000.0, device="cpu", dtype=torch.float32)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _prompt(n: int, seed: int) -> List[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


def _prefill(model, prompt):
    logits, past = model.forward_with_cache(torch.tensor([prompt]))
    return int(logits[0, -1].argmax()), past


def _ref_logits(model, seq):
    with torch.no_grad():
        return model(torch.tensor([seq]))[0, -1]


def _new_batch(model, slots=4, max_len=48):
    return PreallocDecodeBatch(model, n_slots=slots, max_len=max_len, dtype=torch.float32, device="cpu")


def _drive(model, batch, seqs: Dict[str, List[int]], steps: int):
    """Feed greedy tokens back for `steps` steps; every row's logits must match the full forward."""
    for _ in range(steps):
        out = batch.step({r: s[-1] for r, s in seqs.items()})
        for r, logits in out.items():
            ref = _ref_logits(model, seqs[r])
            err = float((logits - ref).abs().max())
            assert err < ATOL, f"row {r} (len {len(seqs[r])}): max|logit diff| {err:.2e} vs full forward"
            seqs[r].append(int(logits.argmax()))


def test_ragged_rows_match_full_forward(model):
    batch = _new_batch(model)
    seqs = {}
    for name, n in (("a", 5), ("b", 9), ("c", 3)):
        p = _prompt(n, seed=n)
        tok, past = _prefill(model, p)
        seqs[name] = p + [tok]
        batch.add(name, past)
    _drive(model, batch, seqs, steps=6)
    assert [batch._lens_cpu[batch._slot[r]] for r in seqs] == [len(s) - 1 for s in seqs.values()]


def test_remove_and_add_between_steps_reuses_slots(model):
    batch = _new_batch(model, slots=3)
    seqs = {}
    for name, n in (("a", 6), ("b", 4)):
        p = _prompt(n, seed=10 + n)
        tok, past = _prefill(model, p)
        seqs[name], _ = p + [tok], batch.add(name, past)
    _drive(model, batch, seqs, steps=3)
    freed = batch._slot["a"]
    batch.remove("a"); del seqs["a"]
    p = _prompt(8, seed=99)
    tok, past = _prefill(model, p)
    seqs["c"] = p + [tok]; batch.add("c", past)
    assert batch._slot["c"] == freed, "a freed slot must be handed out again"
    _drive(model, batch, seqs, steps=4)


def test_cache_is_allocated_once_and_appended_in_place(model):
    batch = _new_batch(model)
    p = _prompt(6, seed=1)
    tok, past = _prefill(model, p)
    batch.add("a", past)
    ptrs = [(k.data_ptr(), v.data_ptr()) for k, v in zip(batch.k, batch.v)]
    ids = [id(k) for k in batch.k]
    seqs = {"a": p + [tok]}
    _drive(model, batch, seqs, steps=5)
    assert ptrs == [(k.data_ptr(), v.data_ptr()) for k, v in zip(batch.k, batch.v)], "cache was re-created"
    assert ids == [id(k) for k in batch.k]


def test_step_tokens_is_the_batched_argmax_of_step(model):
    batch = _new_batch(model)
    toks = {}
    for name, n in (("a", 5), ("b", 7)):
        p = _prompt(n, seed=20 + n)
        t, past = _prefill(model, p)
        toks[name] = t; batch.add(name, past)
    lens_before = list(batch._lens_cpu)
    got = batch.step_tokens(toks)
    # same state advanced twice would differ, so rebuild the identical batch and compare
    batch2 = _new_batch(model)
    for name, n in (("a", 5), ("b", 7)):
        _, past = _prefill(model, _prompt(n, seed=20 + n))
        batch2.add(name, past)
    logits = batch2.step(toks)
    assert got == {r: int(v.argmax()) for r, v in logits.items()}
    assert batch._lens_cpu != lens_before and all(isinstance(t, int) for t in got.values())


def test_slot_capacity_and_oversized_prompt_are_explicit_errors(model):
    batch = _new_batch(model, slots=1, max_len=16)
    _, past = _prefill(model, _prompt(5, seed=3))
    batch.add("a", past)
    with pytest.raises(RuntimeError, match="no free KV slot"):
        batch.add("b", past)
    batch.remove("a")
    _, long_past = _prefill(model, _prompt(20, seed=4))
    with pytest.raises(RuntimeError, match="does not fit"):
        batch.add("c", long_past)


async def _engine_outputs(model, backend: str, prompts: Dict[str, List[int]], new_tokens: int):
    queue: asyncio.Queue = asyncio.Queue()
    store: Dict[str, RequestData] = {}
    adapter = TransformerLMAdapter(model, context_length=CTX, eos_token_id=-1, vocab_size=VOCAB,
                                   use_kv_cache=True, decode_backend=backend, decode_slots=8, decode_max_len=CTX)
    engine = InferenceEngine(adapter, queue, store)
    for rid, p in prompts.items():
        store[rid] = RequestData(request_id=rid, user_id="u", timestamp=time.time(), prompt_text="",
                                 max_token_length=len(p) + new_tokens)
        queue.put_nowait(TokenizedData(request_id=rid, tokens=list(p)))
    engine.run()
    await asyncio.wait_for(engine.shutdown(draining=True), timeout=60)
    assert all(store[r].status == RequestStatus.DONE for r in prompts)
    return {r: list(store[r].generated_tokens) for r in prompts}


@pytest.mark.asyncio
async def test_engine_prealloc_equals_ragged_equals_full_forward(model):
    prompts = {f"r{i}": _prompt(n, seed=40 + i) for i, n in enumerate((4, 11, 7, 2, 9))}
    ragged = await _engine_outputs(model, "ragged", prompts, new_tokens=8)
    prealloc = await _engine_outputs(model, "prealloc", prompts, new_tokens=8)
    assert prealloc == ragged
    for rid, p in prompts.items():           # and both equal greedy decoding with the training forward
        seq = list(p)
        for _ in range(8):
            seq.append(int(_ref_logits(model, seq).argmax()))
        assert prealloc[rid] == seq[len(p):], rid
