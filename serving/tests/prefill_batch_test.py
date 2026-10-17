"""Batched prefill: several waiting prompts in ONE forward (right-padded, causal, last real position per row).

Reference answer = what the same prompt gives when prefilled alone (`LMEngine.prefill`), and for the engine, the
tokens an engine WITHOUT grouping generates. The knob (`prefill_batch_tokens`) changes how work is packed, never
what is computed -- the same rule prefill_budget_test.py holds for `max_prefills_per_round`.

CPU + fp32, tiny model. A kernel / speed question (does it actually go faster?) cannot be answered here; that is
the GPU run's job (benchmarks/compare/remote/run_stage_c.sh). What is checked here is that it is CORRECT.
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
from inferenceLM.engine.lm_engine import LMEngine  # noqa: E402
from inferenceLM.engine.transformer_lm_adapter import TransformerLMAdapter  # noqa: E402

VOCAB, CTX = 64, 64
ATOL = 1e-5


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


def _lm_engine(model) -> LMEngine:
    adapter = TransformerLMAdapter(model, context_length=CTX, eos_token_id=-1, vocab_size=VOCAB, use_kv_cache=True)
    return LMEngine(adapter)


LENGTHS = (3, 11, 7, 11, 1, 9)


def test_group_prefill_equals_prefilling_each_prompt_alone(model):
    eng = _lm_engine(model)
    prompts = [_prompt(n, seed=100 + i) for i, n in enumerate(LENGTHS)]
    tokens, pasts = eng.prefill_batch(prompts)
    for i, p in enumerate(prompts):
        tok, past = eng.prefill(torch.tensor([p]), max_length=CTX)
        assert tokens[i] == int(tok), f"row {i} (len {len(p)}): first token differs from the single-prompt prefill"
        for layer, ((k, v), (k1, v1)) in enumerate(zip(pasts[i], past)):
            assert k.shape == k1.shape == (1, 4, len(p), 8), (layer, k.shape, k1.shape)
            assert float((k - k1).abs().max()) < ATOL, f"row {i} layer {layer}: K differs (padding leaked into the cache?)"
            assert float((v - v1).abs().max()) < ATOL, f"row {i} layer {layer}: V differs"


def test_the_check_has_teeth_taking_the_last_padded_position_is_caught(model):
    """If the head ran on position -1 (a pad token for every row but the longest), the shorter rows must change."""
    adapter = TransformerLMAdapter(model, context_length=CTX, eos_token_id=-1, vocab_size=VOCAB, use_kv_cache=True)
    prompts = [_prompt(n, seed=200 + i) for i, n in enumerate((3, 12, 5))]
    ids = torch.zeros(3, 12, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = torch.tensor(p)
    with torch.no_grad():
        right = adapter(ids, last_index=torch.tensor([2, 11, 4])).logits[:, 0].argmax(-1).tolist()
        wrong = adapter(ids, last_index=torch.tensor([11, 11, 11])).logits[:, 0].argmax(-1).tolist()
    assert right[1] == wrong[1], "the longest row has no padding: both must agree there"
    assert right != wrong, "with these seeds the wrong index must change a shorter row's token, or the test proves nothing"


def test_batch_of_one_is_the_same_as_prefill(model):
    eng = _lm_engine(model)
    p = _prompt(9, seed=5)
    (tok,), (past,) = eng.prefill_batch([p])
    ref_tok, ref_past = eng.prefill(torch.tensor([p]), max_length=CTX)
    assert tok == int(ref_tok)
    assert all(float((k - k1).abs().max()) < ATOL for (k, _), (k1, _) in zip(past, ref_past))


# ---------------------------------------------------------------------------------------- the engine
async def _run(model, backend: str, prompts: Dict[str, List[int]], new_tokens: int, budget, n_per_round=None):
    queue: asyncio.Queue = asyncio.Queue()
    store: Dict[str, RequestData] = {}
    adapter = TransformerLMAdapter(model, context_length=CTX, eos_token_id=-1, vocab_size=VOCAB, use_kv_cache=True,
                                   decode_backend=backend, decode_slots=8, decode_max_len=CTX)
    engine = InferenceEngine(adapter, queue, store, max_prefills_per_round=n_per_round)
    engine.prefill_batch_tokens = budget
    prefill_forwards: List[int] = []
    hook = model.head.register_forward_hook(
        lambda _m, inp, _out: prefill_forwards.append(inp[0].shape[0]) if inp[0].shape[1] != 1 or backend == "ragged" else None)
    try:
        for rid, p in prompts.items():
            store[rid] = RequestData(request_id=rid, user_id="u", timestamp=time.time(), prompt_text="",
                                     max_token_length=len(p) + new_tokens)
            queue.put_nowait(TokenizedData(request_id=rid, tokens=list(p)))
        engine.run()
        await asyncio.wait_for(engine.shutdown(draining=True), timeout=60)
    finally:
        hook.remove()
    assert all(store[r].status == RequestStatus.DONE for r in prompts), {r: store[r].status for r in prompts}
    return {r: list(store[r].generated_tokens) for r in prompts}, engine


PROMPTS = {f"r{i}": _prompt(n, seed=300 + i) for i, n in enumerate((4, 11, 7, 2, 9, 13))}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["ragged", "prealloc"])
@pytest.mark.parametrize("budget", [32, 200])
async def test_engine_outputs_do_not_depend_on_grouping(model, backend, budget):
    base, _ = await _run(model, backend, PROMPTS, new_tokens=7, budget=None)
    grouped, _ = await _run(model, backend, PROMPTS, new_tokens=7, budget=budget)
    assert grouped == base, f"prefill_batch_tokens={budget} changed what the engine generates ({backend})"


@pytest.mark.asyncio
async def test_group_sizes_follow_the_padded_token_budget(model):
    prompts = {f"r{i}": _prompt(n, seed=400 + i) for i, n in enumerate((10, 10, 10, 10, 10))}
    queue, store = asyncio.Queue(), {}
    adapter = TransformerLMAdapter(model, context_length=CTX, eos_token_id=-1, vocab_size=VOCAB, use_kv_cache=True)
    engine = InferenceEngine(adapter, queue, store)
    for rid, p in prompts.items():
        store[rid] = RequestData(request_id=rid, user_id="u", timestamp=time.time(), prompt_text="",
                                 max_token_length=len(p) + 2)
        engine._in_flight[rid] = type("R", (), {"prompt": p})()
    rids = list(prompts)
    engine.prefill_batch_tokens = None
    assert engine._prefill_groups(rids) == [[r] for r in rids], "None = original behaviour: one forward per request"
    engine.prefill_batch_tokens = 30                        # 3 x 10 = 30 fits, 4 x 10 does not
    assert [len(g) for g in engine._prefill_groups(rids)] == [3, 2]
    engine.prefill_batch_tokens = 5                         # a prompt longer than the budget still gets its own forward
    assert [len(g) for g in engine._prefill_groups(rids)] == [1] * 5
    engine._in_flight["r2"].prompt = _prompt(25, seed=1)    # padded size counts the LONGEST prompt in the group
    engine.prefill_batch_tokens = 40
    assert engine._prefill_groups(rids) == [["r0", "r1"], ["r2"], ["r3", "r4"]]


@pytest.mark.asyncio
async def test_grouping_runs_fewer_prefill_forwards_than_requests(model):
    """Six prompts that all fit in one padded group: ONE head call with a batch of six, not six head calls."""
    calls: List[tuple] = []
    hook = model.head.register_forward_hook(lambda _m, inp, _o: calls.append(tuple(inp[0].shape[:2])))
    try:
        await _run(model, "ragged", PROMPTS, new_tokens=1, budget=10_000)
    finally:
        hook.remove()
    assert calls == [(6, 1)], f"expected one grouped prefill (6 rows, last position only), got {calls}"


@pytest.mark.asyncio
async def test_a_failing_group_falls_back_to_one_by_one(model, monkeypatch):
    calls = {"group": 0}
    real = LMEngine.prefill_batch

    def broken(self, prompts):
        calls["group"] += 1
        raise RuntimeError("simulated: the group does not fit in memory")

    monkeypatch.setattr(LMEngine, "prefill_batch", broken)
    out, _ = await _run(model, "ragged", PROMPTS, new_tokens=5, budget=10_000)
    monkeypatch.setattr(LMEngine, "prefill_batch", real)
    ref, _ = await _run(model, "ragged", PROMPTS, new_tokens=5, budget=None)
    assert calls["group"] >= 1 and out == ref, "a failed group must be redone request by request, with the same result"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["ragged", "prealloc"])
async def test_one_prefill_group_per_round_and_a_decode_step_after_every_group(model, backend, monkeypatch):
    """Six prompts of 10 tokens with a budget of 25 padded tokens -> groups of 2. The engine must run
    P(2) [no decoders yet], then P(2) + D, P(2) + D ... : never a second prefill forward before the decode step of
    the round, and the requests already decoding keep advancing while the rest of the queue waits its turn."""
    events: List[str] = []
    real_pb, real_db = LMEngine.prefill_batch, LMEngine.decode_batch

    def pb(self, prompts):
        events.append(f"P{len(prompts)}")
        return real_pb(self, prompts)

    def db(self, batch, latest, *a, **k):
        events.append(f"D{len(latest)}")
        return real_db(self, batch, latest, *a, **k)

    monkeypatch.setattr(LMEngine, "prefill_batch", pb)
    monkeypatch.setattr(LMEngine, "decode_batch", db)
    prompts = {f"r{i}": _prompt(10, seed=500 + i) for i in range(6)}
    out, engine = await _run(model, backend, prompts, new_tokens=4, budget=25)
    seen = list(events)                       # the reference run below decodes too: do not let it into the schedule
    ref, _ = await _run(model, backend, prompts, new_tokens=4, budget=None)
    events = seen
    assert out == ref, "alternating prefill and decode must not change what is generated"
    # round 1: P(r0,r1) -- nothing is decoding yet.  round 2: P(r2,r3) + D(r0,r1).  round 3: P(r4,r5) + D(r0..r3).
    # then the batch drains: D6 (all), D4, D2 as the 4-token requests finish in the order they were prefilled
    assert events == ["P2", "P2", "D2", "P2", "D4", "D6", "D4", "D2"], events
    assert engine.prefill_group_sizes == [2, 2, 2], engine.prefill_group_sizes