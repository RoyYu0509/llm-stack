"""max_prefills_per_round (N): how many prefills run before one batched decode.

A tiny fake model replaces the real one so the SCHEDULE can be asserted exactly (which forward ran
in which round, over which requests) without GPU, downloads or timing.

Greedy rule of the fake model: the next token is (last token + 1) % VOCAB, so the expected output of
every request is known in closed form and must NOT depend on N -- the knob changes when work is
done, never what is computed.
"""
import asyncio
import time
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

from inferenceLM.data.request import RequestData
from inferenceLM.data.request_status import RequestStatus
from inferenceLM.data.tokenized_data import TokenizedData
from inferenceLM.engine.inference_engine import InferenceEngine

VOCAB = 50
NEW_TOKENS = 4


class FakeBatch:
    """Same surface the engine uses on RaggedDecodeBatch: add / remove / in / step."""

    def __init__(self):
        self.rows: Dict[str, bool] = {}

    def add(self, rid, past):
        self.rows[rid] = True

    def remove(self, rid):
        self.rows.pop(rid, None)

    def __contains__(self, rid):
        return rid in self.rows

    def step(self, latest_tokens: Dict[str, int]):
        # the engine must decode exactly the requests that hold a KV row -- never a waiting one
        assert set(latest_tokens) == set(self.rows), (set(latest_tokens), set(self.rows))
        out = {}
        for rid, tok in latest_tokens.items():
            v = torch.zeros(VOCAB)
            v[(tok + 1) % VOCAB] = 1.0
            out[rid] = v
        return out


class FakeModel:
    supports_batched_decode = True

    def __init__(self):
        self.config = SimpleNamespace(n_positions=64, eos_token_id=-1)  # -1: EOS never generated

    def eval(self):
        return self

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        last = int(input_ids[0, -1])
        logits = torch.zeros(1, input_ids.shape[1], VOCAB)
        logits[0, -1, (last + 1) % VOCAB] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=("kv", last))

    def new_decode_batch(self):
        return FakeBatch()


def _expected(prompt: List[int]) -> List[int]:
    out, tok = [], prompt[-1]
    for _ in range(NEW_TOKENS):
        tok = (tok + 1) % VOCAB
        out.append(tok)
    return out


def _submit(queue: asyncio.Queue, store: Dict[str, RequestData], n: int) -> Dict[str, List[int]]:
    prompts = {}
    for i in range(n):
        rid, prompt = f"r{i:02d}", [5 + 3 * i, 6 + 3 * i, 7 + 3 * i]
        prompts[rid] = prompt
        store[rid] = RequestData(request_id=rid, user_id="u", timestamp=time.time(), prompt_text="",
                                 max_token_length=len(prompt) + NEW_TOKENS)
        queue.put_nowait(TokenizedData(request_id=rid, tokens=prompt))
    return prompts


def _spy(engine: InferenceEngine) -> List[str]:
    """Record the forwards in the order they run: 'P:<last prompt token>' / 'D:<rids>'."""
    events: List[str] = []
    prefill, decode_batch = engine.lm_engine.prefill, engine.lm_engine.decode_batch

    def p(input_ids, *a, **k):
        events.append(f"P:{int(input_ids[0, -1])}")
        return prefill(input_ids, *a, **k)

    def d(batch, latest_tokens, *a, **k):
        events.append("D:" + ",".join(sorted(latest_tokens)))
        return decode_batch(batch, latest_tokens, *a, **k)

    engine.lm_engine.prefill, engine.lm_engine.decode_batch = p, d
    return events


async def _run(n_requests: int, n_prefills):
    queue: asyncio.Queue = asyncio.Queue()
    store: Dict[str, RequestData] = {}
    prompts = _submit(queue, store, n_requests)      # everything is queued before the loop starts
    engine = InferenceEngine(FakeModel(), queue, store, max_prefills_per_round=n_prefills)
    events = _spy(engine)
    engine.run()
    await asyncio.wait_for(engine.shutdown(draining=True), timeout=10)
    return engine, store, prompts, events


@pytest.mark.asyncio
async def test_none_prefills_all_before_the_first_decode():
    """Original behaviour: 5 prefills, then one decode over all 5."""
    _, _, _, events = await _run(5, None)
    assert events[:5] == [f"P:{7 + 3 * i}" for i in range(5)], events
    assert events[5] == "D:r00,r01,r02,r03,r04", events


@pytest.mark.asyncio
async def test_n_limits_prefills_between_decodes():
    """N=2, 5 requests -> rounds: [P P] [P P D(r00,r01)] [P D(r00..r03)] [D(all 5)] ..."""
    _, _, _, events = await _run(5, 2)
    assert events[:7] == [
        "P:7", "P:10",                              # round 1: nothing is ready to decode yet
        "P:13", "P:16", "D:r00,r01",                # round 2: only the two prefilled earlier decode
        "P:19", "D:r00,r01,r02,r03",                # round 3
    ], events
    assert events[7] == "D:r00,r01,r02,r03,r04", events


@pytest.mark.asyncio
async def test_n_one_never_decodes_a_request_without_kv():
    """N=1 is the extreme interleave; FakeBatch.step asserts decoders == rows holding a KV."""
    engine, store, _, events = await _run(4, 1)
    assert [e for e in events[:4]] == ["P:7", "P:10", "D:r00", "P:13"], events
    assert all(r.status == RequestStatus.DONE for r in store.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("n_prefills", [None, 1, 2, 3, 100])
async def test_outputs_do_not_depend_on_n(n_prefills):
    """The knob moves work in time; it must not change any generated token."""
    _, store, prompts, _ = await _run(7, n_prefills)
    for rid, prompt in prompts.items():
        assert store[rid].status == RequestStatus.DONE
        assert store[rid].generated_tokens == _expected(prompt), (rid, n_prefills)


@pytest.mark.asyncio
async def test_every_request_is_prefilled_exactly_once():
    _, _, prompts, events = await _run(9, 2)
    prefills = [e for e in events if e.startswith("P:")]
    assert sorted(prefills) == sorted(f"P:{p[-1]}" for p in prompts.values())


def test_invalid_budget_is_rejected():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            InferenceEngine(FakeModel(), asyncio.Queue(), {}, max_prefills_per_round=bad)


@pytest.mark.asyncio
async def test_nondraining_shutdown_drops_requests_still_waiting_for_prefill():
    """Waiting requests are in_flight too: a non-draining shutdown must fail them and empty the FIFO."""
    queue: asyncio.Queue = asyncio.Queue()
    store: Dict[str, RequestData] = {}
    _submit(queue, store, 6)
    engine = InferenceEngine(FakeModel(), queue, store, max_prefills_per_round=1)
    engine.run()
    await asyncio.sleep(0)                              # let the first round run: 1 prefilled, 5 waiting
    await asyncio.wait_for(engine.shutdown(draining=False), timeout=10)
    assert len(engine._prefill_fifo) == 0
    assert all(r.status in (RequestStatus.DONE, RequestStatus.FAILED) for r in store.values())


def test_last_only_logits_match_the_last_row_of_full_prefill():
    """prefill with last_only=True must give the same next-token logits as slicing the full (S, V) logits."""
    TransformerLM = pytest.importorskip("cs336_basics.lm").TransformerLM
    torch.manual_seed(0)
    m = TransformerLM(vocab_size=64, context_length=64, num_layers=2, d_model=32, heads_num=4, d_ff=64,
                      theta=10000.0, device="cpu", dtype=torch.float32).eval()
    with torch.no_grad():
        for n in (3, 17, 40):
            ids = torch.randint(0, 64, (1, n))
            full, kv_full = m.forward_with_cache(ids)
            last, kv_last = m.forward_with_cache(ids, last_only=True)
            assert last.shape == (1, 1, 64)
            assert float((full[:, -1:] - last).abs().max()) < 1e-5
            assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(kv_full, kv_last)), \
                "the KV cache must not depend on last_only"
