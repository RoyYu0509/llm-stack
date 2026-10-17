"""record_demo.record_trial: the OPEN-loop driver (requests arrive on a clock) and the no-KV-cache seq baseline.

The thing to protect: in an open loop a request's arrival time is its SCHEDULED time. If it were the time the engine
happened to notice the request, a slow engine would hide its own queueing delay -- the one thing a load test
exists to show. CPU, tiny random model.
"""
import asyncio
import sys
import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("cs336_basics.lm", reason="needs the training package")
SERVING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING / "benchmarks"))
import record_demo as rd  # noqa: E402

VOCAB = 64
N_NEW = 6


@pytest.fixture(scope="module")
def model():
    cfg = dict(vocab_size=VOCAB, context_length=64, num_layers=2, d_model=32, num_heads=4, d_ff=64, rope_theta=10000.0)
    return rd.build_model(cfg, "cpu", 0, None, rd.get_attention("naive"))


@pytest.fixture(scope="module")
def adapters():
    return rd.load_adapters(SERVING / "tests" / "m2_adapters.py")


def _reqs(n, prompt_len=12, n_new=N_NEW):
    g = torch.Generator().manual_seed(1)
    return [(f"r{i:02d}", torch.randint(0, VOCAB, (prompt_len,), generator=g).tolist(), n_new) for i in range(n)]


def _run(model, adapters, reqs, mode="cb", concurrency=4, **kw):
    return asyncio.run(rd.record_trial(model, adapters, mode, concurrency, reqs, **kw))


def test_open_loop_arrival_is_the_scheduled_time_and_no_request_is_served_before_it(model, adapters):
    reqs = _reqs(8)
    arrivals = [0.03 * (i + 1) for i in range(8)]
    t = _run(model, adapters, reqs, arrivals=arrivals)
    base = t["sent_at"]["r00"] - arrivals[0]
    for (rid, _, _), a in zip(reqs, arrivals):
        assert t["sent_at"][rid] == pytest.approx(base + a, abs=1e-9), "arrival must be the scheduled instant"
        assert t["tok_at"][rid][0] >= t["sent_at"][rid], f"{rid} got a token before it arrived"
        assert len(t["outputs"][rid]) == N_NEW


def test_a_request_that_waits_behind_the_cap_has_that_wait_counted_in_its_ttft(model, adapters):
    """cap = 1 and everything arrives at once: request k can only enter the engine after request k-1 finished."""
    reqs = _reqs(5)
    t = _run(model, adapters, reqs, concurrency=1, arrivals=[0.001 * (i + 1) for i in range(5)])
    ttft = [t["tok_at"][r][0] - t["sent_at"][r] for r, _, _ in reqs]
    for (a, _, _), (b, _, _) in zip(reqs, reqs[1:]):
        assert t["tok_at"][b][0] >= t["tok_at"][a][-1], "cap=1: the next request must wait for the previous to finish"
    assert ttft[-1] > 3 * ttft[0], f"queueing delay not visible in TTFT: {ttft}"


def test_open_loop_and_closed_loop_generate_the_same_tokens(model, adapters):
    reqs = _reqs(6)
    closed = _run(model, adapters, reqs, concurrency=3)
    opened = _run(model, adapters, reqs, concurrency=3, arrivals=[0.01 * (i + 1) for i in range(6)])
    assert closed["outputs"] == opened["outputs"]


def test_arrival_count_must_match_request_count(model, adapters):
    with pytest.raises(ValueError):
        _run(model, adapters, _reqs(3), arrivals=[0.01, 0.02])


def test_no_kv_cache_recomputes_but_computes_the_same_function(model, adapters):
    reqs = _reqs(4)
    cached = _run(model, adapters, reqs, mode="seq", concurrency=1)
    nocache = _run(model, adapters, reqs, mode="seq", concurrency=1, no_kv_cache=True)
    assert cached["cache_calls"] > 0 and nocache["cache_calls"] == 0, "the switch must change which path ran"
    assert nocache["outputs"] == cached["outputs"], "same greedy tokens: it is the same function, only slower"
    for rid, _, _ in reqs:
        ts = nocache["tok_at"][rid]
        assert len(ts) == N_NEW and ts[0] < ts[-1], "tokens must become visible one by one, or TTFT would equal E2E"


def test_guard_aborts_when_the_arm_label_and_the_path_that_ran_disagree():
    args = types.SimpleNamespace(attn="naive", no_kv_cache=True, compile_decode=False)
    attn = rd.AttnCounter(lambda *a, **k: None)
    with pytest.raises(SystemExit, match="no KV cache"):
        rd.guard(args, attn, dict(cache_calls=3))
    args.no_kv_cache = False
    with pytest.raises(SystemExit, match="never called forward_with_cache"):
        rd.guard(args, attn, dict(cache_calls=0))
    rd.guard(args, attn, dict(cache_calls=5))
