"""Closed-loop client tests: a fake SSE server with KNOWN timings checks that TTFT / TPOT /
concurrency semantics are measured correctly, so a number from the real systems means something.
CPU only.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import closed_loop as cl  # noqa: E402

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

PREFILL_S = 0.05
TOKEN_S = 0.01


def _make_app(state, short_after: int | None = None):
    async def generate(request):
        state["inflight"] += 1
        state["total"] = state.get("total", 0) + 1
        state["max_inflight"] = max(state["max_inflight"], state["inflight"])
        if state.get("expect"):   # barrier: hold every response until `expect` requests are in flight (or give up after 5 s),
            gate = state.setdefault("gate", asyncio.Event())   # so the test does not depend on how fast the OS schedules 150 connects
            if state["inflight"] >= state["expect"]:
                gate.set()
            try:
                await asyncio.wait_for(gate.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
        body = await request.json()
        n = body["sampling_params"]["max_new_tokens"]
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            await asyncio.sleep(PREFILL_S)
            limit = n if short_after is None else short_after
            for i in range(1, limit + 1):
                await resp.write(
                    f'data: {json.dumps({"text": "x" * i, "meta_info": {"completion_tokens": i}})}\n\n'.encode())
                await asyncio.sleep(TOKEN_S)
            await resp.write(b"data: [DONE]\n\n")
        finally:
            state["inflight"] -= 1
        return resp

    app = web.Application()
    app.router.add_post("/generate", generate)
    return app


async def _serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def test_ttft_tpot_and_concurrency_cap():
    async def go():
        state = dict(inflight=0, max_inflight=0)
        runner, url = await _serve(_make_app(state))
        client = cl.SGLangClient(url, "native")
        prompts = [[1, 2, 3]] * 8
        traces, makespan = await cl.run_closed_loop(client, prompts, new_tokens=10, concurrency=4)
        await client.close()
        await runner.cleanup()
        return state, traces, makespan

    state, traces, makespan = asyncio.run(go())
    assert len(traces) == 8 and all(t.ok for t in traces), "all requests should complete with 10 tokens"
    assert all(len(t.token_times) == 10 for t in traces)
    assert state["max_inflight"] <= 4, (
        f"client exceeded the concurrency cap: server saw {state['max_inflight']} in flight"
    )
    s = cl.summarize_rep(traces, makespan)
    # TTFT = prefill + first token write; allow generous scheduling slack
    assert PREFILL_S <= s["ttft"]["p50"] < PREFILL_S + 0.08, f"TTFT p50 off: {s['ttft']['p50']}"
    # TPOT ~= per-token delay
    assert TOKEN_S * 0.8 <= s["tpot"]["p50"] < TOKEN_S * 3, f"TPOT p50 off: {s['tpot']['p50']}"
    # two waves of 4 -> makespan is about two requests' worth of time, not eight
    one = PREFILL_S + 10 * TOKEN_S
    assert makespan < 3.2 * one, f"closed loop did not overlap requests: makespan={makespan:.3f}"


def test_short_stream_is_flagged_failed_not_counted():
    async def go():
        state = dict(inflight=0, max_inflight=0)
        runner, url = await _serve(_make_app(state, short_after=3))
        client = cl.SGLangClient(url, "native")
        out = await cl.run_closed_loop(client, [[1]] * 3, new_tokens=10, concurrency=3)
        await client.close()
        await runner.cleanup()
        return out

    traces, makespan = asyncio.run(go())
    assert all(not t.ok and t.status == "failed" for t in traces), (
        "a stream that ended early (EOS not ignored) must be marked failed; counting it would "
        "inflate throughput relative to systems that generated the full length"
    )
    s = cl.summarize_rep(traces, makespan)
    assert s["n_failed"] == 3 and s["n_done"] == 0


def test_workload_roundtrip(tmp_path):
    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps(dict(
        meta=dict(new_tokens=64, prompt_len_range=[1, 2], seed=0, n_requests=2),
        requests=[dict(id="r00", prompt_len=2, prompt_ids=[5, 6], submit_t=0.0, tokens=[]),
                  dict(id="r01", prompt_len=1, prompt_ids=[7], submit_t=0.0, tokens=[])])))
    wl = cl.make_workload(rec, tmp_path / "wl.json")
    assert wl["prompts"] == [[5, 6], [7]] and wl["new_tokens"] == 64
    assert cl.load_workload(tmp_path / "wl.json")["prompts"] == [[5, 6], [7]]


import os  # noqa: E402

# Recordings of the 10-04 closed-loop sweep on the 3090 (kept outside this repo in the manager
# workspace). Point ENGINE_RECORDINGS at the directory holding demo_c32_{cb,seq}_r{1,2,3}.json.
SWEEP = Path(os.environ.get(
    "ENGINE_RECORDINGS",
    Path(__file__).resolve().parents[5] / "docs" / "videos" / "systems" / "data" / "sweep"))


@pytest.mark.skipif(not (SWEEP / "demo_c32_cb_r1.json").exists(), reason="recordings not on this machine")
def test_ingest_reproduces_the_15x_numbers(tmp_path):
    cb = [SWEEP / f"demo_c32_cb_r{i}.json" for i in (1, 2, 3)]
    seq = [SWEEP / f"demo_c32_seq_r{i}.json" for i in (1, 2, 3)]

    def mean_tps(paths):
        vals = []
        for p in paths:
            tr = cl.traces_from_recording(json.loads(p.read_text()))
            ms = max(t.token_times[-1] for t in tr)
            vals.append(cl.summarize_rep(tr, ms)["output_token_throughput"])
        return sum(vals) / len(vals)

    a, b = mean_tps(cb), mean_tps(seq)
    assert a == pytest.approx(427.4, rel=0.01), f"cb throughput {a:.1f} != 427.4 (analyze_3090_closed_loop.txt)"
    assert b == pytest.approx(28.5, rel=0.01), f"seq throughput {b:.1f} != 28.5"
    assert a / b == pytest.approx(14.99, rel=0.02)


def test_hf_client_streams_exact_token_count_and_serialises(tmp_path):
    """HFGenerateClient on a tiny random Llama (CPU): exact new_tokens per request, and requests
    are served strictly one at a time (the 'generate() in a loop' baseline)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=4, max_position_embeddings=128,
                                   eos_token_id=1, bos_token_id=1)
    transformers.LlamaForCausalLM(cfg).save_pretrained(tmp_path / "m")

    async def go():
        client = cl.HFGenerateClient(str(tmp_path / "m"), device="cpu", dtype="float32", tf32=False)
        prompts = [[2, 3, 4, 5]] * 5
        return await cl.run_closed_loop(client, prompts, new_tokens=6, concurrency=3)

    traces, _ = asyncio.run(go())
    assert len(traces) == 5
    assert all(t.ok and len(t.token_times) == 6 for t in traces), (
        "min_new_tokens must force exactly 6 tokens even if EOS is sampled; got "
        f"{[len(t.token_times) for t in traces]}"
    )
    spans = sorted((t.token_times[0], t.token_times[-1]) for t in traces)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0 + 1e-6, "HF requests overlapped; the baseline must be one request at a time"


def test_concurrency_above_100_is_really_reached():
    """aiohttp's default connector caps a session at 100 connections: with C=150 the client used to hold 150
    'slots' but put at most 100 requests on the wire (the wait was counted as TTFT)."""
    async def go():
        state = dict(inflight=0, max_inflight=0, expect=150)   # barrier: see _make_app
        runner, url = await _serve(_make_app(state))
        client = cl.SGLangClient(url, "native")
        try:
            prompts = [[1, 2, 3]] * 150
            traces, _ = await cl.run_closed_loop(client, prompts, 2, 150)
        finally:
            await client.close()
            await runner.cleanup()
        return state, traces

    state, traces = asyncio.run(go())
    assert state["max_inflight"] == 150, f"server saw at most {state['max_inflight']} in flight with C=150"
    assert all(t.ok for t in traces)


def test_mixed_output_lengths_are_honoured_per_request():
    async def go():
        state = dict(inflight=0, max_inflight=0)
        runner, url = await _serve(_make_app(state))
        client = cl.SGLangClient(url, "native")
        try:
            prompts = [[1, 2, 3]] * 6
            lens = [1, 2, 3, 4, 5, 6]
            traces, _ = await cl.run_closed_loop(client, prompts, lens, 3)
        finally:
            await client.close()
            await runner.cleanup()
        return traces, lens

    traces, lens = asyncio.run(go())
    got = sorted(len(t.token_times) for t in traces)
    assert got == lens and all(t.ok for t in traces), (got, lens)


def test_n_min_is_a_floor_for_low_concurrency_and_n_mult_scales_above_it(tmp_path):
    """requests per level = max(n_min, n_mult * C), capped by the workload size: C=1 -> 32, C=8 -> 32, C=16 -> 48."""
    wl = tmp_path / "w.json"
    wl.write_text(json.dumps(dict(prompts=[[1, 2, 3]] * 200, new_tokens=2)))

    async def go(conc):
        state = dict(inflight=0, max_inflight=0)
        runner, url = await _serve(_make_app(state))
        args = cl.argparse.Namespace(backend="sglang", url=url, endpoint="native", workload=wl, n=0, warmup=0,
                                     concurrency=[conc], n_mult=3, n_min=32, warm_mult=0, reps=1, system="x",
                                     notes="", out=tmp_path / f"o{conc}.csv", arrival_rate=None, cb_opts="")
        try:
            await cl.amain_run(args)
        finally:
            await runner.cleanup()
        return state["total"]

    assert asyncio.run(go(1)) == 32 and asyncio.run(go(8)) == 32 and asyncio.run(go(16)) == 48


# ---------------------------------------------------------------------------------------------------------------
# open loop
# ---------------------------------------------------------------------------------------------------------------
def test_open_loop_sends_on_the_clock_even_when_earlier_requests_are_still_running():
    """6 requests 20 ms apart, each takes >= 150 ms: a closed loop at C=1 would never have two in flight."""
    async def go():
        state = dict(inflight=0, max_inflight=0)
        runner, url = await _serve(_make_app(state))
        client = cl.SGLangClient(url, "native")
        try:
            arr = [0.02 * (i + 1) for i in range(6)]
            traces, ms = await cl.run_open_loop(client, [[1, 2, 3]] * 6, 10, arr)
        finally:
            await client.close()
            await runner.cleanup()
        return state, traces, ms, arr

    state, traces, ms, arr = asyncio.run(go())
    assert state["max_inflight"] >= 4, f"requests were not sent on the clock: only {state['max_inflight']} overlapped"
    assert sorted(t.arrival for t in traces) == pytest.approx(arr), "arrival must be the scheduled instant"
    assert all(t.ok and len(t.token_times) == 10 for t in traces)
    assert all(t.token_times[0] - t.arrival >= PREFILL_S * 0.9 for t in traces), "a token cannot precede arrival + prefill"
    assert ms == pytest.approx(max(t.token_times[-1] for t in traces) - arr[0])


def test_open_loop_inflight_cap_queues_the_excess_and_the_wait_counts_as_ttft():
    async def go():
        state = dict(inflight=0, max_inflight=0)
        runner, url = await _serve(_make_app(state))
        client = cl.SGLangClient(url, "native")
        try:
            traces, _ = await cl.run_open_loop(client, [[1, 2, 3]] * 4, 10, [0.001 * (i + 1) for i in range(4)], max_inflight=1)
        finally:
            await client.close()
            await runner.cleanup()
        return state, sorted(traces, key=lambda t: t.arrival)

    state, traces = asyncio.run(go())
    assert state["max_inflight"] == 1
    ttft = [t.token_times[0] - t.arrival for t in traces]
    one = PREFILL_S + 10 * TOKEN_S
    assert ttft[-1] > 2.5 * one, f"the 4th request waited behind 3 others, its TTFT must show it: {ttft}"


def test_open_loop_short_stream_is_failed_and_arrival_count_must_match(tmp_path):
    async def go():
        state = dict(inflight=0, max_inflight=0)
        runner, url = await _serve(_make_app(state, short_after=3))
        client = cl.SGLangClient(url, "native")
        try:
            traces, _ = await cl.run_open_loop(client, [[1]] * 2, 10, [0.0, 0.01])
            with pytest.raises(ValueError):
                await cl.run_open_loop(client, [[1]] * 2, 10, [0.0])
        finally:
            await client.close()
            await runner.cleanup()
        return traces

    assert all(t.status == "failed" for t in asyncio.run(go()))


def test_open_rates_write_experiment_I2_rows_tagged_with_the_rate(tmp_path):
    wl = tmp_path / "w.json"
    wl.write_text(json.dumps(dict(prompts=[[1, 2, 3]] * 5, new_tokens_list=[2, 3, 4], new_tokens=3)))

    async def go():
        state = dict(inflight=0, max_inflight=0)
        runner, url = await _serve(_make_app(state))
        args = cl.argparse.Namespace(backend="sglang", url=url, endpoint="native", workload=wl, n=0, warmup=0,
                                     concurrency=[1], n_mult=0, n_min=0, warm_mult=0, reps=2, system="sys_x",
                                     notes="hello", out=tmp_path / "o.csv", arrival_rate=[40.0, 80.0], arrival_cv=1.0,
                                     duration=0.5, warmup_s=0.1, max_inflight=0, seed=0, slo_ttft_s=1.0,
                                     slo_tpot_ms=50.0, cb_opts="")
        try:
            await cl.amain_run(args)
        finally:
            await runner.cleanup()
        return state["total"]

    total = asyncio.run(go())
    import csv as _csv
    rows = list(_csv.DictReader(open(tmp_path / "o.csv")))
    assert {r["experiment"] for r in rows} == {"I2"} and {r["concurrency"] for r in rows} == {"0"}
    variants = {r["variant"] for r in rows}
    assert any("open_loop;rate=40;cv=1;duration=0.5s;cap=none" in v for v in variants), variants
    assert any("rate=80" in v for v in variants), variants
    assert "mixed" in next(iter(variants)), "mixed output lengths must be visible in the variant"
    assert all("slo ttft<=1.0s tpot<=50.0ms" in r["notes"] for r in rows)
    assert total > 2 * (20 + 40), "warm-up + 2 timed reps must all have been sent"


class _FakeManager:
    """Duck-typed ContinuousBatchingManager: streams CUMULATIVE generated_tokens from a thread, like the real one."""

    def __init__(self, emit_n=None, error=None, alive=True):
        from types import SimpleNamespace

        self.handlers, self.emit_n, self.error = {}, emit_n, error
        self.calls = []                                     # order of add_request / start
        self._generation_thread = SimpleNamespace(is_alive=lambda: self.alive)
        self.alive = alive

    def start(self):
        self.calls.append("start")

    def register_result_handler(self, rid, cb):
        self.handlers[rid] = (cb, asyncio.get_running_loop())

    def add_request(self, ids, request_id, max_new_tokens, streaming, eos_token_id):
        assert streaming and eos_token_id == -1, "EOS must be disabled and streaming on, or counts/timings are wrong"
        self.calls.append("add_request")
        if self.emit_n == 0 and not self.error:
            return                                          # a manager that swallows requests (dead thread)
        import threading
        import time as _t
        from types import SimpleNamespace

        cb, loop = self.handlers[request_id]
        n = max_new_tokens if self.emit_n is None else self.emit_n

        def run():
            for i in range(1, n + 1):
                _t.sleep(0.005)
                fin = i == n
                out = SimpleNamespace(generated_tokens=list(range(100, 100 + i)), error=None, is_finished=lambda f=fin: f)
                loop.call_soon_threadsafe(cb, out)
            if self.error:
                loop.call_soon_threadsafe(cb, SimpleNamespace(generated_tokens=[], error=self.error, is_finished=lambda: True))
        threading.Thread(target=run, daemon=True).start()

    def stop(self, block=True):
        self.stopped = True


def _hf_cb(manager):
    c = object.__new__(cl.HFContinuousClient)   # skip the model load: only the streaming logic is under test
    c.manager, c._n, c._started = manager, 0, False
    c.POLL_S = 0.05
    return c


def test_hf_continuous_client_turns_cumulative_results_into_one_token_each():
    async def go():
        c = _hf_cb(_FakeManager())
        return await cl.run_closed_loop(c, [[1, 2]] * 3, [4, 5, 6], 3)

    traces, _ = asyncio.run(go())
    assert sorted(len(t.token_times) for t in traces) == [4, 5, 6] and all(t.ok for t in traces)
    assert all(b > a for t in traces for a, b in zip(t.token_times, t.token_times[1:])), "tokens must arrive one by one"


def test_hf_continuous_client_wrong_count_or_error_is_a_failure_not_a_result():
    async def go(mgr):
        return (await cl.run_closed_loop(_hf_cb(mgr), [[1]] * 2, 6, 2))[0]

    short = asyncio.run(go(_FakeManager(emit_n=3)))
    assert all(t.status == "failed" for t in short), "fewer tokens than requested (EOS fired?) must be failed"
    err = asyncio.run(go(_FakeManager(emit_n=0, error="boom")))
    assert all(t.status == "failed" for t in err)


def test_ingest_open_loop_recording_writes_I2_and_refuses_a_closed_loop_one(tmp_path):
    def rec(open_loop):
        meta = dict(concurrency=64, new_tokens=64, open_loop=open_loop, arrival_rate=20.0, arrival_cv=1.0, arrival_duration_s=30.0)
        reqs = [dict(id=f"r{i}", prompt_len=3, submit_t=0.05 * i, tokens=[[0.05 * i + 0.02 + 0.01 * k, 1] for k in range(4)])
                for i in range(5)]
        return dict(meta=meta, requests=reqs)

    good, bad = tmp_path / "g.json", tmp_path / "b.json"
    good.write_text(json.dumps(rec(True)))
    bad.write_text(json.dumps(rec(False)))
    out = tmp_path / "o.csv"
    assert cl.main(["ingest", "--recordings", str(good), "--system", "engine_x", "--open-loop", "--out", str(out)]) == 0
    import csv as _csv
    rows = list(_csv.DictReader(open(out)))
    assert {r["experiment"] for r in rows} == {"I2"} and {r["concurrency"] for r in rows} == {"0"}
    assert any("open_loop;rate=20;cv=1;duration=30s;cap=64" in r["variant"] for r in rows)
    with pytest.raises(SystemExit, match="closed loop"):
        cl.main(["ingest", "--recordings", str(bad), "--system", "engine_x", "--open-loop", "--out", str(tmp_path / "o2.csv")])


def test_open_loop_warmup_may_be_longer_than_the_timed_run(tmp_path):
    """Found by the stage-D smoke on the GPU box: a 3 s run with the default 5 s warm-up asked for more warm-up prompts than the run had."""
    wl = tmp_path / "w.json"
    wl.write_text(json.dumps(dict(prompts=[[1, 2, 3]] * 3, new_tokens_list=[2, 3], new_tokens=2)))

    async def go():
        state = dict(inflight=0, max_inflight=0)
        runner, url = await _serve(_make_app(state))
        args = cl.argparse.Namespace(backend="sglang", url=url, endpoint="native", workload=wl, n=0, warmup=0,
                                     concurrency=[1], n_mult=0, n_min=0, warm_mult=0, reps=1, system="s",
                                     notes="", out=tmp_path / "o.csv", arrival_rate=[20.0], arrival_cv=1.0,
                                     duration=0.2, warmup_s=2.0, max_inflight=0, seed=0, slo_ttft_s=1.0,
                                     slo_tpot_ms=50.0, cb_opts="")
        try:
            await cl.amain_run(args)
        finally:
            await runner.cleanup()
        return state["total"]

    assert asyncio.run(go()) == 40 + 4, "40 warm-up requests (2 s x 20/s) plus the 4 timed ones"


def test_hf_continuous_client_starts_the_thread_only_after_the_first_request_is_queued():
    """With async batching the HF generation thread dies at once if it finds no request when it bootstraps."""
    async def go():
        m = _FakeManager()
        c = _hf_cb(m)
        await cl.run_closed_loop(c, [[1]] * 2, 3, 2)
        return m.calls

    calls = asyncio.run(go())
    assert calls[0] == "add_request" and calls.index("start") == 1 and calls.count("start") == 1, calls


def test_hf_continuous_client_fails_fast_instead_of_hanging_when_the_generation_thread_is_dead():
    async def go():
        m = _FakeManager(emit_n=0, alive=False)   # never answers, thread not alive
        c = _hf_cb(m)
        return await asyncio.wait_for(cl.run_closed_loop(c, [[1]] * 2, 4, 2), timeout=10)

    traces, _ = asyncio.run(go())
    assert all(t.status == "failed" for t in traces), "a dead manager must surface as failed requests, not as a hang"
