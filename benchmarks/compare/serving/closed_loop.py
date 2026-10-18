"""I1 -- one closed-loop client and one metric definition for every serving system.

Systems (all fed the SAME prompts, token-id for token-id):
  engine_*   the self-written engine: read from the recordings that `serving/benchmarks/
             record_demo.py` / `cb_benchmark.py` already produce (`ingest`)
  hf         HuggingFace `generate()`, one request at a time (the "naive baseline")
  hf_cb      HuggingFace transformers' OWN continuous batching (`init_continuous_batching`, paged KV)
  sglang     an SGLang server over HTTP (native /generate, or OpenAI-compatible /v1/completions)
  vllm       a vLLM server over HTTP (/v1/completions with continuous_usage_stats; see vllm_client.py)

Workload = the recordings' own `prompt_ids` (128 wiki prompts, 256-900 tokens, 64 new tokens,
EOS ignored), extracted once by `make-workload`, so numbers line up with the existing 15x data.

Metric definitions are NOT re-implemented: everything goes through
`serving/benchmarks/serving_metrics.py` (Trace / summarize), the single place the repo defines
TTFT / TPOT / ITL / goodput.

Closed loop at concurrency C: C requests outstanding; a request's `arrival` is the moment it
gets a slot (so queueing inside the client does not count, queueing inside the server does) --
the same convention as `cb_benchmark.py`.

Open loop (`run --arrival-rate R`): requests arrive on a clock (Poisson / bursty gaps from serving/benchmarks/
arrivals.py), whether or not the server keeps up; a request's `arrival` is its SCHEDULED time, so time spent
queueing -- in the server or in an optional client-side in-flight cap -- is part of its TTFT.

"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "serving" / "benchmarks"))
from arrivals import arrival_times, n_for_duration  # noqa: E402
from schema import ResultWriter  # noqa: E402
from serving_metrics import Slo, Trace, summarize  # noqa: E402


# --------------------------------------------------------------------------------------
# workload
# --------------------------------------------------------------------------------------
def make_workload(recording: Path, out: Path) -> dict:
    rec = json.loads(recording.read_text())
    reqs = rec["requests"]
    wl = dict(
        source=str(recording), meta={k: rec["meta"].get(k) for k in
                                     ("new_tokens", "prompt_len_range", "seed", "prompts_npy",
                                      "n_requests", "model_cfg")},
        new_tokens=int(rec["meta"]["new_tokens"]),
        prompts=[r["prompt_ids"] for r in reqs],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(wl))
    return wl


def load_workload(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------------------
# clients: async generator yielding one item per *new token*, at the moment it becomes visible
# --------------------------------------------------------------------------------------
class SGLangClient:
    """SGLang HTTP server. `native`: POST /generate with input_ids, streaming; exact token counts
    from meta_info.completion_tokens. `openai`: POST /v1/completions (one chunk ~ one token)."""

    def __init__(self, url: str, endpoint: str = "native", model: str = "default"):
        self.url, self.endpoint, self.model = url.rstrip("/"), endpoint, model
        self._session = None

    async def _sess(self):
        import aiohttp

        if self._session is None:
            # limit=0: no cap. aiohttp's default connector allows 100 connections, so with C > 100 the semaphore
            # admitted C tasks but only ~100 could be in flight and the wait was counted as TTFT.
            conn = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=3600))
        return self._session

    async def generate(self, prompt_ids: Sequence[int], new_tokens: int) -> AsyncIterator[int]:
        s = await self._sess()
        if self.endpoint == "native":
            url = f"{self.url}/generate"
            body = {"input_ids": list(prompt_ids), "stream": True,
                    "sampling_params": {"temperature": 0.0, "max_new_tokens": new_tokens,
                                        "ignore_eos": True}}
        else:
            url = f"{self.url}/v1/completions"
            body = {"model": self.model, "prompt": list(prompt_ids), "max_tokens": new_tokens,
                    "temperature": 0.0, "stream": True, "ignore_eos": True}
        seen = 0
        async with s.post(url, json=body) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if self.endpoint == "native":
                    n = int(obj.get("meta_info", {}).get("completion_tokens", seen))
                    for _ in range(max(0, n - seen)):
                        yield 0
                    seen = max(seen, n)
                else:
                    choices = obj.get("choices") or []
                    if choices and choices[0].get("text") not in (None, ""):
                        seen += 1
                        yield 0

    async def close(self):
        if self._session is not None:
            await self._session.close()


class HFGenerateClient:
    """`model.generate()` per request, strictly one at a time (no batching): the baseline a user
    gets from `generate()` in a loop. Requests wait on a lock, so server-side queueing is counted."""

    def __init__(self, model_dir: str, device: str = "cuda", dtype: str = "float32", tf32: bool = True,
                 attn: str = "sdpa"):
        import torch
        from transformers import AutoModelForCausalLM

        if tf32:  # the engine recordings ran with matmul_precision "high"
            torch.set_float32_matmul_precision("high")
        self.torch = torch
        self.device = device
        dt = getattr(torch, dtype)
        kw = dict(attn_implementation=attn)   # explicit: the default differs by version
        try:  # transformers >=5 renamed torch_dtype -> dtype
            m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, **kw)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dt, **kw)
        self.model = m.to(device).eval()
        self._lock = threading.Lock()

    def _run(self, prompt_ids, new_tokens, emit):
        torch = self.torch

        class _Streamer:  # duck-typed BaseStreamer: put(value), end()
            def __init__(s):
                s.first = True

            def put(s, value):
                if s.first:  # first call carries the prompt
                    s.first = False
                    return
                emit(int(value.view(-1)[0]))

            def end(s):
                emit(None)

        ids = torch.tensor([list(prompt_ids)], device=self.device)
        with self._lock, torch.no_grad():
            self.model.generate(
                input_ids=ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                do_sample=False, streamer=_Streamer(), use_cache=True,
                pad_token_id=self.model.config.eos_token_id or 0)

    async def generate(self, prompt_ids: Sequence[int], new_tokens: int) -> AsyncIterator[int]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        fut = loop.run_in_executor(
            None, self._run, prompt_ids, new_tokens,
            lambda x: loop.call_soon_threadsafe(q.put_nowait, x))
        while True:
            x = await q.get()
            if x is None:
                break
            yield x
        await fut

    async def close(self):
        pass


class HFContinuousClient(HFGenerateClient):
    """transformers' own continuous batching: `model.init_continuous_batching()` -> ContinuousBatchingManager (paged KV
    cache, scheduler, optional CUDA graph / torch.compile). Requests are streamed per token through
    `register_result_handler`; each result carries the CUMULATIVE `generated_tokens`, so a result adds
    `len(generated_tokens) - seen` tokens. EOS is disabled per request (eos_token_id=-1) so every request produces
    exactly the requested count -- the driver marks any other count as failed instead of counting it."""

    def __init__(self, model_dir: str, device: str = "cuda", dtype: str = "bfloat16", tf32: bool = True,
                 attn: str = "flash_attention_2", cb_opts: dict | None = None):
        super().__init__(model_dir, device, dtype, tf32, attn)
        from transformers import GenerationConfig

        gen = GenerationConfig(max_new_tokens=64, do_sample=False,
                               pad_token_id=self.model.config.eos_token_id or 0)
        cfg = None
        if cb_opts:
            from transformers.generation import ContinuousBatchingConfig

            cfg = ContinuousBatchingConfig(**cb_opts)
        self.manager = self.model.init_continuous_batching(generation_config=gen, continuous_batching_config=cfg)
        self._started, self._n = False, 0   # started lazily, see generate()

    POLL_S = 2.0   # how often a waiting request checks that the generation thread is still alive

    def _thread_alive(self) -> bool:
        t = getattr(self.manager, "_generation_thread", None)
        return t is not None and t.is_alive()

    async def generate(self, prompt_ids: Sequence[int], new_tokens: int) -> AsyncIterator[int]:
        q: asyncio.Queue = asyncio.Queue()
        self._n += 1
        rid = f"cb{self._n}"
        self.manager.register_result_handler(rid, q.put_nowait)   # runs on this loop (call_soon_threadsafe)
        self.manager.add_request(list(prompt_ids), request_id=rid, max_new_tokens=new_tokens, streaming=True,
                                 eos_token_id=-1)
        if not self._started:   # start the thread only once a request is queued: with async batching (the default when
            self._started = True   # CUDA graphs are on) it bootstraps its first batch at once and DIES if there is none
            self.manager.start()   # (found by the stage-D smoke: an open-loop request that arrived 0.25 s late hung the run)
        seen = 0
        while True:
            try:
                out = await asyncio.wait_for(q.get(), timeout=self.POLL_S)
            except asyncio.TimeoutError:
                if not self._thread_alive():   # a dead manager never answers: fail the request instead of waiting forever
                    raise RuntimeError("HF continuous batching: the generation thread is not running")
                continue
            if out.error:
                raise RuntimeError(f"HF continuous batching: {out.error}")
            toks = out.generated_tokens
            for t in toks[seen:]:
                yield int(t)
            seen = max(seen, len(toks))
            if out.is_finished():
                break

    async def close(self):
        if self._started:
            self.manager.stop(block=True)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
async def _consume(client, ids, nt_i: int, t0: float) -> tuple[list[float], str]:
    """Stream one request; (token times relative to t0, status). A wrong token count is a failure, never a result."""
    times: list[float] = []
    status = "done"
    try:
        async for _ in client.generate(ids, nt_i):
            times.append(time.perf_counter() - t0)
    except Exception:  # noqa: BLE001
        status = "failed"
    if status == "done" and len(times) != nt_i:
        status = "failed"  # EOS / truncation: not comparable, never silently counted
    return times, status


async def run_open_loop(client, prompts: Sequence[Sequence[int]], new_tokens, arrivals: Sequence[float],
                        max_inflight: int = 0) -> tuple[list[Trace], float]:
    """Request i is sent at arrivals[i] seconds after the start, whether or not earlier ones finished. `max_inflight`
    (0 = none) caps how many are inside the server at once, like max_running_requests; the excess waits in FIFO
    order and that wait counts, because `arrival` is the scheduled time. The window for throughput runs from the
    first arrival to the last token."""
    if len(arrivals) != len(prompts):
        raise ValueError(f"{len(arrivals)} arrival times for {len(prompts)} prompts")
    sem = asyncio.Semaphore(max_inflight) if max_inflight else None
    t0 = time.perf_counter()
    traces: list[Trace] = []

    async def one(i: int, ids):
        nt_i = new_tokens[i] if not isinstance(new_tokens, int) else new_tokens
        wait = arrivals[i] - (time.perf_counter() - t0)
        if wait > 0:
            await asyncio.sleep(wait)
        if sem is None:
            times, status = await _consume(client, ids, nt_i, t0)
        else:
            async with sem:
                times, status = await _consume(client, ids, nt_i, t0)
        traces.append(Trace(f"r{i:04d}", arrivals[i], len(ids), times, status == "done", status))

    await asyncio.gather(*(one(i, ids) for i, ids in enumerate(prompts)))
    last = max((t.token_times[-1] for t in traces if t.token_times), default=0.0)
    return traces, last - (arrivals[0] if arrivals else 0.0)


async def run_closed_loop(client, prompts: Sequence[Sequence[int]], new_tokens,
                          concurrency: int) -> tuple[list[Trace], float]:
    """`new_tokens`: one int for every request, or a list with one entry per prompt (mixed output lengths)."""
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    traces: list[Trace] = []

    async def one(i: int, ids):
        nt_i = new_tokens[i] if not isinstance(new_tokens, int) else new_tokens
        async with sem:
            arrival = time.perf_counter() - t0
            times, status = await _consume(client, ids, nt_i, t0)
            traces.append(Trace(f"r{i:03d}", arrival, len(ids), times, status == "done", status))

    await asyncio.gather(*(one(i, ids) for i, ids in enumerate(prompts)))
    makespan = max((t.token_times[-1] for t in traces if t.token_times), default=0.0)
    return traces, makespan


def summarize_rep(traces, makespan, slo: Slo | None = None) -> dict:
    return summarize(traces, window_s=makespan, slo=slo or Slo())


def write_rows(w: ResultWriter, system: str, concurrency: int, reps: list[dict], new_tokens: int,
               notes: str = "", experiment: str = "I1", tag: str = "closed_loop"):
    """closed loop: experiment I1, `concurrency` = C. open loop: experiment I2, concurrency 0, `tag` carries the rate."""
    def stat(path):
        vals = []
        for r in reps:
            x = r
            for k in path:
                x = x[k]
            vals.append(float(x))
        return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    nt_label = new_tokens if isinstance(new_tokens, int) else f"mixed(mean={sum(new_tokens) / len(new_tokens):.0f})"
    base = dict(module="serving", experiment=experiment, system=system, concurrency=concurrency,
                n_reps=len(reps), variant=f"new_tokens={nt_label};{tag}")
    rel = all(r["p99_reliable"] for r in reps)
    for metric, path, unit, scale in [
        ("output_tok_per_s", ("output_token_throughput",), "tok/s", 1),
        ("request_per_s", ("request_throughput",), "req/s", 1),
        ("goodput_req_per_s", ("goodput",), "req/s", 1),
        ("ttft_ms_p50", ("ttft", "p50"), "ms", 1e3),
        ("ttft_ms_p99", ("ttft", "p99"), "ms", 1e3),
        ("tpot_ms_p50", ("tpot", "p50"), "ms", 1e3),
        ("tpot_ms_p99", ("tpot", "p99"), "ms", 1e3),
        ("itl_ms_p99", ("itl", "p99"), "ms", 1e3),
        ("e2e_s_p50", ("e2e", "p50"), "s", 1),
    ]:
        m, s = stat(path)
        w.write(**base, metric=metric, value=round(m * scale, 3), unit=unit,
                std=round(s * scale, 3) if len(reps) > 1 else "",
                notes=(notes + ("" if (rel or "p99" not in metric) else "; p99 unreliable (<100 reqs)")).strip("; "))
    failed = sum(r["n_failed"] for r in reps)
    if failed:
        w.write(**base, metric="n_failed_requests", value=failed, unit="count",
                notes="failed = error, or fewer/more tokens than requested (EOS not ignored?)")


# --------------------------------------------------------------------------------------
# ingest: recordings made by the engine's own harness
# --------------------------------------------------------------------------------------
def traces_from_recording(rec: dict) -> list[Trace]:
    return [Trace(r["id"], r["submit_t"], r["prompt_len"], [t for t, _ in r["tokens"]])
            for r in rec["requests"]]


def cmd_ingest(args) -> int:
    w = ResultWriter(args.out, REPO)
    slo = Slo(ttft_s=args.slo_ttft_s, tpot_s=args.slo_tpot_ms / 1e3)
    reps, conc, nt, meta = [], None, None, {}
    for p in args.recordings:
        rec = json.loads(Path(p).read_text())
        tr = traces_from_recording(rec)
        makespan = max(t.token_times[-1] for t in tr)   # an open-loop recording starts at its first arrival
        reps.append(summarize_rep(tr, makespan, slo))
        meta = rec["meta"]
        conc = meta.get("concurrency", conc)
        nt = meta.get("new_tokens", nt)
        print(f"{Path(p).name}: {reps[-1]['output_token_throughput']:.1f} tok/s  "
              f"TTFT p50 {reps[-1]['ttft']['p50']:.2f}s")
    extra = f"; {args.notes}" if args.notes else ""
    notes = f"from engine recordings ({len(reps)} reps); env columns = ingest host (same instance, same worktree as the recording){extra}"
    if args.open_loop:
        if not meta.get("open_loop"):
            raise SystemExit("--open-loop but the recording was made in closed loop (no arrival schedule in its meta)")
        tag = open_loop_tag(meta["arrival_rate"], meta["arrival_cv"], meta["arrival_duration_s"], conc)
        write_rows(w, args.system, 0, reps, int(nt), notes=f"{notes}; slo ttft<={args.slo_ttft_s}s tpot<={args.slo_tpot_ms}ms",
                   experiment="I2", tag=tag)
    else:
        write_rows(w, args.system, int(conc), reps, nt if isinstance(nt, int) else int(nt), notes=notes)
    m = statistics.fmean(r["output_token_throughput"] for r in reps)
    print(f"{args.system}: mean {m:.1f} tok/s over {len(reps)} reps")
    return 0


# --------------------------------------------------------------------------------------
def open_loop_tag(rate, cv, duration_s, cap) -> str:
    return f"open_loop;rate={rate:g};cv={cv:g};duration={duration_s:g}s;cap={cap or 'none'}"


async def run_open_rates(client, args, prompts, nt, w) -> None:
    """For each --arrival-rate: an untimed open-loop warm-up at that rate, then --reps timed runs of the SAME schedule
    (same seed => same arrival instants for every system and every repetition)."""
    slo = Slo(ttft_s=args.slo_ttft_s, tpot_s=args.slo_tpot_ms / 1e3)
    for rate in args.arrival_rate:
        n = n_for_duration(rate, args.duration)
        P = [prompts[i % len(prompts)] for i in range(n)]
        NT = nt if isinstance(nt, int) else [nt[i % len(nt)] for i in range(n)]
        arr = arrival_times(rate, n, args.seed, args.arrival_cv)
        if args.warmup_s:   # may be longer than the timed run (smoke tests): cycle the prompts instead of slicing
            k = n_for_duration(rate, args.warmup_s)
            PW = [prompts[i % len(prompts)] for i in range(k)]
            NTW = nt if isinstance(nt, int) else [nt[i % len(nt)] for i in range(k)]
            await run_open_loop(client, PW, NTW, arrival_times(rate, k, args.seed + 1000, args.arrival_cv), args.max_inflight)
        reps = []
        for r in range(args.reps):
            traces, ms = await run_open_loop(client, P, NT, arr, args.max_inflight)
            s = summarize_rep(traces, ms, slo)
            reps.append(s)
            print(f"{args.system} rate={rate:g}/s rep{r}: {s['output_token_throughput']:8.1f} tok/s  "
                  f"TTFT p50 {s['ttft'].get('p50', float('nan')):.2f}s p99 {s['ttft'].get('p99', float('nan')):.2f}s  "
                  f"TPOT p50 {1e3 * s['tpot'].get('p50', float('nan')):.1f}ms  failed={s['n_failed']}")
        write_rows(w, args.system, 0, reps, NT, experiment="I2",
                   tag=open_loop_tag(rate, args.arrival_cv, args.duration, args.max_inflight),
                   notes=f"{args.notes}; n_requests={n}; slo ttft<={args.slo_ttft_s}s tpot<={args.slo_tpot_ms}ms".strip("; "))


async def amain_run(args) -> int:
    wl = load_workload(args.workload)
    prompts = wl["prompts"][: args.n or None]
    nt_list = wl.get("new_tokens_list")            # optional: mixed output lengths (one per prompt)
    nt = [int(x) for x in nt_list[: len(prompts)]] if nt_list else int(wl["new_tokens"])
    if args.backend == "sglang":
        client = SGLangClient(args.url, args.endpoint)
    elif args.backend == "vllm":
        from vllm_client import VLLMClient  # exact per-token counts via continuous_usage_stats; see vllm_client.py
        client = VLLMClient(args.url, args.served_model_name)
    elif args.backend == "hf_cb":
        client = HFContinuousClient(args.model_dir, args.device, args.dtype, attn=args.attn,
                                    cb_opts=json.loads(args.cb_opts) if args.cb_opts else None)
    else:
        client = HFGenerateClient(args.model_dir, args.device, args.dtype, attn=args.attn)
    w = ResultWriter(args.out, REPO)
    try:
        if args.arrival_rate:
            await run_open_rates(client, args, prompts, nt, w)
            return 0
        if args.warmup:
            await run_closed_loop(client, prompts[: args.warmup], nt, min(args.warmup, 4))
        for c in args.concurrency:
            n_c = len(prompts) if not args.n_mult else min(len(prompts), max(args.n_min, args.n_mult * c))
            sub = lambda k: (nt[:k] if not isinstance(nt, int) else nt)
            if args.warm_mult:  # untimed pass with the real shapes at THIS concurrency (same rule as record_demo)
                k = min(len(prompts), args.warm_mult * c)
                await run_closed_loop(client, prompts[:k], sub(k), c)
            reps = []
            for r in range(args.reps):
                traces, ms = await run_closed_loop(client, prompts[:n_c], sub(n_c), c)
                s = summarize_rep(traces, ms)
                reps.append(s)
                print(f"{args.system} C={c:3d} rep{r}: {s['output_token_throughput']:8.1f} tok/s  "
                      f"TTFT p50 {s['ttft'].get('p50', float('nan')):.2f}s  "
                      f"TPOT p50 {1e3 * s['tpot'].get('p50', float('nan')):.1f}ms  failed={s['n_failed']}")
            write_rows(w, args.system, c, reps, sub(n_c), notes=f"{args.notes}; n_requests={n_c}".strip("; "))
    finally:
        await client.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("make-workload", help="extract prompt_ids from an engine recording")
    a.add_argument("--recording", required=True, type=Path)
    a.add_argument("--out", type=Path, default=HERE.parent / "workloads" / "workload_wiki128.json")

    b = sub.add_parser("run", help="closed-loop (default) or open-loop (--arrival-rate) run against hf / hf_cb / sglang-or-vLLM")
    b.add_argument("--backend", choices=["hf", "hf_cb", "sglang", "vllm"], required=True)
    b.add_argument("--system", required=True, help="label in the CSV, e.g. hf_generate / sglang_cb_only")
    b.add_argument("--workload", type=Path, default=HERE.parent / "workloads" / "workload_wiki128.json")
    b.add_argument("--concurrency", type=int, nargs="+", default=[8, 16, 32])
    b.add_argument("--reps", type=int, default=3)
    b.add_argument("--n", type=int, default=0, help="use only the first n prompts (0 = all)")
    b.add_argument("--warmup", type=int, default=4)
    b.add_argument("--n-mult", type=int, default=0,
                   help="requests per concurrency level = n_mult x C (0 = every prompt in the workload)")
    b.add_argument("--n-min", type=int, default=0,
                   help="floor for the request count at low concurrency (8 x C = 8 requests at C=1 is too few to read a p50 from)")
    b.add_argument("--warm-mult", type=int, default=0,
                   help="before each concurrency level, run an untimed pass of warm_mult x C requests (0 = off)")
    b.add_argument("--url", default="http://127.0.0.1:30000")
    b.add_argument("--endpoint", choices=["native", "openai"], default="native")
    b.add_argument("--served-model-name", default="default", help="vllm backend: the --served-model-name the server was started with")
    b.add_argument("--model-dir")
    b.add_argument("--device", default="cuda")
    b.add_argument("--dtype", default="float32")
    b.add_argument("--attn", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"],
                   help="HF attn_implementation: eager = materialised softmax (closest to the "
                        "engine's naive attention); sdpa = PyTorch fused kernels; flash_attention_2 = the standalone "
                        "flash-attn package (what 'HF + flash-attn' means)")
    b.add_argument("--cb-opts", default="",
                   help='hf_cb: JSON for ContinuousBatchingConfig, e.g. \'{"use_cuda_graph": true}\' (empty = HF defaults)')
    b.add_argument("--arrival-rate", type=float, nargs="+", default=None,
                   help="OPEN loop: mean request rate(s) in req/s; replaces --concurrency (one set of rows per rate)")
    b.add_argument("--arrival-cv", type=float, default=1.0, help="open loop: cv of the inter-arrival gaps (1 = Poisson)")
    b.add_argument("--duration", type=float, default=30.0, help="open loop: seconds of arrivals per timed run")
    b.add_argument("--warmup-s", type=float, default=5.0, help="open loop: seconds of untimed arrivals at the same rate before each rate")
    b.add_argument("--max-inflight", type=int, default=0,
                   help="open loop: cap on requests inside the server at once (0 = none; the server's own limit applies)")
    b.add_argument("--seed", type=int, default=0, help="open loop: arrival-schedule seed (same for every system)")
    b.add_argument("--slo-ttft-s", type=float, default=2.0)
    b.add_argument("--slo-tpot-ms", type=float, default=100.0)
    b.add_argument("--notes", default="")
    b.add_argument("--out", type=Path, default=HERE.parent / "results" / "final" / "serving.csv")

    c = sub.add_parser("ingest", help="engine recordings -> schema rows")
    c.add_argument("--recordings", nargs="+", required=True)
    c.add_argument("--system", required=True, help="e.g. engine_cb / engine_seq")
    c.add_argument("--notes", default="", help="appended to every row's notes (e.g. N, attn, weights)")
    c.add_argument("--open-loop", action="store_true",
                   help="the recordings were made with record_demo --arrival-rate: write experiment I2 rows tagged with the rate")
    c.add_argument("--slo-ttft-s", type=float, default=2.0)
    c.add_argument("--slo-tpot-ms", type=float, default=100.0)
    c.add_argument("--out", type=Path, default=HERE.parent / "results" / "final" / "serving.csv")

    args = ap.parse_args(argv)
    if args.cmd == "make-workload":
        wl = make_workload(args.recording, args.out)
        print(f"{len(wl['prompts'])} prompts, new_tokens={wl['new_tokens']} -> {args.out}")
        return 0
    if args.cmd == "ingest":
        return cmd_ingest(args)
    if args.backend in ("hf", "hf_cb") and not args.model_dir:
        ap.error("--model-dir is required for --backend hf / hf_cb")
    return asyncio.run(amain_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
