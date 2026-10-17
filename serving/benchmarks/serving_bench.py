"""Serving 的完整指标:延迟(TTFT / TPOT / ITL / E2E + 尾部)、吞吐、goodput、MFU / MBU、显存。

两个子命令:

  analyze  读 record_demo.py 录下来的闭环录像(3090 上那批 sweep),不重跑,按 serving_metrics.py 的定义
           补算全部延迟 / 吞吐 / goodput 指标。
  run      开环压测:请求按泊松过程到达(到达速率与服务速度无关),工作负载是变长的 prompt / 输出。
           对比 seq(一次一条)和 cb(continuous batching),扫到达速率,画「延迟 / goodput vs 负载」。
           --profile 额外给每次模型前向计时,算 MFU / MBU / 前向时间占比 / 显存占用。

为什么还需要 run(闭环的 cb_benchmark / record_demo 回答不了的问题):
  · 闭环客户端「完成一条才发下一条」,到达速率被服务速度反过来限制,排队延迟被藏住;真实流量不会等你。
  · 固定输出长度 + 忽略 EOS + 均匀 prompt,不像真实对话的长尾分布。
  · 没有 goodput:吞吐高但每条请求都超 SLO 的系统,闭环看不出来。

开环的两处诚实性处理:
  · 请求的 arrival = 它「被调度到达」的时刻,不是引擎循环腾出手来实际收到它的时刻。引擎的一次前向会阻塞
    事件循环(prefill 可达几百毫秒),如果用实际收到的时刻算 TTFT,就把「被前向挡住的等待」从延迟里抹掉了
    (coordinated omission)。
  · 过载时 seq 的队列会无限长。到达结束后只再等 --drain 秒,没完成的请求记 aborted:计入 SLO 未达标,
    不计入吞吐。

用法(在 serving/ 下):
    uv run --no-sync python benchmarks/serving_bench.py analyze ../docs/videos/systems/data/sweep
    uv run --no-sync python benchmarks/serving_bench.py run --preset small --workload chat --duration 5 --profile
    # GPU 上的真实压测见 run_cb_sweep.sh 的 load 阶段
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import record_demo  # noqa: E402  (复用它的模型构造、预设、adapter 加载;它自己会把 serving/ 和 training/ 放进 sys.path)
from serving_metrics import (  # noqa: E402
    ForwardRecord, ModelCost, Slo, Trace, hw_summary, lookup_peak, summarize,
)

from inferenceLM.data.request import RequestData  # noqa: E402
from inferenceLM.data.request_status import RequestStatus  # noqa: E402
from inferenceLM.data.tokenized_data import TokenizedData  # noqa: E402

SERVING = HERE.parent
NO_EOS = -1

# ----------------------------------------------------------------------------------------
# analyze:已有录像 → 全部指标
# ----------------------------------------------------------------------------------------


def traces_from_recording(rec: Dict) -> List[Trace]:
    return [Trace(r["id"], r["submit_t"], r["prompt_len"], [t for t, _ in r["tokens"]]) for r in rec["requests"]]


def _mean_std(xs: Sequence[float]) -> str:
    return f"{statistics.fmean(xs):8.1f} ± {statistics.stdev(xs) if len(xs) > 1 else 0.0:5.1f}"


def analyze(paths: List[str], slo: Slo) -> None:
    files: List[str] = []
    for p in paths:
        files += sorted(glob.glob(str(Path(p) / "*.json"))) if Path(p).is_dir() else [p]
    groups: Dict[tuple, List[Dict]] = {}
    for f in files:
        rec = json.load(open(f))
        if "requests" not in rec:  # 目录里混着别的 json(例如 gate_wiki.json)
            continue
        m = rec["meta"]
        model = f"L{m['model_cfg']['num_layers']}d{m['model_cfg']['d_model']}"
        groups.setdefault(((m["gpu_name"] or m["device"])[-8:] + "/" + model + "/" + m["attn"], m["mode"], m["concurrency"]),
                          []).append(rec)

    print(f"SLO:TTFT ≤ {slo.ttft_s} s 且 TPOT ≤ {slo.tpot_s * 1000:.0f} ms。时间线 = 录像里每个 token 的可见时刻,"
          f"到达 = 客户端发出时刻(闭环,含排队)。\n分位数把各次重复的请求混在一起算(尾部样本更多);吞吐是各次重复的均值 ± 标准差。\n")
    hdr = (f"{'设备/attn':<22}{'模式':<4}{'C':>3}{'重复':>3}{'N':>5} | {'输出 tok/s':^17}{'总 tok/s':>9}{'req/s':>7} | "
           f"{'TTFT p50/p95/p99 (s)':^23} | {'TPOT p50/p95/p99 (ms)':^23} | {'ITL p50/p99 (ms)':^17} | "
           f"{'E2E p50/p99 (s)':^16} | {'SLO 达标':>7}{'goodput':>9}")
    print(hdr)
    for (dev, mode, c), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        per = [summarize(traces_from_recording(r), r["meta"]["makespan_s"], slo) for r in recs]
        pooled = summarize([t for r in recs for t in traces_from_recording(r)],
                           sum(r["meta"]["makespan_s"] for r in recs), slo)
        # 每条请求被计一次、窗口是各次之和 → 汇总里的「吞吐」其实是各次的合并均值;用 per 的均值/标准差展示
        ttf, tpo, itl, e2e = pooled["ttft"], pooled["tpot"], pooled["itl"], pooled["e2e"]
        flag = "" if pooled["p99_reliable"] else "*"
        print(f"{dev:<22}{mode:<4}{c:>3}{len(recs):>3}{pooled['n_total']:>5} | "
              f"{_mean_std([s['output_token_throughput'] for s in per])}{statistics.fmean([s['total_token_throughput'] for s in per]):9.0f}"
              f"{statistics.fmean([s['request_throughput'] for s in per]):7.2f} | "
              f"{ttf['p50']:6.2f}/{ttf['p95']:6.2f}/{ttf['p99']:6.2f}{flag:<1} | "
              f"{tpo['p50'] * 1e3:6.1f}/{tpo['p95'] * 1e3:6.1f}/{tpo['p99'] * 1e3:6.1f}{flag:<1} | "
              f"{itl['p50'] * 1e3:6.1f}/{itl['p99'] * 1e3:7.1f} | {e2e['p50']:6.1f}/{e2e['p99']:6.1f}{flag:<1} | "
              f"{pooled['slo_attainment'] * 100:6.1f}%{statistics.fmean([s['goodput'] for s in per]):9.2f}")
    print("\n* = 混合后样本数 < 100,p99 实际上是最大值附近的插值,不可靠。")


# ----------------------------------------------------------------------------------------
# 工作负载与到达过程
# ----------------------------------------------------------------------------------------

# 形状参考聊天类流量(长尾、输出和 prompt 同量级),不是 ShareGPT 数据集本身 —— 没有下载任何数据。
# 长度按 context 截断。想换分布只改这里。
WORKLOADS = {
    "fixed": dict(prompt=("uniform", 256, 900), out=("const", 64)),  # 与 record_demo / cb_benchmark 的闭环录像同口径
    "chat": dict(prompt=("lognormal", 200, 0.8, 8, 700), out=("lognormal", 100, 0.9, 8, 400)),
    "rag": dict(prompt=("uniform", 400, 800), out=("uniform", 16, 64)),  # 长 prompt、短输出
}


def _sample(spec: tuple, rng: random.Random) -> int:
    kind = spec[0]
    if kind == "const":
        return spec[1]
    if kind == "uniform":
        return rng.randint(spec[1], spec[2])
    median, sigma, lo, hi = spec[1:]
    return int(min(hi, max(lo, rng.lognormvariate(math.log(median), sigma))))


@dataclass
class Req:
    rid: str
    prompt: List[int]
    n_new: int


def make_requests(workload: str, n: int, seed: int, vocab: int, ctx: int, npy: Optional[str] = None) -> List[Req]:
    """n 条请求;prompt + 输出不超过 ctx。npy 给了就从真实文本里取连续片段,否则随机 id(吞吐与内容无关)。"""
    rng, w = random.Random(seed), WORKLOADS[workload]
    arr = None
    if npy:
        import numpy as np

        arr = np.load(npy, mmap_mode="r")
    reqs = []
    for i in range(n):
        p = min(_sample(w["prompt"], rng), ctx - 2)
        o = max(1, min(_sample(w["out"], rng), ctx - p))
        if arr is not None:
            s = rng.randrange(0, len(arr) - p)
            ids = [int(x) for x in arr[s:s + p]]
        else:
            ids = [rng.randrange(vocab) for _ in range(p)]
        reqs.append(Req(f"r{i:04d}", ids, o))
    return reqs


def poisson_arrivals(rate: float, n: int, seed: int) -> List[float]:
    """第 0 条在 t=0,之后间隔 ~ Exp(rate)。同一个 (rate, n, seed) 给 seq 和 cb 的到达序列逐位相同。"""
    rng, t, out = random.Random(seed), 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate)
    return out


# ----------------------------------------------------------------------------------------
# 前向计时 + 显存 + GPU 利用率
# ----------------------------------------------------------------------------------------


class ForwardProfiler:
    """给每次 forward_with_cache 计时并记下形状:in_embedding 的 pre-hook 是起点,head 的 hook 是终点。

    注意力宽度靠临时包装每个 MultiHeadsAttention 实例的 attention_fn 拿到
    (forward_with_cache 调用的是 self.attention_fn,实例创建时就绑定了,所以替换模块全局的 scaled_dot_product_attention 抓不到)。
    只在 --profile 时启用;CUDA 上前后各 synchronize 一次。
    """

    def __init__(self, model, sync: bool) -> None:
        self.records: List[ForwardRecord] = []
        self._sync = sync
        self._t0 = 0.0
        self._rows = self._new = 0
        self._kv: List[int] = []
        profiler = self

        def wrap(orig):
            def wrapped(Q, K, V, *a, **kw):
                profiler._kv.append(int(K.shape[-2]))
                return orig(Q, K, V, *a, **kw)
            return wrapped

        # one attention_fn per MultiHeadsAttention instance (one per layer); remember the originals for close()
        self._wrapped = [(m, m.attention_fn) for m in model.modules() if hasattr(m, "attention_fn")]
        for m, orig in self._wrapped:
            m.attention_fn = wrap(orig)
        self._h1 = model.in_embedding.register_forward_pre_hook(self._start)
        self._h2 = model.head.register_forward_hook(self._end)

    def _start(self, _m, inputs):
        if self._sync:
            torch.cuda.synchronize()
        self._rows, self._new, self._kv = int(inputs[0].shape[0]), int(inputs[0].shape[1]), []
        self._t0 = time.perf_counter()

    def _end(self, *_):
        if self._sync:
            torch.cuda.synchronize()
        self.records.append(ForwardRecord(time.perf_counter() - self._t0, self._rows, self._new, self._kv))

    def close(self) -> None:
        self._h1.remove()
        self._h2.remove()
        for m, orig in self._wrapped:
            m.attention_fn = orig


class UtilSampler:
    """后台线程定时读 nvidia-smi 口径的 GPU 利用率(需要 pynvml)。只用来和 MFU 对照 ——「kernel 在跑」不等于「算力用满」。"""

    def __init__(self, device: str, interval: float = 0.05) -> None:
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread = None
        if device == "cuda":
            try:
                torch.cuda.utilization()
                self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
            except Exception:
                self._thread = None

    def _loop(self, interval: float) -> None:
        while not self._stop.is_set():
            self.samples.append(float(torch.cuda.utilization()))
            time.sleep(interval)

    def __enter__(self):
        if self._thread:
            self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def mean(self) -> Optional[float]:
        return statistics.fmean(self.samples) if self.samples else None


# ----------------------------------------------------------------------------------------
# 开环 trial
# ----------------------------------------------------------------------------------------


async def run_open_loop(model, adapters, arm: str, reqs: List[Req], arrivals: List[float], drain_s: float) -> Dict:
    """按 arrivals 把 reqs 灌进引擎。arm='seq':一次只放一条进引擎,其余在门外排队;'cb':到了就交给引擎。"""
    queue: asyncio.Queue = asyncio.Queue()
    store: Dict[str, RequestData] = {}
    engine = adapters.build_engine(model, queue, store, NO_EOS)
    task = engine.run()
    by_id = {r.rid: r for r in reqs}
    gate: deque = deque()
    in_engine: set = set()
    arrival: Dict[str, float] = {}
    tok_at: Dict[str, List[float]] = {r.rid: [] for r in reqs}
    status: Dict[str, str] = {}
    last_done = 0.0
    t0 = time.perf_counter()
    deadline = t0 + arrivals[-1] + drain_s
    nxt = 0

    def to_engine(rid: str) -> None:
        r = by_id[rid]
        store[rid] = RequestData(request_id=rid, user_id="bench", timestamp=time.time(), prompt_text="",
                                 max_token_length=len(r.prompt) + r.n_new)
        queue.put_nowait(TokenizedData(request_id=rid, tokens=list(r.prompt)))
        in_engine.add(rid)

    while len(status) < len(reqs):
        now = time.perf_counter()
        if now > deadline:
            break
        while nxt < len(reqs) and t0 + arrivals[nxt] <= now:
            rid = reqs[nxt].rid
            arrival[rid] = t0 + arrivals[nxt]  # 调度到达时刻,不是 now(见模块文档:coordinated omission)
            (to_engine if arm == "cb" else gate.append)(rid)
            nxt += 1
        if arm == "seq" and not in_engine and gate:
            to_engine(gate.popleft())
        now = time.perf_counter()
        for rid in list(in_engine):
            seen = adapters.generated_so_far(engine, store, rid)
            tok_at[rid].extend([now] * (len(seen) - len(tok_at[rid])))
            st = store[rid].status
            if st in (RequestStatus.DONE, RequestStatus.FAILED):
                status[rid] = "done" if st == RequestStatus.DONE else "failed"
                in_engine.discard(rid)
                last_done = now
        busy = in_engine or gate
        wait = 0.0 if busy or nxt >= len(reqs) else max(0.0, min(t0 + arrivals[nxt] - now, 0.005))
        await asyncio.sleep(wait)

    aborted = len(status) < len(reqs)
    t_end = deadline if aborted else last_done
    for r in reqs:  # 超时没完成的:已到达的记 aborted;(deadline 在最后一次到达之后,所以都已到达)
        status.setdefault(r.rid, "aborted")
        arrival.setdefault(r.rid, t0 + arrivals[reqs.index(r)])
    if aborted:
        await asyncio.wait_for(engine.shutdown(draining=False), timeout=60)
    else:
        await asyncio.wait_for(queue.join(), timeout=60)
        await asyncio.wait_for(engine.shutdown(draining=True), timeout=60)
    await asyncio.wait_for(task, timeout=10)

    traces = [Trace(r.rid, arrival[r.rid], len(r.prompt), tok_at[r.rid], ok=status[r.rid] == "done",
                    status=status[r.rid]) for r in reqs]
    for t, r in zip(traces, reqs):
        if t.ok and len(t.token_times) != r.n_new:
            raise RuntimeError(f"{r.rid}: 完成了但只看到 {len(t.token_times)}/{r.n_new} 个 token —— 时间线不完整")
    return dict(traces=traces, window_s=max(t_end - t0, 1e-9), aborted=aborted)


async def one_trial(model, adapters, arm: str, reqs, arrivals, args, cost: Optional[ModelCost], peaks) -> Dict:
    cuda = args.device == "cuda"
    prof = ForwardProfiler(model, sync=cuda) if args.profile else None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        with UtilSampler(args.device if args.profile else "cpu") as util:
            res = await run_open_loop(model, adapters, arm, reqs, arrivals, args.drain)
    finally:
        if prof:
            prof.close()
    out = dict(arm=arm, aborted=res["aborted"], summary=summarize(res["traces"], res["window_s"], Slo(args.slo_ttft, args.slo_tpot)))
    if args.save_traces:
        out["traces"] = [dict(id=t.request_id, arrival=t.arrival, prompt_len=t.prompt_len, status=t.status,
                              token_times=list(t.token_times)) for t in res["traces"]]
    if prof:
        out["hw"] = hw_summary(prof.records, cost, peaks["tflops"], peaks["gbps"], res["window_s"])
        out["hw"]["gpu_util_mean_pct"] = util.mean()
    if cuda:
        out["peak_cuda_alloc_bytes"] = torch.cuda.max_memory_allocated()
    return out


# ----------------------------------------------------------------------------------------
# run 的汇总、打印、画图
# ----------------------------------------------------------------------------------------


def fmt_row(rate: float, t: Dict) -> str:
    s = t["summary"]
    nan = float("nan")
    star = "" if s["p99_reliable"] else "*"  # n_done < 100:p99 ≈ 最大值附近的插值
    return (f"{rate:6.2f} {t['arm']:>3} | {s['request_throughput']:5.2f} {s['output_token_throughput']:7.1f} "
            f"{s['total_token_throughput']:7.0f} | {s['ttft'].get('p50', nan):6.2f} "
            f"{s['ttft'].get('p99', nan):7.2f}{star:<1} | {s['tpot'].get('p50', nan) * 1e3:6.1f} "
            f"{s['tpot'].get('p99', nan) * 1e3:7.1f}{star:<1} | {s['itl'].get('p99', nan) * 1e3:7.1f} | "
            f"{s['e2e'].get('p99', nan):7.2f}{star:<1} | {s['slo_attainment'] * 100:5.1f}% {s['goodput']:6.2f} | "
            f"{s['n_failed']}/{s['n_aborted']}/{s['n_total']}")


HEADER = ("到达率 arm | req/s  out tok/s 总tok/s | TTFT p50    p99 | TPOT p50   p99(ms) | ITL p99(ms) | E2E p99(s) | "
          "SLO达标 goodput | 失败/中止/总")


def na(x: float, scale: float = 1.0, suffix: str = "", digits: int = 1) -> str:
    """nan(没量到 / 峰值未知)显示成 n/a,不显示成一个看起来像数字的 nan%。"""
    return "n/a" if x != x else f"{x * scale:.{digits}f}{suffix}"


def mean_over(trials: List[Dict], path: Sequence[str]) -> float:
    xs = []
    for t in trials:
        v = t
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
            if v is None:
                break
        if v is not None:
            xs.append(v)
    return statistics.fmean(xs) if xs else float("nan")


def plot(results: Dict, path: Path, title: str, profiled: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors, labels = {"seq": "#8a877c", "cb": "#2563eb"}, {"seq": "一次一条 (seq)", "cb": "continuous batching (cb)"}
    rates = sorted(float(r) for r in results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    def line(ax, arm, path_, scale=1.0, ls="-", suffix=""):
        ys = [mean_over(results[str(r)][arm], path_) * scale for r in rates]
        ax.plot(rates, ys, marker="o", color=colors[arm], ls=ls, label=labels[arm] + suffix)

    panels = [("goodput(req/s,满足 SLO)", ("summary", "goodput"), 1.0),
              ("SLO 达标率(%)", ("summary", "slo_attainment"), 100.0),
              ("输出 token 吞吐 (tok/s)", ("summary", "output_token_throughput"), 1.0)]
    for ax, (ttl, p, sc) in zip(axes[0], panels):
        for arm in ("seq", "cb"):
            line(ax, arm, p, sc)
        ax.set_title(ttl)
    axes[0][0].plot(rates, rates, ls=":", color="k", lw=1, label="y = 到达率(理想)")
    ax = axes[1][0]
    for arm in ("seq", "cb"):
        line(ax, arm, ("summary", "ttft", "p50"), suffix=" p50")
        line(ax, arm, ("summary", "ttft", "p99"), ls="--", suffix=" p99")
    ax.set_yscale("log")
    ax.set_title("TTFT (s,对数轴)")
    ax = axes[1][1]
    for arm in ("seq", "cb"):
        line(ax, arm, ("summary", "tpot", "p50"), 1e3, suffix=" p50")
        line(ax, arm, ("summary", "tpot", "p99"), 1e3, ls="--", suffix=" p99")
    ax.set_title("TPOT (ms)")
    ax = axes[1][2]
    if profiled:
        line(ax, "cb", ("hw", "decode", "mbu"), 100, suffix=" decode MBU%")
        line(ax, "cb", ("hw", "decode", "mfu"), 100, ls="--", suffix=" decode MFU%")
        line(ax, "cb", ("hw", "forward_time_fraction"), 100, ls=":", suffix=" 前向时间占比%")
        ax.set_title("cb 的硬件效率(%)")
    else:
        for arm in ("seq", "cb"):
            line(ax, arm, ("summary", "e2e", "p99"))
        ax.set_yscale("log")
        ax.set_title("E2E p99 (s)")
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("到达率 (req/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def cpu_name() -> str:
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        out = ""
    return out or platform.processor() or "cpu"


def environment(device: str) -> Dict:
    return dict(date=datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
                torch=torch.__version__, device=device, threads=torch.get_num_threads(), platform=platform.platform(),
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else "", **record_demo.provenance())


async def calibrate(model, adapters, wl: str, vocab: int, ctx: int, seed: int, args) -> float:
    """量一次 seq 的单条服务时间,返回 seq 的服务速率 μ(req/s)。同时是两种方式的预热。"""
    reqs = make_requests(wl, args.calib_n, seed + 999, vocab, ctx, args.prompts_npy)
    for arm in ("cb", "seq"):
        r = await run_open_loop(model, adapters, arm, reqs, [0.0] * len(reqs), drain_s=600)
    return len(reqs) / r["window_s"]


async def amain(args) -> None:
    if args.device == "cuda" and not args.no_tf32:
        torch.set_float32_matmul_precision("high")
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = record_demo.PRESETS[args.preset] if args.preset else json.load(open(args.config))["model"]
    model = record_demo.build_model(cfg, args.device, args.seed, args.checkpoint, record_demo.get_attention("naive"))
    adapters = record_demo.load_adapters(SERVING / "tests" / "m2_adapters.py")
    ctx, vocab = cfg["context_length"], cfg["vocab_size"]
    cost = ModelCost.from_model(model, cfg["num_heads"])
    gpu = torch.cuda.get_device_name(0) if args.device == "cuda" else cpu_name()
    peaks = lookup_peak(gpu)
    peaks = dict(tflops=args.peak_tflops or peaks["tflops"], gbps=args.peak_gbps or peaks["gbps"])
    print(f"[bench] {args.preset or args.config} device={args.device} workload={args.workload} "
          f"峰值 FLOPs={peaks['tflops']} TFLOPS 带宽={peaks['gbps']} GB/s(None → MFU/MBU 输出 n/a)")

    mu = await calibrate(model, adapters, args.workload, vocab, ctx, args.seed, args)
    rates = args.rates or [round(mu * m, 4) for m in args.rate_mult]
    print(f"[bench] seq 单条服务速率 μ ≈ {mu:.3f} req/s;扫描到达率 {rates} req/s;每档 {args.duration}s 的到达 "
          f"(N 夹在 [{args.min_n}, {args.max_n}]),到达结束后最多再等 {args.drain}s")
    print("\n" + HEADER + "\n(* = 完成的请求 < 100 条,p99 不可靠;N 小时所有分位数都只能看趋势)")

    results: Dict[str, List[Dict]] = {}
    for rep in range(args.repeats):
        for rate in rates:
            n = max(args.min_n, min(args.max_n, round(rate * args.duration)))
            reqs = make_requests(args.workload, n, args.seed + rep * 1000 + int(rate * 1000), vocab, ctx, args.prompts_npy)
            arrivals = poisson_arrivals(rate, n, args.seed + rep * 1000 + int(rate * 1000) + 1)
            for arm in (("seq", "cb") if (rep + rates.index(rate)) % 2 == 0 else ("cb", "seq")):
                t = await one_trial(model, adapters, arm, reqs, arrivals, args, cost, peaks)
                t.update(rate=rate, rep=rep, n=n)
                results.setdefault(str(rate), {}).setdefault(arm, []).append(t)
                print(fmt_row(rate, t), flush=True)
                dump(args, results, cfg, cost, mu, peaks)

    print_summary(results, rates, args, cost)
    out = dump(args, results, cfg, cost, mu, peaks)
    title = (f"{args.preset or 'config'} · {gpu} · 工作负载 {args.workload} · 泊松到达 · "
             f"{args.repeats} 次重复 · SLO TTFT≤{args.slo_ttft}s TPOT≤{args.slo_tpot * 1e3:.0f}ms")
    plot(results, out.with_suffix(".png"), title, args.profile)
    print(f"\n[bench] wrote {out} and {out.with_suffix('.png')}")


def print_summary(results: Dict, rates: List[float], args, cost: ModelCost) -> None:
    if not args.profile:
        return
    print("\n硬件效率(各次重复均值;prefill = 一条 prompt 一次前向,decode = 全部 DECODING 请求一次前向):")
    print(f"{'到达率':>7} {'arm':>3} | {'前向时间占比':>11} | {'decode 平均行数':>13} {'MFU':>6} {'MBU':>6} {'GB/s':>7} | "
          f"{'prefill MFU':>11} | {'GPU util(nvidia-smi 口径)':>22} | {'峰值显存 GB':>10}")
    for rate in rates:
        for arm in ("seq", "cb"):
            ts = results[str(rate)][arm]
            g = lambda *p: mean_over(ts, p)  # noqa: E731
            print(f"{rate:7.2f} {arm:>3} | {na(g('hw', 'forward_time_fraction'), 100, '%'):>11} | "
                  f"{g('hw', 'decode', 'rows_mean'):13.1f} {na(g('hw', 'decode', 'mfu'), 100, '%', 2):>6} "
                  f"{na(g('hw', 'decode', 'mbu'), 100, '%'):>6} {g('hw', 'decode', 'gbps'):7.1f} | "
                  f"{na(g('hw', 'prefill', 'mfu'), 100, '%'):>11} | {na(g('hw', 'gpu_util_mean_pct')):>22} | "
                  f"{na(g('peak_cuda_alloc_bytes'), 1 / 2**30):>10}")
    print(f"\n权重 {cost.weight_bytes / 2**20:.0f} MiB;KV cache 每 token {cost.kv_bytes_per_token() / 1024:.0f} KiB "
          f"(fp32,K+V,{cost.num_layers} 层)→ 一条 1024 token 的请求占 {cost.kv_bytes_per_token() * 1024 / 2**20:.0f} MiB。"
          + (f"\n按 {args.vram_gb} GB 显存估算的并发上限(不分页、90% 利用率):"
             f"{cost.max_concurrent_requests(int(args.vram_gb * 2**30), 1024)} 条 × 1024 token。" if args.vram_gb else ""))


def dump(args, results, cfg, cost, mu, peaks) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"load_{args.workload}_{args.preset or 'config'}{'' if args.device == 'cpu' else '_' + args.device}.json"
    json.dump(dict(environment=environment(args.device), args={k: v for k, v in vars(args).items() if k != "fn"},
                   model=cfg, seq_service_rate_req_s=mu, peaks=peaks,
                   cost=dict(matmul_params=cost.matmul_params, weight_bytes=cost.weight_bytes),
                   results=results), open(out, "w"), indent=1, ensure_ascii=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="按完整指标重新分析 record_demo 的闭环录像")
    a.add_argument("paths", nargs="+", help="json 文件或目录")
    for p in (a, sub.add_parser("run", help="开环压测")):
        p.add_argument("--slo-ttft", type=float, default=2.0)
        p.add_argument("--slo-tpot", type=float, default=0.1)
    r = sub.choices["run"]
    r.add_argument("--preset", choices=list(record_demo.PRESETS))
    r.add_argument("--config", help="json,读它的 model 段")
    r.add_argument("--checkpoint")
    r.add_argument("--prompts-npy", help="GPT-2 token 的 .npy,取连续片段当 prompt;不给则随机 id")
    r.add_argument("--workload", choices=list(WORKLOADS), default="chat")
    r.add_argument("--rates", type=float, nargs="+", help="到达率 req/s;不给则按 seq 服务速率 μ × --rate-mult")
    r.add_argument("--rate-mult", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16])
    r.add_argument("--duration", type=float, default=60, help="每档到达持续的秒数,N = 到达率 × duration")
    r.add_argument("--calib-n", type=int, default=6, help="校准 seq 服务速率 μ 用的请求数(同时是预热)")
    r.add_argument("--min-n", type=int, default=40)
    r.add_argument("--max-n", type=int, default=300)
    r.add_argument("--drain", type=float, default=60, help="最后一条到达之后最多再等多少秒,超时的记 aborted")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--profile", action="store_true", help="前向计时 → MFU / MBU / 前向占比 / 显存 / GPU util")
    r.add_argument("--peak-tflops", type=float)
    r.add_argument("--peak-gbps", type=float)
    r.add_argument("--vram-gb", type=float, help="按这个显存估算并发上限")
    r.add_argument("--save-traces", action="store_true", help="结果里保留每条请求每个 token 的时间(文件会很大)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-tf32", action="store_true")
    r.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    if args.cmd == "analyze":
        return analyze(args.paths, Slo(args.slo_ttft, args.slo_tpot))
    if not (args.preset or args.config):
        ap.error("run 需要 --preset 或 --config")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
