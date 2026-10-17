"""Profile the attention kernels on prefill-shaped work (forward only), one script, per sequence length.

Decision record: docs/sprints/week7-discussion.md D-7, "Attention":
  * a separate script from the timing sweep; different prefill lengths; profile in detail;
  * start-up and compile time are NOT in the profile: import, CUDA context, Triton autotune, torch.compile and
    lazy init all happen in warm-up calls before the profiler (or the Nsight capture range) opens;
  * both tools: torch.profiler and Nsight Systems; results are presented **grouped by attention kernel type**,
    comparing the mean of the key metrics of each group;
  * measurement-only attribution: nothing in his kernel is modified.

Per cell (system, seq) -- always a fresh process, shape 12 heads x hd64, causal, bf16, B*S = 32768 tokens:

  tool=torch   1. warm-up calls (compile / autotune / lazy init)
               2. `baseline_ms`: CUDA-event wall time over N calls with NO profiler attached (the honest latency)
               3. torch.profiler over N calls: per-kernel time and launch counts -> GPU time per call, launches
                  per call, GPU-busy fraction (kernel time / baseline wall), top kernels and their shares
  tool=nsys    the same warm-up, then cudaProfilerStart/Stop around N NVTX-marked calls; the orchestrator runs it
               under `nsys profile --capture-range=cudaProfilerApi`, then `nsys stats` -> kernel summary and the
               GPU trace (kernel union / first-start..last-end = busy fraction inside the region)

Output: <out-dir>/cells/*.json per cell, one CSV (schema.py) and a markdown summary grouped by system.
nsys is optional: when it is not on PATH the nsys half is skipped and the summary says so.

"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bench_attention as ba  # noqa: E402
from schema import ResultWriter  # noqa: E402
import manifest  # noqa: E402

REPO = ba.REPO

# kernel-name -> coarse category (checked in order; first match wins). The system is always known (one
# system per process), so 'flash' here means "that system's fused attention kernel".
CATEGORIES = [
    ("memcpy_memset", ("memcpy", "memset")),
    ("fused_attention", ("flash_fwd", "flash_bwd", "pytorch_flash", "fmha", "mem_eff", "attention")),
    ("gemm", ("gemm", "cutlass", "xmma", "gemv", "cublas", "bmm", "ampere_", "sm80_", "sm86_", "wmma")),
    ("softmax_reduce", ("softmax", "reduce", "logsumexp", "exp", "max_kernel")),
    ("mask_elementwise", ("elementwise", "masked_fill", "where", "triu", "tril", "arange", "fill", "copy", "div", "mul", "sub")),
]


def categorize(kernel_name: str) -> str:
    n = kernel_name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


# ---------------------------------------------------------------------------------------- parsers (unit-tested)
def merged_busy_ns(intervals: list[tuple[float, float]]) -> tuple[float, float]:
    """(union of [start, start+dur) in ns, span first-start..last-end in ns) for possibly overlapping intervals."""
    if not intervals:
        return 0.0, 0.0
    ivs = sorted((s, s + d) for s, d in intervals)
    busy, cur_s, cur_e = 0.0, ivs[0][0], ivs[0][1]
    for s, e in ivs[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
    busy += cur_e - cur_s
    return busy, max(e for _, e in ivs) - ivs[0][0]


def _col(header: list[str], *prefixes: str) -> int:
    low = [h.strip().lower() for h in header]
    for p in prefixes:
        for i, h in enumerate(low):
            if h.startswith(p):
                return i
    raise KeyError(f"none of {prefixes} in header {header}")


def parse_nsys_csv(text: str) -> tuple[list[str], list[list[str]]]:
    """`nsys stats --format csv` prints some banner lines before the CSV header; find the header row."""
    rows = list(csv.reader(io.StringIO(text)))
    for i, r in enumerate(rows):
        if len(r) >= 3 and any(c.strip().lower().startswith(("start", "time (%)", "name")) for c in r):
            return r, [x for x in rows[i + 1:] if len(x) == len(r)]
    raise ValueError("no CSV header found in nsys output")


def parse_nsys_kernel_summary(text: str) -> list[dict]:
    header, rows = parse_nsys_csv(text)
    i_total, i_inst, i_name = _col(header, "total time"), _col(header, "instances"), _col(header, "name")
    return [dict(name=r[i_name], total_ns=float(r[i_total]), instances=int(float(r[i_inst]))) for r in rows]


def parse_nsys_gpu_trace(text: str) -> dict:
    """Kernel rows of `cuda_gpu_trace`: busy fraction of the captured region and the number of kernels."""
    header, rows = parse_nsys_csv(text)
    i_s, i_d, i_n = _col(header, "start"), _col(header, "duration"), _col(header, "name")
    ivs = [(float(r[i_s]), float(r[i_d])) for r in rows if "memcpy" not in r[i_n].lower() and "memset" not in r[i_n].lower()]
    busy, span = merged_busy_ns(ivs)
    return dict(n_kernels=len(ivs), busy_ns=busy, span_ns=span, busy_frac=(busy / span if span else float("nan")))


def summarize_kernels(stats: list[dict], n_calls: int) -> dict:
    """stats: [{name, total_ns, instances}] over `n_calls` calls -> per-call numbers and the top kernels."""
    total = sum(k["total_ns"] for k in stats)
    by_cat: dict[str, float] = defaultdict(float)
    for k in stats:
        by_cat[categorize(k["name"])] += k["total_ns"]
    top = sorted(stats, key=lambda k: k["total_ns"], reverse=True)[:5]
    return dict(
        gpu_kernel_ms_per_call=total / n_calls / 1e6,
        launches_per_call=sum(k["instances"] for k in stats) / n_calls,
        top_kernels=[dict(name=k["name"][:90], share=(k["total_ns"] / total if total else 0.0),
                          ms_per_call=k["total_ns"] / n_calls / 1e6, per_call=k["instances"] / n_calls) for k in top],
        category_share={c: v / total for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1])} if total else {},
    )


# ---------------------------------------------------------------------------------------- one cell
def _cuda_wall_ms(run, n: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(n):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def _setup(args):
    dtype = ba.DTYPES[args.dtype]
    S, H, D = args.seq, args.heads, args.head_dim
    B = max(1, args.tokens // S)
    spec = ba.get_spec(args.system)
    g = torch.Generator(device=args.device).manual_seed(0)
    flat = [torch.randn(B * H, S, D, device=args.device, dtype=dtype, generator=g) for _ in range(3)]
    q, k, v = [spec.to_native(t, B, H) for t in flat]
    return spec, q, k, v, B


def run_single(args) -> int:
    out = dict(system=args.system, seq=args.seq, tool=args.tool, status="ok", n_calls=args.active_iters)
    try:
        spec, q, k, v, B = _setup(args)
        out["batch"] = B
        run = lambda: spec.fn(q, k, v, True)
        cuda = args.device.startswith("cuda")
        for _ in range(args.warmup_calls):  # compile / autotune / lazy init -- before any profiler is attached
            run()
        if cuda:
            torch.cuda.synchronize()

        if args.tool == "nsys":
            # Only the region between these two calls is captured (nsys --capture-range=cudaProfilerApi).
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("timed_region")
            for _ in range(args.active_iters):
                torch.cuda.nvtx.range_push("prefill_attention")
                run()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        else:
            from torch.profiler import ProfilerActivity, profile

            baseline = _cuda_wall_ms(run, args.active_iters) if cuda else float("nan")
            out["baseline_ms"] = baseline
            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
            wall_ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if cuda else None
            with profile(activities=acts, record_shapes=False, with_stack=False) as prof:
                if cuda:
                    wall_ev[0].record()
                for _ in range(args.active_iters):
                    run()
                if cuda:
                    wall_ev[1].record()
                    torch.cuda.synchronize()
            if cuda:   # same region as the kernel sum: the profiler's own overhead is in both numerator and denominator
                out["profiled_wall_ms"] = wall_ev[0].elapsed_time(wall_ev[1]) / args.active_iters
            stats = []
            for e in prof.key_averages():
                if e.device_type.name == "CUDA":
                    t_us = getattr(e, "device_time_total", None)
                    if t_us is None:
                        t_us = getattr(e, "cuda_time_total", 0.0)
                    stats.append(dict(name=e.key, total_ns=float(t_us) * 1e3, instances=int(e.count)))
            if cuda:
                s = summarize_kernels(stats, args.active_iters)
                out.update(s)
                out["gpu_busy_frac"] = s["gpu_kernel_ms_per_call"] / out["profiled_wall_ms"]
            else:
                cpu_ops = sorted(((e.key, e.self_cpu_time_total, e.count) for e in prof.key_averages()),
                                 key=lambda t: -t[1])[:5]
                out["cpu_top_ops"] = [dict(name=n[:60], self_cpu_us=t, count=c) for n, t, c in cpu_ops]
            if cuda:
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                run()
                torch.cuda.synchronize()
                out["peak_mem_mb"] = (torch.cuda.max_memory_allocated() - base) / 2**20
    except ImportError as e:
        out.update(status="unavailable", note=f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        out.update(status="oom" if ba._is_oom(e) else "error", note=f"{type(e).__name__}: {str(e)[:300]}")
    print("CELL " + json.dumps(out), flush=True)
    return 0


# ---------------------------------------------------------------------------------------- orchestration
def _spawn(args, system: str, seq: int, tool: str, prefix: list[str] | None = None, timeout_s: int = 900):
    cmd = (prefix or []) + [sys.executable, str(Path(__file__).resolve()), "--single", "--system", system,
                            "--seq", str(seq), "--tool", tool, "--dtype", args.dtype, "--tokens", str(args.tokens),
                            "--heads", str(args.heads), "--head-dim", str(args.head_dim), "--device", args.device,
                            "--warmup-calls", str(args.warmup_calls), "--active-iters", str(args.active_iters)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return dict(system=system, seq=seq, tool=tool, status="timeout")
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("CELL ")]
    if not lines:
        return dict(system=system, seq=seq, tool=tool, status="crash", note=f"rc={p.returncode} {(p.stderr or p.stdout)[-300:]}")
    return json.loads(lines[-1][5:])


NSYS_CANDIDATES = [  # oldest-first is deliberate: nsys must match the DRIVER's CUDA, not be the newest (see check_capture)
    "/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys",
]


def find_nsys(explicit: str | None = None) -> str | None:
    for c in [explicit, os.environ.get("NSYS_BIN"), *NSYS_CANDIDATES]:
        if c and Path(c).exists():
            return c
    return shutil.which("nsys")


def check_capture(kernel_instances: int, n_calls: int) -> str | None:
    """Every profiled call launches at least one kernel, so fewer captured kernels than calls means the trace is truncated.
    2026-10-16: nsys 2026.3.2 on a 550 driver (CUDA 12.4) recorded exactly ONE kernel and no CUDA API calls out of 10 calls
    while the NVTX ranges were all there; 2024.5.1 recorded all of them. A silent under-count would have produced
    plausible-looking numbers that are 10x too small."""
    if kernel_instances < n_calls:
        return (f"nsys captured {kernel_instances} kernel launch(es) for {n_calls} calls: the CUDA trace is truncated "
                "(nsys/driver mismatch?). Use a nsys release that matches the driver's CUDA version (NSYS_BIN=...).")
    return None


def nsys_cell(args, system: str, seq: int, out_dir: Path) -> dict:
    nsys = find_nsys(args.nsys_bin)
    if not nsys:
        return dict(system=system, seq=seq, tool="nsys", status="unavailable", note="nsys not on PATH")
    rep = out_dir / "nsys" / f"{system}_s{seq}"
    rep.parent.mkdir(parents=True, exist_ok=True)
    prefix = [nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
              "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true",
              "-o", str(rep)]
    cell = _spawn(args, system, seq, "nsys", prefix)
    if cell["status"] != "ok":
        return cell
    f = Path(str(rep) + ".nsys-rep")
    if not f.exists():
        return dict(cell, status="error", note="nsys produced no .nsys-rep (capture range never opened?)")

    def stats(report: str) -> str:
        r = subprocess.run([nsys, "stats", "--report", report, "--format", "csv", "--force-export=true", str(f)],
                           capture_output=True, text=True)
        return r.stdout

    try:
        ks = parse_nsys_kernel_summary(stats("cuda_gpu_kern_sum"))
        bad = check_capture(sum(k["instances"] for k in ks), args.active_iters)
        if bad:
            return dict(cell, status="error", note=bad, nsys_rep=str(f))
        cell.update(summarize_kernels(ks, args.active_iters))
        cell.update(parse_nsys_gpu_trace(stats("cuda_gpu_trace")))
        cell["nsys_rep"] = str(f)
    except Exception as e:  # noqa: BLE001  (CSV layout differs between nsys versions: keep the raw rep)
        cell.update(status="error", note=f"nsys stats parse failed: {type(e).__name__}: {e}; raw report kept at {f}")
    return cell


def run_all(args) -> int:
    out_dir = Path(args.out_dir)
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    manifest.record("attention_profile_prefill", out_dir=out_dir.parent,
                    extra=dict(tool=args.tool, nsys=find_nsys(args.nsys_bin) or "unavailable"))
    w = ResultWriter(out_dir.parent / "attention_profile.csv", REPO)
    cells: list[dict] = []
    tools = ["torch", "nsys"] if args.tool == "both" else [args.tool]
    for S in args.seqs:
        for system in args.systems:
            for tool in tools:
                cell = nsys_cell(args, system, S, out_dir) if tool == "nsys" else _spawn(args, system, S, "torch")
                (out_dir / "cells" / f"{tool}_{system}_s{S}.json").write_text(json.dumps(cell, indent=1))
                cells.append(cell)
                write_rows(w, args, cell)
                bits = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cell.items()
                        if k in ("status", "baseline_ms", "gpu_kernel_ms_per_call", "launches_per_call", "gpu_busy_frac", "busy_frac")}
                print(f"{tool:5s} S={S:6d} {system:26s} {bits}", flush=True)
    md = summary_markdown(cells, args)
    (out_dir / "SUMMARY.md").write_text(md)
    print("\n" + md)
    return 0


def write_rows(w: ResultWriter, args, cell: dict):
    base = dict(module="attention", experiment="A1-profile", system=cell["system"], dtype=args.dtype, causal=True,
                forward_only=True, seq_len=cell["seq"], batch=cell.get("batch") or max(1, args.tokens // cell["seq"]),
                n_gpus=1, variant=f"tool={cell['tool']};B*S={args.tokens}")
    if cell["status"] != "ok":
        w.write(**base, metric=cell["status"], value=1, unit="bool", n_reps=1, notes=str(cell.get("note", ""))[:300])
        return
    n = cell.get("n_calls", args.active_iters)
    for key, unit in (("baseline_ms", "ms"), ("gpu_kernel_ms_per_call", "ms"), ("launches_per_call", "count"),
                      ("gpu_busy_frac", "fraction"), ("busy_frac", "fraction"), ("peak_mem_mb", "MiB")):
        if key in cell and cell[key] == cell[key]:
            w.write(**base, metric=key, value=round(cell[key], 4), unit=unit, n_reps=n, warmup=args.warmup_calls)
    if cell.get("top_kernels"):
        w.write(**base, metric="top_kernels", value=len(cell["top_kernels"]), unit="count", n_reps=n,
                notes=json.dumps(cell["top_kernels"])[:300])


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.fmean(xs) if xs else float("nan")


def summary_markdown(cells: list[dict], args) -> str:
    """Group by attention kernel type (= system); compare the mean of each key metric over the sequence lengths,
    plus the geometric-mean latency ratio against `sdpa_flash` on the same lengths."""
    ok = [c for c in cells if c["status"] == "ok"]
    tor = {(c["system"], c["seq"]): c for c in ok if c["tool"] == "torch"}
    systems = [s for s in args.systems]
    lines = ["# Attention prefill profile -- grouped by attention kernel type",
             f"bf16, causal, 12 heads x hd64, B*S = {args.tokens}, forward only; torch.profiler over {args.active_iters} calls after warm-up.",
             "", "| attention kernel type | lengths ok | mean baseline ms | mean GPU kernel ms/call | mean launches/call | mean GPU-busy frac | geo-mean latency vs sdpa_flash |",
             "|---|---|---|---|---|---|---|"]
    for s in systems:
        mine = [c for (sys_, _), c in tor.items() if sys_ == s]
        ratios = [c["baseline_ms"] / tor[("sdpa_flash", c["seq"])]["baseline_ms"] for c in mine
                  if ("sdpa_flash", c["seq"]) in tor and c.get("baseline_ms", 0) > 0]
        geo = math.exp(statistics.fmean(math.log(r) for r in ratios)) if ratios else float("nan")
        lines.append(f"| {s} | {len(mine)}/{len(args.seqs)} | {_mean([c.get('baseline_ms') for c in mine]):.3f} | "
                     f"{_mean([c.get('gpu_kernel_ms_per_call') for c in mine]):.3f} | "
                     f"{_mean([c.get('launches_per_call') for c in mine]):.1f} | "
                     f"{_mean([c.get('gpu_busy_frac') for c in mine]):.2f} | {geo:.2f}x |")
    n_nsys = sum(1 for c in ok if c["tool"] == "nsys")
    lines += ["", f"Nsight Systems cells captured: {n_nsys}"
              + ("" if n_nsys else " (nsys unavailable or not requested -- torch.profiler only)"),
              "", "Means over lengths mix very different sizes; use them to compare kernel *types*, and read the per-length "
              "CSV (`attention_profile.csv`) for any single claim."]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "final" / "profile_prefill"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=list(ba.DTYPES))
    ap.add_argument("--seqs", type=int, nargs="+", default=ba.DEFAULT_SEQS)
    ap.add_argument("--systems", nargs="+", default=ba.DEFAULT_SYSTEMS, choices=ba.ALL_SYSTEMS)
    ap.add_argument("--tokens", type=int, default=ba.TOKENS)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--tool", default="both", choices=["both", "torch", "nsys"])
    ap.add_argument("--nsys-bin", default=None, help="default: $NSYS_BIN, then nsys 2024.5.1, then PATH")
    ap.add_argument("--warmup-calls", type=int, default=10)
    ap.add_argument("--active-iters", type=int, default=20, help="calls inside the profiler / capture range")
    ap.add_argument("--single", action="store_true", help="(internal) one cell in this process")
    ap.add_argument("--system", default=None, choices=ba.ALL_SYSTEMS)
    ap.add_argument("--seq", type=int, default=0)
    args = ap.parse_args(argv)
    if args.single:
        args.tool = "torch" if args.tool == "both" else args.tool
        return run_single(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
