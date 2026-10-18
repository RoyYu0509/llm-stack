"""租机之后、跑任何 benchmark 之前先跑这个(10 秒):这台机器的 kernel launch 开销是否正常?

起因(2026-10-15):一台 vast 的 3090 实例 extra_env 里 CONTAINER_RUNTIME=gvisor,eager 模式下
空 kernel launch 约 0.124 ms/次,同一个 4096^3 fp16 matmul eager 11.9 ms vs CUDA-graph replay 2.5 ms,
约 47% 的 iteration 带 ~24 ms 的停顿。launch 密集的路径(自引擎 / HF generate / 无 CUDA graph 的 SGLang)
的数字因此不可信,而且会让「有 CUDA graph 的 SGLang default」显得比实际更快。

用法:
  python benchmarks/compare/env_check.py            # 打印并给出 PASS / WARN
  python benchmarks/compare/env_check.py --strict   # WARN 时退出码 1(可在脚本里 gate)

阈值是经验值(裸机 / 普通容器上 PyTorch eager 的一次小算子 launch 通常是几到十几微秒量级),
不是本机实测过的标准;gVisor 那台是 124 微秒。它只用来「拦住明显不正常的机器」。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

WARN_US = 50.0       # 单次 eager 小算子 launch 超过这个就告警(经验阈值)
STALL_MS = 5.0       # 单次 iteration 比中位数多出这么多算「停顿」


def runtime_hints() -> list[str]:
    hints = []
    for k in ("CONTAINER_RUNTIME", "VAST_CONTAINERLABEL", "RUNPOD_POD_ID"):
        if os.environ.get(k):
            hints.append(f"{k}={os.environ[k]}")
    try:  # gVisor 的 dmesg / 内核版本里通常带 gVisor 字样
        with open("/proc/version") as f:
            v = f.read().strip()
        if "gvisor" in v.lower() or "runsc" in v.lower():
            hints.append(f"/proc/version: {v[:80]}")
    except Exception:
        pass
    return hints


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--iters", type=int, default=3000)
    args = ap.parse_args(argv)

    import torch

    hints = runtime_hints()
    print("runtime hints:", hints or "none")
    gv = any("gvisor" in h.lower() or "runsc" in h.lower() for h in hints)

    if not torch.cuda.is_available():
        print("no CUDA device: only the runtime hints above are meaningful")
        return 1 if (args.strict and gv) else 0

    x = torch.zeros(1, device="cuda")
    for _ in range(200):          # warm up
        x.add_(1)
    torch.cuda.synchronize()

    # (1) average cost of one tiny eager op, one sync at the end
    t0 = time.perf_counter()
    for _ in range(args.iters):
        x.add_(1)
    torch.cuda.synchronize()
    per_launch_us = (time.perf_counter() - t0) / args.iters * 1e6

    # (2) per-iteration stalls: time each op boundary with CUDA events
    stalls = []
    for _ in range(300):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            x.add_(1)
        b.record()
        b.synchronize()
        stalls.append(a.elapsed_time(b))
    med = statistics.median(stalls)
    stalled = sum(1 for s in stalls if s - med > STALL_MS) / len(stalls)

    # (3) the same matmul eager vs CUDA graph (only if there is room)
    n = 2048
    A = torch.randn(n, n, device="cuda", dtype=torch.float16)
    B = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(20):
        A @ B
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        A @ B
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / 200 * 1e3
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        A @ B
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        C = A @ B  # noqa: F841
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        g.replay()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t0) / 200 * 1e3

    print(f"gpu                         : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"eager tiny-op launch        : {per_launch_us:8.1f} us   (warn > {WARN_US:.0f} us)")
    print(f"iterations with a >{STALL_MS:.0f} ms stall: {100 * stalled:5.1f} %  (median 10-op block {med * 1e3:.0f} us)")
    print(f"matmul {n}^3 fp16: eager {eager_ms:.3f} ms  vs  CUDA-graph {graph_ms:.3f} ms  "
          f"(eager/graph = {eager_ms / graph_ms:.2f}x)")

    bad = []
    if per_launch_us > WARN_US:
        bad.append(f"launch {per_launch_us:.0f} us > {WARN_US:.0f} us")
    if stalled > 0.05:
        bad.append(f"{100 * stalled:.0f}% iterations stalled")
    if gv:
        bad.append("gVisor runtime detected")
    if bad:
        print("WARN:", "; ".join(bad))
        print("     launch-bound numbers (eager attention sweep, HF generate, engine without CUDA graph) "
              "from this machine are NOT comparable. Prefer another instance, or label every number.")
        return 1 if args.strict else 0
    print("PASS: launch overhead looks normal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
