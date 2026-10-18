"""Where does ONE batched decode step of the engine spend its time?  (RUNBOOK I3, "比数字值钱")

Builds B requests (prompts from a closed_loop workload), prefills them, puts them in the engine's
RaggedDecodeBatch and profiles `LMEngine.decode_batch` -- exactly the call the scheduler makes per round.

Reports per step: wall time, GPU kernel-busy time (=> GPU idle share = launch / Python / sync gaps),
number of kernel launches, the top ops by CUDA time, and the bytes of KV that `torch.cat` moves.

Usage (from serving/):
    python benchmarks/profile_decode_step.py --workload ../benchmarks/compare/results/workload_wiki128.json \
        --batch 32 --attn naive --dtype float32
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

SERVING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING))
for _p in (SERVING.parent / "training" / "cs336-basics", SERVING.parent / "training"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("record_demo", HERE / "record_demo.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--attn", choices=["naive", "sdpa", "fa2"], default="naive")
    ap.add_argument("--batched-sampling", action="store_true", help="ragged: one argmax + one tolist() per step")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--preset", default="190m")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--compile-decode", action="store_true", help="fa2: the decode forward as one torch.compile'd graph")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    dev = "cuda"
    cfg = rd.PRESETS[args.preset]
    dtype = getattr(torch, args.dtype)
    model = rd.build_model(cfg, dev, 0, None, rd.get_attention(args.attn), dtype=dtype)
    adapters = rd.load_adapters(SERVING / "tests" / "m2_adapters.py")
    engine = adapters.build_engine(model, None, {}, -1)
    lm = engine.lm_engine
    lm.batched_sampling = args.batched_sampling
    if args.attn == "fa2":
        ad = lm.model
        ad.decode_backend, ad.decode_slots = "prealloc", args.batch + 2
        ad.decode_max_len = 1024
        ad.compile_decode = args.compile_decode

    prompts = json.load(open(args.workload))["prompts"][: args.batch]
    batch = lm.new_decode_batch()
    latest = {}
    with torch.no_grad():
        for i, p in enumerate(prompts):
            tok, past = lm.prefill(torch.tensor(p).unsqueeze(0), max_length=len(p) + 64)
            batch.add(i, past)
            latest[i] = int(tok.item())

    def step():
        nonlocal latest
        latest = lm.decode_batch(batch, latest)

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / args.steps * 1e3

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()

    ev = prof.key_averages()
    cuda_total_us = sum(e.self_device_time_total for e in ev if e.device_type.name == "CUDA")
    launches = sum(e.count for e in ev if e.key == "cudaLaunchKernel")
    # Triton (torch.compile) kernels are launched through the DRIVER API (cuLaunchKernel), which the line above does not
    # see: count what actually RAN on the GPU instead (kernels + the small copies / memsets between them)
    gpu_execs = sum(e.count for e in ev if e.device_type.name == "CUDA")
    # blocking calls: every one of them makes the CPU wait for the GPU, so the CPU cannot run ahead and queue work
    sync_keys = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy", "cudaEventSynchronize")
    syncs = {k: sum(e.count for e in ev if e.key == k) for k in sync_keys}
    h2d_async = sum(e.count for e in ev if e.key == "cudaMemcpyAsync")
    esz = torch.empty((), dtype=dtype).element_size()
    per_tok = 2 * cfg["num_layers"] * cfg["num_heads"] * (cfg["d_model"] // cfg["num_heads"]) * esz  # K+V, all layers
    if hasattr(batch, "width"):       # ragged: the whole rectangle is re-cat'ed every step
        width = batch.width
        kv_bytes = per_tok * args.batch * width
        kv_note = "torch.cat in forward_with_cache"
    else:                             # prealloc: only the new token's K/V is written, in place
        width = max(batch._lens_cpu)
        kv_bytes = per_tok * args.batch
        kv_note = "appended in place (new token only)"

    print(f"\n== decode step, B={args.batch}, KV width={width}, attn={args.attn}, dtype={args.dtype}")
    print(f"wall per step           : {wall_ms:8.2f} ms")
    print(f"GPU kernel busy / step  : {cuda_total_us / args.steps / 1e3:8.2f} ms   "
          f"(GPU idle share {100 * (1 - cuda_total_us / args.steps / 1e3 / wall_ms):.0f}%)")
    print(f"GPU kernel runs / step  : {gpu_execs / args.steps:8.0f}   (cudaLaunchKernel calls {launches / args.steps:.0f}; compiled Triton kernels use cuLaunchKernel)")
    print("blocking CUDA calls/step: " + ", ".join(f"{k.removeprefix('cuda')} {v / args.steps:.1f}" for k, v in syncs.items())
          + f"   (cudaMemcpyAsync {h2d_async / args.steps:.1f})")
    print(f"flags                   : compile_decode={args.compile_decode}")
    print(f"KV bytes moved / step   : {kv_bytes / 1e9:8.4f} GB read + same written ({kv_note})")
    print("\ntop CUDA kernels by self time (per step):")
    rows = sorted((e for e in ev if e.device_type.name == "CUDA"), key=lambda e: -e.self_device_time_total)[:12]
    for e in rows:
        print(f"  {e.self_device_time_total / args.steps / 1e3:7.3f} ms  x{e.count / args.steps:5.0f}  {e.key[:100]}")
    print("\ntop CPU ops by self CPU time (per step):")
    rows = sorted((e for e in ev if e.device_type.name == "CPU"), key=lambda e: -e.self_cpu_time_total)[:8]
    for e in rows:
        print(f"  {e.self_cpu_time_total / args.steps / 1e3:7.3f} ms  x{e.count / args.steps:5.0f}  {e.key[:100]}")


if __name__ == "__main__":
    main()
