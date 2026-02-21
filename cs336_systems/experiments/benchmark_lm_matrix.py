#!/usr/bin/env python3
"""
benchmark_lm_matrix.py — Full LM training benchmark matrix.

Benchmarks a short training run across:
  - 4 attention kernels × {naive DDP, FlashDDP}  (+ optional single-GPU baseline)
  - Measures: wall-clock time per epoch, peak GPU memory, tokens/sec

Produces:
  artifacts/lm_matrix_results.csv
  artifacts/lm_matrix_time.png
  artifacts/lm_matrix_memory.png
  artifacts/lm_matrix_report.md

Requires CUDA and ≥2 GPUs for DDP benchmarks (single-GPU rows are always generated).

Usage:
  uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
      --train_path cs336-basics/data/tokenized/ts_train.npy \
      --val_path   cs336-basics/data/tokenized/ts_valid.npy \
      --epochs 3 --tr_batch_size 8 --context_length 256
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# Kernel & wrapper registries
# ---------------------------------------------------------------------------
KERNELS = [
    "scaled_dot_prod_attention",
    "vectorized_torch",
    "flash_attention_triton",
]

DDP_WRAPPERS = ["none", "naive", "flashddp"]


def _get_kernel_fn(name: str):
    from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNEL_REGISTRY
    return ATTENTION_KERNEL_REGISTRY[name]


# ---------------------------------------------------------------------------
# Single-GPU benchmark helper
# ---------------------------------------------------------------------------
def _bench_single_gpu(
    kernel_name: str,
    train_path: str,
    val_path: str,
    epochs: int,
    tr_batch_size: int,
    context_length: int,
    model_cfg: dict,
    dtype: torch.dtype,
) -> dict:
    """Run a short training loop on GPU-0, return timing & memory stats."""
    from cs336_basics.lm import TransformerLM
    from cs336_basics.train.loss import cross_entropy
    from cs336_basics.train.optimizer import AdamW
    from cs336_systems.Parallelization.DDP.stream_dataset import TokenStreamDataset
    from torch.utils.data import DataLoader

    device = "cuda:0"
    kernel_fn = _get_kernel_fn(kernel_name)

    model = TransformerLM(
        **model_cfg, device=device, dtype=dtype, attention_fn=kernel_fn,
    )
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999))

    dataset = TokenStreamDataset(train_path, context_length)
    loader = DataLoader(dataset, batch_size=tr_batch_size, shuffle=True, drop_last=True, num_workers=2)

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    total_tokens = 0
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x)
            loss = cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            total_tokens += x.numel()
            if i >= 50:
                break  # cap iterations per epoch for benchmark

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "kernel": kernel_name,
        "ddp": "none",
        "gpus": 1,
        "epochs": epochs,
        "wall_sec": round(elapsed, 3),
        "sec_per_epoch": round(elapsed / max(epochs, 1), 3),
        "tokens_per_sec": round(total_tokens / max(elapsed, 1e-6), 1),
        "peak_gpu_mb": round(peak_mb, 1),
    }


# ---------------------------------------------------------------------------
# DDP benchmark helper (runs inside mp.spawn)
# ---------------------------------------------------------------------------
def _ddp_worker(
    rank: int,
    world_size: int,
    wrapper_name: str,
    kernel_name: str,
    train_path: str,
    context_length: int,
    tr_batch_size: int,
    epochs: int,
    model_cfg: dict,
    dtype: torch.dtype,
    result_dict,
    bucket_size_mb: int = 1,
):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from cs336_basics.lm import TransformerLM
    from cs336_basics.train.loss import cross_entropy
    from cs336_basics.train.optimizer import AdamW
    from cs336_systems.Parallelization.DDP.stream_dataset import TokenStreamDataset
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    kernel_fn = _get_kernel_fn(kernel_name)
    model = TransformerLM(**model_cfg, device="cpu", dtype=dtype, attention_fn=kernel_fn)
    model = model.to(device)

    # Broadcast params from rank 0
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    if wrapper_name == "naive":
        pass  # manual all-reduce per param after backward
    elif wrapper_name == "flashddp":
        from cs336_systems.Parallelization.FlashDDP.FlashDDP import DDPOverlapBucketed
        model = DDPOverlapBucketed(model, bucket_size_mb=bucket_size_mb)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999))
    dataset = TokenStreamDataset(train_path, context_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=tr_batch_size, sampler=sampler, num_workers=2)

    torch.cuda.reset_peak_memory_stats(rank)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    total_tokens = 0
    for epoch in tqdm(range(epochs), desc=f"Rank {rank} Epochs", disable=(rank != 0)):
        sampler.set_epoch(epoch)
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if wrapper_name == "flashddp":
                optimizer.zero_grad(set_to_none=False)
            else:
                optimizer.zero_grad()
            pred = model(x)
            loss = cross_entropy(pred, y)
            loss.backward()

            if wrapper_name == "naive":
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world_size
            elif wrapper_name == "flashddp":
                model.finish_gradient_synchromnization()

            optimizer.step()
            total_tokens += x.numel()
            if i >= 50:
                break

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated(rank) / (1024 ** 2)

    # Only rank 0 writes results
    if rank == 0:
        result_dict["wall_sec"] = round(elapsed, 3)
        result_dict["sec_per_epoch"] = round(elapsed / max(epochs, 1), 3)
        result_dict["tokens_per_sec"] = round(total_tokens / max(elapsed, 1e-6), 1)
        result_dict["peak_gpu_mb"] = round(peak_mb, 1)

    dist.barrier()
    dist.destroy_process_group()


def _bench_ddp(
    wrapper_name: str,
    kernel_name: str,
    train_path: str,
    context_length: int,
    tr_batch_size: int,
    epochs: int,
    model_cfg: dict,
    dtype: torch.dtype,
) -> dict | None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        return None

    manager = mp.Manager()
    result_dict = manager.dict()
    result_dict["kernel"] = kernel_name
    result_dict["ddp"] = wrapper_name
    result_dict["gpus"] = world_size
    result_dict["epochs"] = epochs

    mp.spawn(
        _ddp_worker,
        args=(world_size, wrapper_name, kernel_name, train_path, context_length,
              tr_batch_size, epochs, model_cfg, dtype, result_dict),
        nprocs=world_size,
        join=True,
    )
    return dict(result_dict)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _plot_results(results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib not installed — skipping PNG generation.")
        return

    # --- Time bar chart ---
    labels = [f"{r['kernel']}\n{r['ddp']}" for r in results]
    times = [r["sec_per_epoch"] for r in results]
    colors = []
    ddp_color_map = {"none": "#4C72B0", "naive": "#DD8452", "flashddp": "#55A868"}
    for r in results:
        colors.append(ddp_color_map.get(r["ddp"], "#999999"))

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.5), 5))
    x = np.arange(len(results))
    bars = ax.bar(x, times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Seconds / Epoch")
    ax.set_title("LM Training: Time per Epoch (lower is better)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{t:.2f}s", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "lm_matrix_time.png", dpi=150)
    plt.close(fig)

    # --- Memory bar chart ---
    mems = [r["peak_gpu_mb"] for r in results]
    fig2, ax2 = plt.subplots(figsize=(max(8, len(results) * 1.5), 5))
    bars2 = ax2.bar(x, mems, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Peak GPU Memory (MB)")
    ax2.set_title("LM Training: Peak GPU Memory")
    for bar, m in zip(bars2, mems):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{m:.0f}", ha="center", va="bottom", fontsize=7)
    fig2.tight_layout()
    fig2.savefig(out_dir / "lm_matrix_memory.png", dpi=150)
    plt.close(fig2)

    # --- Throughput bar chart ---
    tps = [r["tokens_per_sec"] for r in results]
    fig3, ax3 = plt.subplots(figsize=(max(8, len(results) * 1.5), 5))
    bars3 = ax3.bar(x, tps, color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax3.set_ylabel("Tokens / Second")
    ax3.set_title("LM Training: Throughput (higher is better)")
    for bar, tp in zip(bars3, tps):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{tp:.0f}", ha="center", va="bottom", fontsize=7)
    fig3.tight_layout()
    fig3.savefig(out_dir / "lm_matrix_throughput.png", dpi=150)
    plt.close(fig3)
    print(f"[plot] Saved PNGs to {out_dir}")


def _write_markdown(results: list[dict], out_dir: Path) -> None:
    md_path = out_dir / "lm_matrix_report.md"
    with open(md_path, "w") as f:
        f.write("# LM Training Benchmark Matrix\n\n")
        f.write("| Kernel | DDP | GPUs | Epochs | Wall (s) | s/epoch | tok/s | Peak MB |\n")
        f.write("|--------|-----|------|--------|----------|---------|-------|--------|\n")
        for r in results:
            f.write(
                f"| {r['kernel']} | {r['ddp']} | {r['gpus']} | {r['epochs']} "
                f"| {r['wall_sec']} | {r['sec_per_epoch']} | {r['tokens_per_sec']} "
                f"| {r['peak_gpu_mb']} |\n"
            )
        f.write("\n![Time per Epoch](lm_matrix_time.png)\n")
        f.write("\n![Peak GPU Memory](lm_matrix_memory.png)\n")
        f.write("\n![Throughput](lm_matrix_throughput.png)\n")
    print(f"[report] Saved {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LM Training Benchmark Matrix")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tr_batch_size", type=int, default=8)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--kernels", nargs="*", default=None,
                        help="Subset of kernels to benchmark (default: all)")
    parser.add_argument("--wrappers", nargs="*", default=None,
                        help="Subset of DDP wrappers (default: none + all DDP if multi-GPU)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    model_cfg = dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        heads_num=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    )

    kernels = args.kernels or KERNELS
    wrappers = args.wrappers or (["none"] + (["naive", "flashddp"] if torch.cuda.device_count() >= 2 else []))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for kernel in kernels:
        for wrapper in wrappers:
            tag = f"{kernel} / {wrapper}"
            print(f"\n{'='*60}\n  Benchmarking: {tag}\n{'='*60}")
            try:
                if wrapper == "none":
                    row = _bench_single_gpu(kernel, args.train_path, args.val_path,
                                            args.epochs, args.tr_batch_size,
                                            args.context_length, model_cfg, dtype)
                else:
                    row = _bench_ddp(wrapper, kernel, args.train_path,
                                     args.context_length, args.tr_batch_size,
                                     args.epochs, model_cfg, dtype)
                if row:
                    results.append(row)
                    print(f"  ✓ {tag}: {row['sec_per_epoch']:.3f} s/epoch, {row['peak_gpu_mb']:.0f} MB")
                else:
                    print(f"  ⊘ {tag}: skipped (not enough GPUs)")
            except Exception as e:
                print(f"  ✗ {tag}: {e}")
                results.append({
                    "kernel": kernel, "ddp": wrapper, "gpus": 0,
                    "epochs": args.epochs, "wall_sec": 0, "sec_per_epoch": 0,
                    "tokens_per_sec": 0, "peak_gpu_mb": 0, "error": str(e),
                })

    # Save CSV
    csv_path = out_dir / "lm_matrix_results.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\n[csv] Saved {csv_path}")

    _plot_results(results, out_dir)
    _write_markdown(results, out_dir)
    print("\n[benchmark_lm_matrix] Done.")


if __name__ == "__main__":
    main()
