#!/usr/bin/env python3
"""
benchmark_lm_matrix.py — Full LM training benchmark matrix.

Benchmarks a short training run across:
  - 3 attention kernels × {single-GPU, Naive DDP DDP, FlashDDP, Torch DDP}
  - Measures: wall-clock time per epoch, peak per-GPU memory, tokens/sec,
    and per-step loss convergence.

DDP semantics (reflecting real-world usage):
  - Each GPU keeps the SAME local batch size as single-GPU.
  - Global batch size = local_batch_size × world_size.
  - Steps per epoch is identical across all configurations.
  - DDP therefore processes world_size× more samples per epoch in roughly
    the same wall-clock time, demonstrating higher throughput.
  - Per-GPU memory stays comparable (or lower for FlashDDP).
  - Loss converges faster in wall-clock time thanks to the larger effective
    batch.

Produces:
  artifacts/lm_matrix_results.csv
  artifacts/lm_matrix_loss_curves.csv
  artifacts/lm_matrix_table_<kernel>.pdf
  artifacts/lm_matrix_time.png
  artifacts/lm_matrix_memory.png
  artifacts/lm_matrix_throughput.png
  artifacts/lm_matrix_convergence.png
  artifacts/lm_matrix_report.md

Requires CUDA and ≥2 GPUs for DDP benchmarks (single-GPU rows are always generated).

Usage:
  uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
      --train_path data/tokenized/ts_train.npy \
      --val_path   data/tokenized/ts_valid.npy \
      --epochs 3 --tr_batch_size 8 --context_length 256
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
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

DDP_WRAPPERS = ["Local No DDP", "Naive DDP", "Bucketed Overlapping DDP", "Pytorch DDP"]
BENCH_STEPS_PER_EPOCH = 51
WARMUP_STEPS = 5


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
    """Run a short training loop on GPU-0, return timing & memory stats + loss curve."""
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

    # ── Warmup iterations (not timed) ──
    warmup_loader = DataLoader(dataset, batch_size=tr_batch_size, shuffle=True, drop_last=True, num_workers=2)
    for wi, (x, y) in enumerate(warmup_loader):
        print(f"[warmup] Iter {wi+1}/{WARMUP_STEPS}", end="\r")
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(x)
        loss = cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        if (wi + 1) >= WARMUP_STEPS:
            break
    torch.cuda.synchronize()
    del warmup_loader

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    

    total_tokens = 0
    loss_curve = []  # list of (wall_sec, global_step, loss_val)
    global_step = 0
    
    t0 = time.perf_counter()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x)
            loss = cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            total_tokens += x.numel()
            global_step += 1
            # Record loss every few steps
            if global_step % 5 == 1 or (i + 1) >= BENCH_STEPS_PER_EPOCH:
                loss_curve.append((
                    round(time.perf_counter() - t0, 4),
                    global_step,
                    round(loss.item(), 4),
                ))
            if (i + 1) >= BENCH_STEPS_PER_EPOCH:
                break  # cap iterations per epoch for benchmark

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "kernel": kernel_name,
        "ddp": "Local No DDP",
        "gpus": 1,
        "epochs": epochs,
        "steps_per_epoch": BENCH_STEPS_PER_EPOCH,
        "global_batch_size": tr_batch_size,
        "local_batch_size": tr_batch_size,
        "samples_per_epoch": BENCH_STEPS_PER_EPOCH * tr_batch_size,
        "wall_sec": round(elapsed, 3),
        "sec_per_epoch": round(elapsed / max(epochs, 1), 3),
        "tokens_per_sec": round(total_tokens / max(elapsed, 1e-6), 1),
        "peak_gpu_mb": round(peak_mb, 1),
        "loss_curve": loss_curve,
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
    local_batch_size: int,
    steps_per_epoch: int,
    epochs: int,
    model_cfg: dict,
    dtype: torch.dtype,
    result_dict,
    loss_list,
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

    if wrapper_name == "Naive DDP":
        pass  # manual all-reduce per param after backward
    elif wrapper_name == "Bucketed Overlapping DDP":
        from cs336_systems.Parallelization.FlashDDP.FlashDDP import DDPOverlapBucketed
        model = DDPOverlapBucketed(model, bucket_size_mb=bucket_size_mb)
    elif wrapper_name == "Pytorch DDP":
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        model = TorchDDP(model, device_ids=[rank], bucket_cap_mb=bucket_size_mb)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999))
    dataset = TokenStreamDataset(train_path, context_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler, num_workers=2)

    # ── Warmup iterations (not timed) ──
    warmup_loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler, num_workers=2)
    if rank == 0:
        print(f"[warmup-{wrapper_name}] starting {WARMUP_STEPS} iterations")
    for wi, (x, y) in enumerate(warmup_loader):
        if rank == 0:
            print(f"[warmup-{wrapper_name}] Iter {wi+1}/{WARMUP_STEPS}", end="\r", flush=True)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if wrapper_name == "Bucketed Overlapping DDP":
            optimizer.zero_grad(set_to_none=False)
        else:
            optimizer.zero_grad()
        pred = model(x)
        loss = cross_entropy(pred, y)
        loss.backward()
        if wrapper_name == "Naive DDP":
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size
        elif wrapper_name == "Bucketed Overlapping DDP":
            model.finish_gradient_synchromnization()
        # Pytorch DDP: gradient sync happens automatically in backward()
        optimizer.step()
        if (wi + 1) >= WARMUP_STEPS:
            break
    torch.cuda.synchronize(device)
    dist.barrier()
    del warmup_loader

    torch.cuda.reset_peak_memory_stats(rank)
    dist.barrier()
    torch.cuda.synchronize(device)
    

    total_tokens = 0
    global_step = 0

    t0 = time.perf_counter()
    for epoch in tqdm(range(epochs), desc=f"Rank {rank} Epochs", disable=(rank != 0)):
        sampler.set_epoch(epoch)
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if wrapper_name == "Bucketed Overlapping DDP":
                optimizer.zero_grad(set_to_none=False)
            else:
                optimizer.zero_grad()
            pred = model(x)
            loss = cross_entropy(pred, y)
            loss.backward()

            if wrapper_name == "Naive DDP":
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world_size
            elif wrapper_name == "Bucketed Overlapping DDP":
                model.finish_gradient_synchromnization()
            # Pytorch DDP: gradient sync happens automatically in backward()

            optimizer.step()
            total_tokens += x.numel()
            global_step += 1

            # Rank 0 records loss curve
            if rank == 0 and (global_step % 5 == 1 or (i + 1) >= steps_per_epoch):
                loss_list.append((
                    round(time.perf_counter() - t0, 4),
                    global_step,
                    round(loss.item(), 4),
                ))

            if (i + 1) >= steps_per_epoch:
                break

    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated(rank) / (1024 ** 2)
    # Use MAX to get the *per-GPU* peak (worst-case GPU), not the sum
    peak_tensor = torch.tensor([peak_mb], device=device, dtype=torch.float64)
    token_tensor = torch.tensor([total_tokens], device=device, dtype=torch.float64)
    elapsed_tensor = torch.tensor([elapsed], device=device, dtype=torch.float64)
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)

    # Only rank 0 writes results
    if rank == 0:
        wall_sec = elapsed_tensor.item()
        global_batch = local_batch_size * world_size
        result_dict["steps_per_epoch"] = steps_per_epoch
        result_dict["samples_per_epoch"] = global_batch * steps_per_epoch
        result_dict["wall_sec"] = round(wall_sec, 3)
        result_dict["sec_per_epoch"] = round(wall_sec / max(epochs, 1), 3)
        result_dict["tokens_per_sec"] = round(token_tensor.item() / max(wall_sec, 1e-6), 1)
        result_dict["peak_gpu_mb"] = round(peak_tensor.item(), 1)

    dist.barrier()
    dist.destroy_process_group()


def _find_free_port() -> int:
    """Return an unused TCP port by binding to port 0."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


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

    # ── KEY FIX: keep the SAME local batch size as single-GPU ──
    # Each GPU processes tr_batch_size samples per step.
    # Global batch = tr_batch_size × world_size.
    # Steps per epoch is the SAME as single-GPU (BENCH_STEPS_PER_EPOCH).
    # Result: DDP processes world_size× more samples per epoch in ~same time.
    local_batch_size = tr_batch_size
    global_batch = local_batch_size * world_size
    steps_per_epoch = BENCH_STEPS_PER_EPOCH

    # choose a free port for this run, avoids EADDRINUSE from previous runs
    port = _find_free_port()
    os.environ.setdefault("MASTER_PORT", str(port))

    manager = mp.Manager()
    result_dict = manager.dict()
    loss_list = manager.list()
    result_dict["kernel"] = kernel_name
    result_dict["ddp"] = wrapper_name
    result_dict["gpus"] = world_size
    result_dict["epochs"] = epochs
    result_dict["global_batch_size"] = global_batch
    result_dict["local_batch_size"] = local_batch_size

    mp.spawn(
        _ddp_worker,
        args=(world_size, wrapper_name, kernel_name, train_path, context_length,
              local_batch_size, steps_per_epoch,
              epochs, model_cfg, dtype, result_dict, loss_list),
        nprocs=world_size,
        join=True,
    )
    out = dict(result_dict)
    out["loss_curve"] = list(loss_list)
    return out


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

    # Filter out error rows for plotting
    valid = [r for r in results if r.get("wall_sec", 0) > 0]
    if not valid:
        print("[plot] No valid results to plot.")
        return

    labels = [f"{r['kernel']}\n{r['ddp']}" for r in valid]
    ddp_color_map = {"Local No DDP": "#4C72B0", "Naive DDP": "#DD8452", "Bucketed Overlapping DDP": "#55A868", "Pytorch DDP": "#C44E52"}
    colors = [ddp_color_map.get(r["ddp"], "#999999") for r in valid]
    x = np.arange(len(valid))

    # --- Time bar chart ---
    times = [r["sec_per_epoch"] for r in valid]
    fig, ax = plt.subplots(figsize=(max(8, len(valid) * 1.5), 5))
    bars = ax.bar(x, times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Seconds / Epoch")
    ax.set_title("LM Training: Wall-clock Time per Epoch\n(same steps/epoch — DDP processes more samples)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{t:.2f}s", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "lm_matrix_time.png", dpi=150)
    plt.close(fig)

    # --- Per-GPU Memory bar chart ---
    mems = [r["peak_gpu_mb"] for r in valid]
    fig2, ax2 = plt.subplots(figsize=(max(8, len(valid) * 1.5), 5))
    bars2 = ax2.bar(x, mems, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Peak Per-GPU Memory (MB)")
    ax2.set_title("LM Training: Peak Per-GPU Memory")
    for bar, m in zip(bars2, mems):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{m:.0f}", ha="center", va="bottom", fontsize=7)
    fig2.tight_layout()
    fig2.savefig(out_dir / "lm_matrix_memory.png", dpi=150)
    plt.close(fig2)

    # --- Throughput bar chart ---
    tps = [r["tokens_per_sec"] for r in valid]
    fig3, ax3 = plt.subplots(figsize=(max(8, len(valid) * 1.5), 5))
    bars3 = ax3.bar(x, tps, color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax3.set_ylabel("Tokens / Second (aggregate)")
    ax3.set_title("LM Training: Aggregate Throughput (higher is better)")
    for bar, tp in zip(bars3, tps):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{tp:.0f}", ha="center", va="bottom", fontsize=7)
    fig3.tight_layout()
    fig3.savefig(out_dir / "lm_matrix_throughput.png", dpi=150)
    plt.close(fig3)

    # --- Samples-per-epoch bar chart (shows DDP advantage) ---
    spe = [r.get("samples_per_epoch", 0) for r in valid]
    fig4, ax4 = plt.subplots(figsize=(max(8, len(valid) * 1.5), 5))
    bars4 = ax4.bar(x, spe, color=colors, edgecolor="black", linewidth=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax4.set_ylabel("Samples / Epoch")
    ax4.set_title("LM Training: Total Samples Processed per Epoch\n(DDP = world_size × single-GPU)")
    for bar, s in zip(bars4, spe):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{s}", ha="center", va="bottom", fontsize=7)
    fig4.tight_layout()
    fig4.savefig(out_dir / "lm_matrix_samples.png", dpi=150)
    plt.close(fig4)

    # --- Loss convergence plot ---
    has_curves = any(r.get("loss_curve") for r in valid)
    if has_curves:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        style_map = {"Local No DDP": "-", "Naive DDP": "--", "Bucketed Overlapping DDP": "-.", "Pytorch DDP": ":"}
        for r in valid:
            curve = r.get("loss_curve", [])
            if not curve:
                continue
            wall_secs = [pt[0] for pt in curve]
            losses = [pt[2] for pt in curve]
            label = f"{r['kernel']} / {r['ddp']} ({r['gpus']}GPU)"
            ax5.plot(wall_secs, losses,
                     linestyle=style_map.get(r["ddp"], "-"),
                     color=ddp_color_map.get(r["ddp"], "#999999"),
                     label=label, linewidth=1.2, alpha=0.85)
        ax5.set_xlabel("Wall-clock Time (s)")
        ax5.set_ylabel("Training Loss")
        ax5.set_title("Loss Convergence vs. Wall-clock Time")
        ax5.legend(fontsize=7, loc="upper right")
        ax5.grid(True, alpha=0.3)
        fig5.tight_layout()
        fig5.savefig(out_dir / "lm_matrix_convergence.png", dpi=150)
        plt.close(fig5)

    print(f"[plot] Saved PNGs to {out_dir}")


def _write_markdown(results: list[dict], out_dir: Path) -> None:
    md_path = out_dir / "lm_matrix_report.md"
    with open(md_path, "w") as f:
        f.write("# LM Training Benchmark Matrix\n\n")
        f.write("## Design\n\n")
        f.write("- **Local batch size** is the **same** across single-GPU and DDP.\n")
        f.write("- **Steps per epoch** is identical for all configurations.\n")
        f.write("- DDP therefore processes `world_size ×` more samples per epoch "
                "in roughly the same wall-clock time.\n")
        f.write("- Memory is reported **per-GPU** (worst-case across ranks).\n\n")
        f.write("## Results\n\n")
        f.write("| Kernel | DDP | GPUs | Epochs | Steps/ep | Global BS | Local BS "
                "| Samples/ep | Wall (s) | s/ep | tok/s | Peak GPU MB |\n")
        f.write("|--------|-----|------|--------|----------|-----------|----------"
                "|------------|----------|------|-------|-------------|\n")
        for r in results:
            f.write(
                f"| {r['kernel']} | {r['ddp']} | {r.get('gpus',0)} | {r.get('epochs',0)} "
                f"| {r.get('steps_per_epoch',0)} | {r.get('global_batch_size',0)} "
                f"| {r.get('local_batch_size',0)} "
                f"| {r.get('samples_per_epoch',0)} | {r.get('wall_sec',0)} "
                f"| {r.get('sec_per_epoch',0)} "
                f"| {r.get('tokens_per_sec',0)} | {r.get('peak_gpu_mb',0)} |\n"
            )
        f.write("\n## Charts\n\n")
        f.write("![Time per Epoch](lm_matrix_time.png)\n\n")
        f.write("![Peak Per-GPU Memory](lm_matrix_memory.png)\n\n")
        f.write("![Throughput](lm_matrix_throughput.png)\n\n")
        f.write("![Samples per Epoch](lm_matrix_samples.png)\n\n")
        f.write("![Loss Convergence](lm_matrix_convergence.png)\n")
    print(f"[report] Saved {md_path}")


def _write_kernel_table_pdfs(results: list[dict], out_dir: Path) -> None:
    """
    Generate one PDF summary table per attention kernel.

    Table format matches README style:
      DDP rows with columns: tok/s, Scaling Efficiency, memory Overhead.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[pdf-table] matplotlib not installed — skipping PDF table generation.")
        return

    valid = [r for r in results if r.get("wall_sec", 0) > 0]
    if not valid:
        print("[pdf-table] No valid rows to render.")
        return

    kernels = sorted({r["kernel"] for r in valid})
    emitted = 0

    for kernel in kernels:
        rows_for_kernel = {r["ddp"]: r for r in valid if r["kernel"] == kernel}
        if not rows_for_kernel:
            continue

        baseline = rows_for_kernel.get("Local No DDP")
        baseline_tps = baseline.get("tokens_per_sec", 0) if baseline else 0
        baseline_mem = baseline.get("peak_gpu_mb", 0) if baseline else 0

        table_rows = []
        for ddp in DDP_WRAPPERS:
            row = rows_for_kernel.get(ddp)
            if row is None:
                continue

            gpus = row.get("gpus", 0)
            tps = row.get("tokens_per_sec", 0)
            peak_mem = row.get("peak_gpu_mb", 0)
            ddp_label = f"`{ddp}` ({gpus} GPU)" if gpus == 1 else f"`{ddp}` ({gpus} GPUs)"

            if ddp == "Local No DDP":
                scaling = "N/A (no scaling)"
                overhead = "0 MB (0.0%)"
            else:
                if baseline_tps and gpus > 1:
                    scale_eff = (tps / (baseline_tps * gpus)) * 100.0
                    scaling = f"{scale_eff:.1f}%"
                else:
                    scaling = "N/A"

                if baseline_mem:
                    delta = peak_mem - baseline_mem
                    pct = (delta / baseline_mem) * 100.0
                    overhead = f"{delta:+.1f} MB ({pct:.1f}%)"
                else:
                    overhead = "N/A"

            table_rows.append([ddp_label, f"{tps:.1f}", scaling, overhead])

        if not table_rows:
            continue

        col_labels = [
            f"DDP Method\\Attention Kernel (`{kernel}`)",
            "tok/s",
            "Scaling Efficiency",
            "memory Overhead",
        ]

        fig_height = max(2.6, 1.0 + 0.55 * (len(table_rows) + 1))
        fig, ax = plt.subplots(figsize=(12.8, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            colLoc="center",
            colWidths=[0.48, 0.14, 0.19, 0.19],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.45)

        # Header styling
        for c in range(len(col_labels)):
            cell = table[(0, c)]
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")

        # Left-align the first column (DDP labels)
        for r_i in range(1, len(table_rows) + 1):
            table[(r_i, 0)].set_text_props(ha="left")

        out_path = out_dir / f"lm_matrix_table_{kernel}.png"
        fig.savefig(out_path, format="png", bbox_inches="tight")
        plt.close(fig)
        emitted += 1
        print(f"[image-table] Saved {out_path}")

    if emitted == 0:
        print("[image-table] No kernel tables generated.")


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
                        help="Subset of DDP wrappers (default: Local No DDP + all DDP if multi-GPU)")
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
    wrappers = args.wrappers or (["Local No DDP"] + (["Naive DDP", "Bucketed Overlapping DDP", "Pytorch DDP"] if torch.cuda.device_count() >= 2 else []))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    all_loss_curves = []  # for CSV export
    for kernel in kernels:
        for wrapper in wrappers:
            tag = f"{kernel} / {wrapper}"
            print(f"\n{'='*60}\n  Benchmarking: {tag}\n{'='*60}")
            try:
                if wrapper == "Local No DDP":
                    row = _bench_single_gpu(kernel, args.train_path, args.val_path,
                                            args.epochs, args.tr_batch_size,
                                            args.context_length, model_cfg, dtype)
                else:
                    row = _bench_ddp(wrapper, kernel, args.train_path,
                                     args.context_length, args.tr_batch_size,
                                     args.epochs, model_cfg, dtype)
                if row:
                    results.append(row)
                    # Collect loss curve for CSV
                    for pt in row.get("loss_curve", []):
                        all_loss_curves.append({
                            "kernel": kernel, "ddp": wrapper,
                            "wall_sec": pt[0], "step": pt[1], "loss": pt[2],
                        })
                    print(
                        f"  ✓ {tag}: {row['sec_per_epoch']:.3f} s/epoch, "
                        f"{row['peak_gpu_mb']:.0f} MB/GPU, "
                        f"samples/epoch={row.get('samples_per_epoch', 'N/A')}, "
                        f"{row['tokens_per_sec']:.0f} tok/s"
                    )
                else:
                    print(f"  ⊘ {tag}: skipped (not enough GPUs)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  ✗ {tag}: {e}")
                results.append({
                    "kernel": kernel, "ddp": wrapper, "gpus": 0,
                    "epochs": args.epochs, "steps_per_epoch": BENCH_STEPS_PER_EPOCH,
                    "global_batch_size": args.tr_batch_size, "local_batch_size": 0,
                    "samples_per_epoch": 0,
                    "wall_sec": 0, "sec_per_epoch": 0,
                    "tokens_per_sec": 0, "peak_gpu_mb": 0,
                    "error": str(e),
                })
            # Cleanup between runs
            torch.cuda.empty_cache()
            gc.collect()

    # Save main results CSV
    csv_path = out_dir / "lm_matrix_results.csv"
    if results:
        preferred_keys = [
            "kernel", "ddp", "gpus", "epochs",
            "steps_per_epoch", "global_batch_size", "local_batch_size",
            "samples_per_epoch",
            "wall_sec", "sec_per_epoch", "tokens_per_sec",
            "peak_gpu_mb", "error",
        ]
        keys = [k for k in preferred_keys if any(k in r for r in results)]
        rows_no_curve = [{k: v for k, v in r.items() if k != "loss_curve"} for r in results]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_no_curve)
        print(f"\n[csv] Saved {csv_path}")

    # Save loss curves CSV
    if all_loss_curves:
        lc_path = out_dir / "lm_matrix_loss_curves.csv"
        with open(lc_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kernel", "ddp", "wall_sec", "step", "loss"])
            w.writeheader()
            w.writerows(all_loss_curves)
        print(f"[csv] Saved {lc_path}")

    _plot_results(results, out_dir)
    _write_markdown(results, out_dir)
    _write_kernel_table_pdfs(results, out_dir)
    print("\n[benchmark_lm_matrix] Done.")


if __name__ == "__main__":
    main()
