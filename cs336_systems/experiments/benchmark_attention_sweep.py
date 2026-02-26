#!/usr/bin/env python3
"""
benchmark_attention_sweep.py — Pure attention forward-only benchmark from Small to XXL.

Sweeps head dimension + sequence length tiers with all attention kernels.
Supports both square (Q_N == K_N) and rectangular (Q_N ≠ K_N) patterns.
All tier dimensions must be powers of 2; invalid custom tiers fail fast.

Produces:
  artifacts/attention_sweep_results.csv
  artifacts/attention_sweep_table_<config>.pdf
  artifacts/attention_sweep_forward.png
  artifacts/attention_sweep_heatmap.png
  artifacts/attention_sweep_report.md

Usage:
  uv run python cs336_systems/experiments/benchmark_attention_sweep.py
  uv run python cs336_systems/experiments/benchmark_attention_sweep.py \\
      --custom_tiers 128:128 256:128 512:256 1024:512 \\
      --warmup 10 --iters 50
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import re
import time
import timeit
from pathlib import Path

import torch
torch.set_float32_matmul_precision('high')
# ---------------------------------------------------------------------------
# Tier definitions (powers of 2)
# ---------------------------------------------------------------------------
DEFAULT_TIERS = [
    
    # ==========================================
    # A XS-size LM with 64 per-head dimension (d_model=768/heads=12), 12 heads, batch size 4
    # ==========================================
    ("XS-seq2048-hd64-12h-8b",   2048,   2048,   64, 12, 4),
    ("XS-seq4096-hd64-12h-8b",    4096,   4096,   64, 12, 4),
    ("XS-seq8192-hd64-12h-8b",    8192,   8192,   64, 12, 4),
    ("XS-seq16384-hd64-12h-8b",  16384,  16384,   64, 12, 4),
]

KERNEL_NAMES = [
    "scaled_dot_prod_attention",
    "vectorized_torch",
    "vectorized_torch_compiled",
    "flash_attention_triton",
]


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _validate_tier(label: str, q_n: int, k_n: int, head_dim: int):
    for name, val in [("Q_N", q_n), ("K_N", k_n), ("head_dim", head_dim)]:
        if not _is_power_of_2(val):
            raise ValueError(f"Tier '{label}': {name}={val} is not a power of 2.")


def _get_kernel_fn(name: str):
    from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNEL_REGISTRY
    return ATTENTION_KERNEL_REGISTRY[name]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Benchmark one (kernel, tier) pair
# ---------------------------------------------------------------------------
def bench_one(
    kernel_name: str,
    kernel_fn,
    tier_label: str,
    q_n: int,
    k_n: int,
    head_dim: int,
    num_heads: int,
    batch_size: int,
    warmup: int,
    iters: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """Forward-only benchmark. Returns timing dict."""
    shape_q = (batch_size * num_heads, q_n, head_dim)
    shape_kv = (batch_size * num_heads, k_n, head_dim)

    q = torch.randn(shape_q, device=device, dtype=dtype)
    k = torch.randn(shape_kv, device=device, dtype=dtype)
    v = torch.randn(shape_kv, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(q, k, v, True)
    _sync()
    print("finished warmup, starting timed iterations...")

    # Timed iterations
    times = []
    for _ in range(iters):
        _sync()
        t0 = timeit.default_timer()
        _ = kernel_fn(q, k, v, True)
        _sync()
        t1 = timeit.default_timer()
        times.append(t1 - t0)

    avg_ms = 1000.0 * sum(times) / len(times)
    std_ms = 1000.0 * (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5
    min_ms = 1000.0 * min(times)

    del q, k, v
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "kernel": kernel_name,
        "tier": tier_label,
        "Q_N": q_n,
        "K_N": k_n,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "avg_ms": round(avg_ms, 4),
        "std_ms": round(std_ms, 4),
        "min_ms": round(min_ms, 4),
    }


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
        print("[plot] matplotlib not available — skipping PNG.")
        return

    # drop any rows where an error (e.g. OOM) occurred
    clean_results = [r for r in results if not r.get("error")]
    # preserve ordering defined in KERNEL_NAMES; any unexpected kernels appended alphabetically
    kernels = [k for k in KERNEL_NAMES if any(r["kernel"] == k for r in clean_results)]
    extras = sorted({r["kernel"] for r in clean_results} - set(kernels))
    kernels += extras
    tiers = list(dict.fromkeys(r["tier"] for r in clean_results))  # preserves order

    kernel_colors = {
        "scaled_dot_prod_attention": "#4C72B0",
        "vectorized_torch": "#DD8452",
        "vectorized_torch_compiled": "#D30000",
        "flash_attention_triton": "#55A868",
    }

    # --- Line plots by configuration: seq_len x-axis ---
    # We only plot square tiers (Q_N == K_N) and create one subplot per
    # unique combination of head_dim, num_heads, batch_size.
    square = [r for r in clean_results if r["Q_N"] == r["K_N"]]
    if square:
        # group by config tuple
        configs = {}
        for r in square:
            cfg = (r["head_dim"], r["num_heads"], r["batch_size"])
            configs.setdefault(cfg, []).append(r)

        ncfg = len(configs)
        cols = min(3, ncfg)
        rows = math.ceil(ncfg / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
        markers = ["o", "s", "^", "D", "x"]
        for idx, (cfg, cfg_rows) in enumerate(configs.items()):
            hd, nh, bs = cfg
            ax = axes[idx // cols][idx % cols]
            ax.set_title(f"hd={hd}, heads={nh}, batch={bs}")
            ax.set_xlabel("Sequence Length (Q_N=K_N)")
            ax.set_ylabel("Forward Time (ms)")
            for i, kern in enumerate(kernels):
                rows_k = sorted([r for r in cfg_rows if r["kernel"] == kern], key=lambda r: r["Q_N"])
                if not rows_k:
                    continue
                xs = [r["Q_N"] for r in rows_k]
                ys = [r["avg_ms"] for r in rows_k]
                yerrs = [r.get("std_ms", 0) for r in rows_k]
                ax.errorbar(xs, ys, yerr=yerrs,
                            label=kern.replace("_", " "),
                            color=kernel_colors.get(kern, f"C{i}"),
                            marker=markers[i % len(markers)],
                            linestyle="-", linewidth=1.5, markersize=5, capsize=2)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6)
        # remove empty subplots
        for j in range(ncfg, rows * cols):
            fig.delaxes(axes[j // cols][j % cols])
        fig.tight_layout()
        fig.savefig(out_dir / "attention_sweep_forward.png", dpi=150)
        plt.close(fig)
    else:
        print("[plot] No square tiers available for sequence-length plots.")

    # --- Heatmap: kernel × tier ---
    mat = np.zeros((len(kernels), len(tiers)))
    for i, kern in enumerate(kernels):
        for j, tier in enumerate(tiers):
            row = next((r for r in clean_results if r["kernel"] == kern and r["tier"] == tier), None)
            mat[i, j] = row["avg_ms"] if row else float("nan")

    fig2, ax2 = plt.subplots(figsize=(max(8, len(tiers) * 1.2), max(4, len(kernels) * 1.2)))
    im = ax2.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax2.set_xticks(range(len(tiers)))
    ax2.set_xticklabels(tiers, fontsize=8, rotation=30, ha="right")
    ax2.set_yticks(range(len(kernels)))
    ax2.set_yticklabels([k.replace("_", " ") for k in kernels], fontsize=8)
    for i in range(len(kernels)):
        for j in range(len(tiers)):
            v = mat[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if v > np.nanmax(mat) * 0.6 else "black")
    ax2.set_title("Forward Time Heatmap (ms)")
    fig2.colorbar(im, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "attention_sweep_heatmap.png", dpi=150)
    plt.close(fig2)

    # --- Log-scale line chart: seq_len vs time (square tiers only) ---
    square_tiers = [r for r in results if r["Q_N"] == r["K_N"]]
    if square_tiers:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        for kern in kernels:
            rows = sorted([r for r in square_tiers if r["kernel"] == kern], key=lambda r: r["Q_N"])
            if rows:
                xs = [r["Q_N"] for r in rows]
                ys = [r["avg_ms"] for r in rows]
                ax3.plot(xs, ys, "o-", label=kern.replace("_", " "),
                         color=kernel_colors.get(kern, "#999"), linewidth=2, markersize=6)
        ax3.set_xscale("log", base=2)
        ax3.set_yscale("log")
        ax3.set_xlabel("Sequence Length (Q_N = K_N)")
        ax3.set_ylabel("Forward Time (ms, log)")
        ax3.set_title("Attention Scaling: Sequence Length vs Forward Time")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(out_dir / "attention_sweep_scaling.png", dpi=150)
        plt.close(fig3)

    print(f"[plot] Saved PNGs to {out_dir}")


def _write_markdown(results: list[dict], out_dir: Path) -> None:
    md = out_dir / "attention_sweep_report.md"
    with open(md, "w") as f:
        f.write("# Attention Forward Benchmark Sweep\n\n")
        f.write("| Kernel | Tier | Q_N | K_N | head_dim | heads | batch | avg_ms | std_ms | min_ms |\n")
        f.write("|--------|------|-----|-----|----------|-------|-------|--------|--------|--------|\n")
        for r in results:
            f.write(
                f"| {r['kernel']} | {r['tier']} | {r['Q_N']} | {r['K_N']} "
                f"| {r['head_dim']} | {r['num_heads']} | {r['batch_size']} "
                f"| {r['avg_ms']} | {r['std_ms']} | {r['min_ms']} |\n"
            )
        f.write("\n![Forward Time](attention_sweep_forward.png)\n")
        f.write("\n![Heatmap](attention_sweep_heatmap.png)\n")
        f.write("\n![Scaling Curve](attention_sweep_scaling.png)\n")
    print(f"[report] Saved {md}")


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")


def _write_pdf_tables(results: list[dict], out_dir: Path) -> None:
    """
    Export one PDF table per attention configuration.

    Table columns:
      - Kernel
      - Avg Forward Time (ms)
      - Relative to Triton Flash
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[pdf-table] matplotlib not installed — skipping PDF table generation.")
        return

    if not results:
        print("[pdf-table] No results to render.")
        return

    # Group by full configuration; tier labels may repeat, so include full shape metadata.
    grouped: dict[tuple, list[dict]] = {}
    for r in results:
        key = (
            r.get("tier"),
            r.get("Q_N"),
            r.get("K_N"),
            r.get("head_dim"),
            r.get("num_heads"),
            r.get("batch_size"),
        )
        grouped.setdefault(key, []).append(r)

    kernel_order = {k: i for i, k in enumerate(KERNEL_NAMES)}
    emitted = 0

    for (tier, q_n, k_n, hd, nh, bs), rows in grouped.items():
        by_kernel = {r["kernel"]: r for r in rows}
        ordered_kernels = sorted(by_kernel.keys(), key=lambda k: kernel_order.get(k, 999))
        flash_row = by_kernel.get("flash_attention_triton")
        flash_ms = None
        if flash_row and not flash_row.get("error") and flash_row.get("avg_ms", 0) > 0:
            flash_ms = float(flash_row["avg_ms"])

        table_rows = []
        for kernel in ordered_kernels:
            row = by_kernel[kernel]
            if row.get("error") or row.get("avg_ms", 0) <= 0:
                avg_text = "ERROR"
                rel_text = "N/A"
            else:
                avg_ms = float(row["avg_ms"])
                avg_text = f"{avg_ms:.2f}"
                if flash_ms is None:
                    rel_text = "N/A"
                else:
                    ratio = avg_ms / flash_ms
                    if kernel == "flash_attention_triton":
                        rel_text = "1.00x"
                    elif ratio >= 1.0:
                        rel_text = f"{ratio:.2f}x slower"
                    else:
                        rel_text = f"{(1.0 / ratio):.2f}x faster"

            table_rows.append([kernel, avg_text, rel_text])

        if not table_rows:
            continue

        fig_h = max(2.8, 1.15 + 0.5 * (len(table_rows) + 1))
        fig, ax = plt.subplots(figsize=(11.8, fig_h))
        ax.axis("off")
        title = (
            f"{tier} | Q={q_n}, K={k_n}, head_dim={hd}, heads={nh}, batch={bs}"
        )
        ax.set_title(title, fontsize=11, pad=8)

        col_labels = ["Kernel", "Avg Forward Time (ms)", "Relative to Triton Flash"]
        table = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            colLoc="center",
            colWidths=[0.34, 0.24, 0.42],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.42)

        # Header style
        for c in range(len(col_labels)):
            cell = table[(0, c)]
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")

        # Highlight flash row values to mirror emphasis in README
        for r_i, (kernel, _, _) in enumerate(table_rows, start=1):
            if kernel == "flash_attention_triton":
                for c in range(len(col_labels)):
                    table[(r_i, c)].set_text_props(weight="bold")

        slug = _slugify(f"{tier}_Q{q_n}_K{k_n}_hd{hd}_h{nh}_b{bs}")
        out_path = out_dir / f"attention_sweep_table_{slug}.png"
        fig.savefig(out_path, format="png", dpi=1200, bbox_inches="tight")
        plt.close(fig)
        emitted += 1
        print(f"[image-table] Saved {out_path}")

    if emitted == 0:
        print("[image-table] No image tables generated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_tier(s: str, idx: int):
    """Parse 'Q_N:K_N' or 'Q_N' (square) from CLI, with power-of-2 validation."""
    parts = s.split(":")
    if len(parts) == 1:
        q_n = k_n = int(parts[0])
    elif len(parts) == 2:
        q_n, k_n = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Invalid tier spec: {s}")
    label = f"Custom-{idx}"
    _validate_tier(label, q_n, k_n, 128)
    return (label, q_n, k_n, 128, 8, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention forward benchmark sweep")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--kernels", nargs="*", default=None,
                        help="Kernel subset to benchmark (default: all)")
    parser.add_argument("--custom_tiers", nargs="*", default=None,
                        help="Custom tiers as Q_N:K_N or Q_N (square). Must be powers of 2.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA required.")
    torch.manual_seed(args.seed)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Resolve tiers
    if args.custom_tiers:
        tiers = [_parse_tier(s, i) for i, s in enumerate(args.custom_tiers)]
    else:
        tiers = DEFAULT_TIERS

    # Validate all tiers
    for label, q_n, k_n, hd, *_ in tiers:
        _validate_tier(label, q_n, k_n, hd)

    kernels = args.kernels or KERNEL_NAMES

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for kernel_name in kernels:
        kernel_fn = _get_kernel_fn(kernel_name)
        for tier in tiers:
            label, q_n, k_n, hd, nh, bs = tier
            tag = f"{kernel_name} / {label} (Q={q_n}, K={k_n})"
            print(f"\n--- {tag} ---")
            try:
                row = bench_one(kernel_name, kernel_fn, label, q_n, k_n, hd, nh, bs,
                                args.warmup, args.iters, args.device, dtype)
                results.append(row)
                print(f"  avg={row['avg_ms']:.3f} ms  std={row['std_ms']:.3f} ms")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "kernel": kernel_name, "tier": label, "Q_N": q_n, "K_N": k_n,
                    "head_dim": hd, "num_heads": nh, "batch_size": bs,
                    "avg_ms": 0, "std_ms": 0, "min_ms": 0, "error": str(e),
                })

    # Save CSV
    csv_path = out_dir / "attention_sweep_results.csv"
    if results:
        # use union of all keys to tolerate optional 'error' field
        keys = sorted({k for r in results for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\n[csv] Saved {csv_path}")

    _plot_results(results, out_dir)
    _write_markdown(results, out_dir)
    _write_pdf_tables(results, out_dir)
    print("\n[benchmark_attention_sweep] Done.")


if __name__ == "__main__":
    main()
