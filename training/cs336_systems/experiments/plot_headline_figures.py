"""Regenerate the headline figures the top-level README embeds.

Why this script exists separately from `benchmark_lm_matrix.py`
--------------------------------------------------------------
`benchmark_lm_matrix.py` emits a default-styled bar chart per run. Those defaults
colour every bar a different hue, which encodes nothing -- the four bars are four
values of one measure, not four series -- and they draw no reference line, so the
one number the chart exists to communicate (scaling efficiency against ideal
linear speedup) is not actually visible in it. These figures are the
README-facing versions: one emphasis colour, ideal-linear reference, direct
labels.

Provenance of the numbers -- read before changing them
-----------------------------------------------------
FIGURE 1 (`ddp_scaling.png`) plots the 2x RTX 3090 run that the README quotes.
Its results CSV did not survive: the rented instance was destroyed and the remote
data is permanently gone (see `docs/ROADMAP-month.md`, Context item 2). What DID
survive is the rendered summary table committed at
`artifacts/lm_matrix_table_flash_attention_triton.png`, and the four rows below
are transcribed from it verbatim. `training/CLAUDE.md` names that PNG as the
source of record for these numbers.

So: these constants are transcribed from a committed artifact, not recomputed
from raw data, and re-running the benchmark on different hardware will NOT
reproduce them. That is why they are hardcoded here with this note instead of
being read from a CSV that does not exist.

A bucket-size sweep figure was here and was REMOVED on 2026-09-05. The five
`artifacts/bench_bucket_*/` runs it plotted do not record which bucket size each
one used -- `benchmark_lm_matrix.py` wrote a fixed column whitelist that omitted
it, there was no CLI flag, and both configs read `bucket_size_mb: 25`. The only
thing distinguishing the five runs was the output directory's name, so the
experiment cannot tell "five bucket sizes" apart from "one bucket size, five
times", and the observed spread (four runs within 2.6%, one outlier at +18%)
is what five identical runs would look like. Plotting it implied a causal
variable that was never shown to vary.

`benchmark_lm_matrix.py` now records `bucket_size_mb` in its CSV and accepts
`--bucket_size_mb`, so a future sweep is falsifiable. Re-running it needs 2 CUDA
GPUs, which is why the figure is deleted rather than regenerated.

Usage:
    uv run python cs336_systems/experiments/plot_headline_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- design tokens
# Two colours carry meaning: one emphasis, one de-emphasis. Validated against the
# light chart surface (#fcfcfb): contrast >= 3:1 for both, CVD dE 15.9 (protan) /
# 11.2 (tritan), normal-vision dE 17.9 -- all clear of the >=8 / >=15 floors.
EMPHASIS = "#2a78d6"
MUTED = "#8a8983"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")


def _style(ax):
    """Hairline, recessive chrome: solid grid one shade off the surface, no box."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, colors=INK_2, labelsize=10)


# --------------------------------------------------------------------- figure 1
# Transcribed from artifacts/lm_matrix_table_flash_attention_triton.png -- see the
# module docstring. Do not "correct" these against bench_matrix/*.csv; that is a
# different, smaller run.
DDP_ROWS = [
    ("Local, no DDP\n1 GPU", 16132.4, None),
    ("Naive DDP\n2 GPU", 25336.7, 78.5),
    ("Bucketed + overlapped\n2 GPU  (mine)", 27719.1, 85.9),
    ("PyTorch official DDP\n2 GPU", 27692.6, 85.8),
]
SINGLE_GPU = 16132.4
IDEAL_2GPU = 2 * SINGLE_GPU


def figure_ddp_scaling(out_path):
    labels = [r[0] for r in DDP_ROWS]
    values = [r[1] for r in DDP_ROWS]
    effs = [r[2] for r in DDP_ROWS]
    colors = [EMPHASIS if "mine" in lab else MUTED for lab in labels]

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=120)
    x = range(len(values))
    ax.bar(x, values, width=0.56, color=colors, linewidth=0, zorder=2)

    ax.axhline(
        IDEAL_2GPU, color=INK_2, linewidth=1.2, linestyle=(0, (5, 4)), zorder=3
    )
    ax.text(
        -0.46,
        IDEAL_2GPU + 450,
        f"ideal 2x linear scaling  ({IDEAL_2GPU:,.0f} tok/s)",
        ha="left", va="bottom", fontsize=9.5, color=INK_2,
    )

    for xi, (val, eff) in enumerate(zip(values, effs)):
        ax.text(xi, val + 600, f"{val:,.0f}", ha="center", va="bottom",
                fontsize=12, color=INK, fontweight="bold")
        if eff is not None:
            ax.text(xi, val / 2, f"{eff}%\nof ideal", ha="center", va="center",
                    fontsize=10.5, color="white", linespacing=1.4)

    _style(ax)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10.5, color=INK)
    ax.set_ylabel("Training throughput (tokens / sec, aggregate)",
                  fontsize=10.5, color=INK_2)
    ax.set_ylim(0, IDEAL_2GPU * 1.20)
    ax.set_title(
        "Custom bucketed + overlapped DDP matches PyTorch's official DDP\n"
        "27,719 vs 27,693 tok/s - a 0.1% gap, within run-to-run noise",
        fontsize=13, color=INK, fontweight="bold", loc="left", pad=14,
    )
    fig.text(0.008, 0.015,
             "2x RTX 3090 - flash_attention_triton kernel - "
             "source: artifacts/lm_matrix_table_flash_attention_triton.png",
             fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------- figure 3
# Two series -> a legend is required; <=4 series -> also direct-labelled.
# #2a78d6 / #eb6834 pass all six checks on the light surface (CVD dE 24.7,
# normal-vision dE 33.6, both >= 3:1 contrast).
SERIES = {"recompute": "#eb6834", "cache": "#2a78d6"}
SERIES_LABEL = {
    "recompute": "recompute decode  (no KV cache)",
    "cache": "KV-cache decode",
}


def figure_kv_cache_decode(out_path):
    src = os.path.join(ARTIFACTS, "serving_benchmark.json")
    if not os.path.exists(src):
        return None
    import json

    d = json.load(open(src))
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=120)

    for mode in ("recompute", "cache"):
        runs = d["modes"][mode]["runs"]
        xs = [r["prompt_tokens"] for r in runs]
        ys = [r["decode_p50_s"] * 1e3 for r in runs]
        ax.plot(xs, ys, color=SERIES[mode], linewidth=2, zorder=3,
                label=SERIES_LABEL[mode])
        ax.plot(xs, ys, marker="o", markersize=8, linestyle="none",
                color=SERIES[mode], markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
        # Direct-label the endpoint only, not every point.
        ax.text(xs[-1] * 1.06, ys[-1], f"{ys[-1]:.1f} ms",
                fontsize=11, color=INK, va="center", fontweight="bold")

    rc = d["modes"]["recompute"]["runs"][-1]["decode_p50_s"] * 1e3
    ca = d["modes"]["cache"]["runs"][-1]["decode_p50_s"] * 1e3
    ax.annotate(
        "", xy=(512, rc), xytext=(512, ca),
        arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=1.1))
    ax.text(455, (rc + ca) / 2, f"{rc / ca:.1f}x", ha="right", va="center",
            fontsize=13, color=INK, fontweight="bold")

    _style(ax)
    ax.set_xscale("log")
    ax.set_xticks([16, 64, 128, 256, 512])
    ax.set_xticklabels(["16", "64", "128", "256", "512"], fontsize=10.5, color=INK)
    ax.minorticks_off()
    ax.set_xlim(13, 780)
    ax.set_ylim(0, 245)
    ax.set_xlabel("Prompt length (tokens)", fontsize=10.5, color=INK_2)
    ax.set_ylabel("Per-token decode latency, p50 (ms)", fontsize=10.5, color=INK_2)
    leg = ax.legend(frameon=False, fontsize=10.5, loc="upper left",
                    bbox_to_anchor=(0.02, 0.97))
    for t in leg.get_texts():
        t.set_color(INK)
    # Both ratios below are LATENCY ratios, matching the y-axis. The JSON's
    # cache_speedup_by_prompt_len is a THROUGHPUT ratio and differs slightly
    # (14.2x vs 15.2x at 512) -- mixing the two in one figure would be a bug.
    rc0 = d["modes"]["recompute"]["runs"][0]["decode_p50_s"]
    ca0 = d["modes"]["cache"]["runs"][0]["decode_p50_s"]
    ax.set_title(
        "Without a KV cache, every decode step re-runs the whole prefix\n"
        f"The gap widens with prompt length: {rc0 / ca0:.1f}x at 16 tokens, "
        f"{rc / ca:.1f}x at 512",
        fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=14,
    )
    fig.text(0.008, 0.015,
             f"0.19B TransformerLM, {d['new_tokens']} new tokens, CPU - "
             "source: artifacts/serving_benchmark.json, generated by bench_serving.py",
             fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    a = figure_ddp_scaling(os.path.join(ARTIFACTS, "ddp_scaling.png"))
    print("wrote", os.path.normpath(a))
    c = figure_kv_cache_decode(os.path.join(ARTIFACTS, "kv_cache_decode.png"))
    print("wrote", os.path.normpath(c) if c else "kv cache: run bench_serving.py first")
