"""Regenerate every figure on the README from the CSVs in ../results/final.

    cd benchmarks/compare/figures && ../../../.venv/bin/python make_figures.py

Colour = identity, fixed order: blue = this repo, orange = SGLang, aqua = vLLM (light-mode categorical slots 1-3, validated
all-pairs); the published *default* (CUDA-graph) configurations reuse the framework's hue, dashed. Text is always ink, never series colour.
The DDP numbers are the only hand-entered data: their original CSV was lost with the rented machine (see README).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "final"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a998f"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(HERE / name, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------------------------- data
def serving_stage_d():
    d = defaultdict(dict)
    for r in csv.DictReader(open(RES / "serving_stage_d.csv")):
        if r["experiment"] == "I1":
            d[(r["system"], int(r["concurrency"]))][r["metric"]] = float(r["value"])
    return d


def open_loop():
    d = defaultdict(dict)
    for r in csv.DictReader(open(RES / "serving_stage_d.csv")):
        if r["experiment"] == "I2":
            rate = int(r["variant"].split("rate=")[1].split(";")[0])
            d[(r["system"], rate)][r["metric"]] = float(r["value"])
    return d


SYS = {
    "serial": "engine_seq_kv_bfloat16",
    "cb": "engine_cb_naive_bfloat16",
    "fa2": "engine_cb_fa2_bfloat16",
    "prefill": "engine_cb_fa2_prefill_bfloat16",
    "final": "engine_cb_fa2_prefill_compile_bfloat16",
    "sg_cb": "sglang_cb_only_flashinfer_bfloat16",
    "sg_def": "sglang_default_noradix_flashinfer_bfloat16_gb64",
    "vl_ng": "vllm_nograph_noprefix_bfloat16",
    "vl_def": "vllm_default_noprefix_bfloat16",
}
CONC = [1, 8, 32, 64]


# ------------------------------------------------------------------------------ 1. throughput vs concurrency
def fig_throughput():
    d = serving_stage_d()
    tp = lambda k, c: d[(SYS[k], c)]["output_tok_per_s"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    spec = [  # key, colour, style, label
        ("final", BLUE, "-", "This engine (final)"),
        ("cb", BLUE, ":", "This engine, continuous batching only"),
        ("sg_cb", ORANGE, "-", "SGLang, same feature set"),
        ("vl_ng", AQUA, "-", "vLLM, no CUDA graph"),
        ("sg_def", ORANGE, "--", "SGLang default (CUDA graph)"),
        ("vl_def", AQUA, "--", "vLLM default (CUDA graph)"),
    ]
    for k, col, ls, lab in spec:
        ys = [tp(k, c) for c in CONC]
        ax.plot(range(4), ys, color=col, ls=ls, lw=2.2 if ls == "-" else 1.6, marker="o", ms=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        dy = {"vl_def": 7, "sg_def": -7}.get(k, 0)
        ax.annotate(lab, (3, ys[-1]), xytext=(9, dy), textcoords="offset points", va="center", fontsize=9.5, color=INK2)
    ax.axhline(tp("serial", 1), color=GRAY, lw=1.2, ls=(0, (2, 2)))
    ax.annotate("serial, batch 1: 47 tok/s", (2, tp("serial", 1)), xytext=(0, 8), textcoords="offset points",
                fontsize=9.5, color=INK2, ha="center")
    ax.set_xticks(range(4), [f"C={c}" for c in CONC])
    ax.set_xlim(-0.15, 3.0)
    ax.set_ylabel("output tokens / s")
    ax.set_title("Throughput vs. concurrency (190M, bf16, RTX 3090 Ti)")
    ax.margins(y=0.06)
    fig.subplots_adjust(right=0.60)
    save(fig, "serving_throughput.png")


# ------------------------------------------------------------------------------------ 2. optimisation ladder
def fig_ladder():
    d = serving_stage_d()
    v = lambda k: d[(SYS[k], 64)]["output_tok_per_s"] if k != "serial" else d[(SYS[k], 1)]["output_tok_per_s"]
    rungs = [("one request at a time\n+ KV cache", "serial"), ("+ continuous batching", "cb"),
             ("+ pre-allocated KV, FA-2 decode kernel", "fa2"), ("+ batched prefill", "prefill"),
             ("+ torch.compile", "final")]
    vals = [v(k) for _, k in rungs]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ys = list(range(len(rungs)))[::-1]
    cols = [GRAY, BLUE, BLUE, BLUE, BLUE]
    ax.barh(ys, vals, color=cols, height=0.56, edgecolor=SURFACE, linewidth=2)
    for y, val, (lab, k) in zip(ys, vals, rungs):
        ax.text(-30, y, lab, ha="right", va="center", fontsize=10, color=INK)
        ax.text(val + 25, y, f"{val:,.0f}", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(vals[1] + 330, ys[1], f"{vals[1] / vals[0]:.0f}× over serial", va="center", fontsize=9.5, color=INK2)
    ax.text(vals[4] + 560, ys[4] + 0.12, f"{vals[4] / vals[1]:.1f}× over continuous batching", va="center", fontsize=9.5, color=INK2)
    ax.text(vals[4] + 560, ys[4] - 0.22, f"{vals[4] / vals[0]:.0f}× over serial", va="center", fontsize=9.5, color=INK2, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 4700)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("output tokens / s at 64 concurrent requests")
    ax.set_title("Where the 60× comes from")
    fig.subplots_adjust(left=0.36)
    save(fig, "serving_ladder.png")


# ------------------------------------------------------------------------------------- 3. vs SGLang / vLLM
def fig_relative():
    d = serving_stage_d()
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
    for ax, c in zip(axs, (32, 64)):
        ours = d[(SYS["final"], c)]["output_tok_per_s"]
        refs = [("SGLang\nsame feature\nset", "sg_cb", ORANGE), ("vLLM\nno CUDA\ngraph", "vl_ng", AQUA),
                ("SGLang\ndefault", "sg_def", ORANGE), ("vLLM\ndefault", "vl_def", AQUA)]
        pct = [100 * ours / d[(SYS[k], c)]["output_tok_per_s"] for _, k, _ in refs]
        xs = range(len(refs))
        bars = ax.bar(xs, pct, color=[col for *_, col in refs], width=0.55, edgecolor=SURFACE, linewidth=2)
        for b, (_, k, col) in zip(bars, refs):
            if k.endswith("def"):
                b.set_hatch("///"); b.set_edgecolor(SURFACE); b.set_alpha(0.55)
        for x, p in zip(xs, pct):
            ax.text(x, p + 1.5, f"{p:.0f}%", ha="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.axhline(100, color=GRAY, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(list(xs), [r[0] for r in refs], fontsize=9)
        ax.set_title(f"C={c}", fontsize=11)
        ax.set_ylim(0, 118)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("this engine's throughput\nas % of the reference")
    fig.suptitle("Throughput relative to SGLang and vLLM (same GPU, same workload)", x=0.125, ha="left", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "serving_vs_frameworks.png")


# --------------------------------------------------------------------------------------- 4. open loop + SLO
def fig_open_loop():
    d = open_loop()
    rates = [10, 50, 75]
    series = [("engine_cb_fa2_prefill_compile_bfloat16", BLUE, "-", "This engine"),
              ("sglang_cb_only_flashinfer_bfloat16", ORANGE, "-", "SGLang, same feature set"),
              ("sglang_default_noradix_flashinfer_bfloat16_gb64", ORANGE, "--", "SGLang default")]
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axs[0]
    ax.plot([0, 80], [0, 80], color=GRAY, lw=1, ls=(0, (3, 3)), label="ideal: every request completes")
    for s, col, ls, lab in series:
        ys = [d[(s, r)]["request_per_s"] for r in rates]
        ax.plot(rates, ys, color=col, ls=ls, lw=2.2 if ls == "-" else 1.6, marker="o", ms=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab)
    ax.text(53, 30, "this engine saturates\nat ~43 req/s", fontsize=9.5, color=INK2, va="center")
    ax.set_xlabel("offered load (requests / s, Poisson arrivals)")
    ax.set_ylabel("completed requests / s")
    ax.set_xlim(0, 80); ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Open-loop load")
    ax = axs[1]
    w = 0.26
    for i, (s, col, ls, lab) in enumerate(series):
        att = [100 * d[(s, r)]["goodput_req_per_s"] / d[(s, r)]["request_per_s"] for r in rates]
        xs = [j + (i - 1) * w for j in range(3)]
        bars = ax.bar(xs, att, width=w, color=col, edgecolor=SURFACE, linewidth=2, label=lab)
        if ls == "--":
            for b in bars: b.set_hatch("///"); b.set_alpha(0.55)
        for x, a in zip(xs, att):
            ax.text(x, a + 2, f"{a:.0f}%", ha="center", fontsize=8.8, color=INK)
    ax.set_xticks(range(3), [f"{r} req/s" for r in rates])
    ax.set_ylim(0, 118); ax.grid(axis="x", visible=False)
    ax.set_ylabel("requests meeting the SLO")
    ax.set_title("SLO attainment  (TTFT ≤ 0.5 s and TPOT ≤ 50 ms)")
    fig.tight_layout()
    save(fig, "serving_open_loop.png")


# ------------------------------------------------------------------------------------------- 5. attention
def fig_attention():
    rows = [r for r in csv.DictReader(open(RES / "attention.csv"))
            if r["experiment"] == "A1" and r["forward_only"] == "True" and r["seq_len"]]
    tab = defaultdict(dict)
    for r in rows:
        tab[r["system"]][(int(r["seq_len"]), r["metric"])] = float(r["value"]) if r["value"] else None
    seqs = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    spec = [("attention", ORANGE, "-", "naive attention (materialises N×N)"),
            ("torch_compile_attention", AQUA, "-", "torch.compile'd attention"),
            ("triton_fa_mine", BLUE, "-", "Triton FlashAttention-2 (this repo)"),
            ("sdpa_flash", GRAY, "--", "PyTorch SDPA flash (reference)")]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, metric, ylabel, title in ((axs[0], "peak_mem_delta_mb", "extra peak memory (MiB)", "Memory"),
                                      (axs[1], "latency_ms", "forward latency (ms)", "Latency")):
        for s, col, ls, lab in spec:
            xs = [q for q in seqs if tab[s].get((q, metric)) is not None]
            ys = [tab[s][(q, metric)] for q in xs]
            ax.plot(xs, ys, color=col, ls=ls, lw=2.2 if ls == "-" else 1.5, marker="o", ms=4.5, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3 if s == "triton_fa_mine" else 2)
            if s == "attention" and xs[-1] < seqs[-1]:
                ax.plot([xs[-1]], [ys[-1]], marker="X", ms=10, color=col, markeredgecolor=SURFACE)
                ax.annotate("OOM at 8,192", (xs[-1], ys[-1]), xytext=(-6, 12), textcoords="offset points", fontsize=9, color=INK2, ha="right")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(seqs, [f"{q // 1024}K" if q >= 1024 else str(q) for q in seqs], fontsize=8.5)
        ax.minorticks_off()
        ax.set_xlabel("sequence length (batch × sequence fixed at 32,768 tokens)")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    m = lambda s, q: tab[s][(q, "peak_mem_delta_mb")]
    l = lambda s, q: tab[s][(q, "latency_ms")]
    axs[0].annotate(f"4K: {m('attention', 4096) / 1024:.1f} GiB → {m('triton_fa_mine', 4096):.0f} MiB", (4096, m("triton_fa_mine", 4096)),
                    xytext=(0, 38), textcoords="offset points", ha="center", fontsize=9.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axs[1].text(16384, 0.62, f"at 4K, this kernel is\n{l('attention', 4096) / l('triton_fa_mine', 4096):.1f}× faster than naive attention\n{l('torch_compile_attention', 4096) / l('triton_fa_mine', 4096):.1f}× faster than torch.compile",
                ha="right", va="center", fontsize=9.5, color=INK, linespacing=1.4)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2.2 if ls == "-" else 1.5, marker="o", ms=4.5) for _, c, ls, _ in spec]
    fig.legend(handles, [s[3] for s in spec], loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Attention forward, bf16, causal, 12 heads × head_dim 64 — RTX 3090 Ti", x=0.065, ha="left", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "attention_forward.png")


# ----------------------------------------------------------------------------------------------- 6. DDP
def fig_ddp():
    # transcribed from training/artifacts/lm_matrix_table_flash_attention_triton.png (original CSV lost with the instance)
    names = ["1 GPU\n(no DDP)", "naive DDP\n(per-parameter all-reduce)", "bucketed + overlapped\n(this repo)", "PyTorch\nDistributedDataParallel"]
    tps = [16132.4, 25336.7, 27719.1, 27692.6]
    eff = [None, 78.5, 85.9, 85.8]
    cols = [GRAY, GRAY, BLUE, GRAY]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars = ax.bar(range(4), [t / 1000 for t in tps], color=cols, width=0.58, edgecolor=SURFACE, linewidth=2)
    ax.plot([0.6, 3.4], [tps[0] * 2 / 1000] * 2, color=INK2, lw=1.1, ls=(0, (3, 3)))
    ax.text(3.4, tps[0] * 2 / 1000 + 0.5, "ideal 2× scaling", ha="right", fontsize=9, color=INK2)
    for x, t, e in zip(range(4), tps, eff):
        ax.text(x, t / 1000 + 0.5, f"{t / 1000:.1f}k tok/s", ha="center", fontsize=10, fontweight="bold")
        if e:
            ax.text(x, t / 1000 / 2, f"{e:.1f}%\nscaling\nefficiency", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
    ax.set_xticks(range(4), names, fontsize=9)
    ax.set_ylim(0, 38); ax.set_ylabel("training throughput (k tokens / s)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Data-parallel training, 2 × RTX 3090")
    ax.text(2, 3.2, "+3.3% peak\nmemory", ha="center", va="center", fontsize=9, color="white")
    save(fig, "ddp_scaling.png")


# ------------------------------------------------------------------------------------- 7/8. diagrams
def box(ax, x, y, w, h, text, fc="#ffffff", ec=GRID, fs=9.5, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, a, b, color=INK2, style="-|>", rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.grid(False)
    return fig, ax


def fig_stack():
    fig, ax = canvas(11.4, 6.6)
    def band(y, h, title):
        ax.add_patch(FancyBboxPatch((0.15, y), 11.1, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f3f2ee", ec="none"))
        ax.text(0.4, y + h - 0.28, title, fontsize=10.5, fontweight="bold", color=INK2)
    band(4.55, 1.85, "MODEL  —  PyTorch tensors only")
    box(ax, 0.4, 4.7, 2.0, 1.1, "BPE tokenizer\n(Python + C++)")
    box(ax, 2.85, 4.7, 4.9, 1.1, "Llama-style Transformer\nMHA · RMSNorm pre-norm · RoPE · SwiGLU FFN\nmatches HF LlamaForCausalLM logits (fp32)", fs=9.2)
    box(ax, 8.2, 4.7, 2.8, 1.1, "AdamW · top-p sampling")
    arrow(ax, (2.4, 5.25), (2.85, 5.25)); arrow(ax, (7.75, 5.25), (8.2, 5.25))
    band(2.45, 1.9, "TRAINING SYSTEMS")
    box(ax, 0.4, 2.6, 5.1, 1.1, "Triton FlashAttention-2 forward\nmixed precision · online softmax · autotune\n4K tokens: 9.0 GiB → 49 MiB, 3.0× faster than torch.compile", fs=9.2)
    box(ax, 5.9, 2.6, 5.1, 1.1, "NCCL data-parallel training\ngradient bucketing + communication/compute overlap\n86% scaling on 2 GPUs, on par with PyTorch DDP", fs=9.2)
    band(0.1, 2.15, "SERVING  —  asyncio inference engine")
    box(ax, 0.4, 0.5, 2.3, 1.1, "RequestReceiver\ntokenize · enqueue")
    box(ax, 3.2, 0.5, 1.9, 1.1, "asyncio.Queue")
    box(ax, 5.6, 0.5, 3.3, 1.1, "Engine loop\nadmit → prefill → decode\ncontinuous batching", bold=True, ec=BLUE, fs=9.2)
    box(ax, 9.4, 0.5, 1.6, 1.1, "request_store\nstatus · tokens", fs=9)
    for a, b in ((2.7, 3.2), (5.1, 5.6), (8.9, 9.4)): arrow(ax, (a, 1.05), (b, 1.05))
    ax.text(0.4, 0.22, "KV cache · batched prefill · pre-allocated slots · torch.compile · greedy output = HF generate (fp32)", fontsize=8.8, color=INK2)
    save(fig, "stack_overview.png")


def fig_engine_loop():
    fig, ax = canvas(11.4, 5.2)
    ax.text(0.2, 4.85, "One scheduling round", fontsize=12, fontweight="bold")
    steps = [("① admit", "drain the queue\n(block only\nwhen idle)"), ("② prefill", "up to N requests\nper round (N tunable)\none batched forward"),
             ("③ decode", "every in-flight request\nadvances one token:\none batched forward"), ("④ join / leave", "prefilled requests\njoin the batch;\nfinished ones leave"),
             ("⑤ yield", "await sleep(0):\nreceiver and\nshutdown() can run")]
    w, gap, x0 = 1.95, 0.3, 0.2
    for i, (t, d) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, 2.75, w, 1.5, "", ec=BLUE if i in (1, 2) else GRID)
        ax.text(x + w / 2, 3.95, t, ha="center", fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, 3.3, d, ha="center", va="center", fontsize=8.8, color=INK2, linespacing=1.35)
        if i < 4: arrow(ax, (x + w, 3.5), (x + w + gap, 3.5))
    arrow(ax, (x0 + 4 * (w + gap) + w / 2, 2.75), (x0 + w / 2, 2.75), rad=-0.18)
    ax.text(5.7, 2.4, "repeat while any request is in flight", ha="center", fontsize=9, color=INK2)
    ax.text(0.2, 1.45, "Engine lifecycle", fontsize=12, fontweight="bold")
    st = ["INIT", "RUNNING", "DRAINING", "STOPPED"]
    for i, s in enumerate(st):
        box(ax, 0.2 + i * 2.15, 0.5, 1.55, 0.6, s, fc="#ffffff", ec=GRID, bold=True, fs=9.5)
        if i < 3: arrow(ax, (1.75 + i * 2.15, 0.8), (2.35 + i * 2.15, 0.8))
    ax.text(9.0, 0.95, "shutdown(drain=True): finish in-flight work", fontsize=9, color=INK2)
    ax.text(9.0, 0.55, "shutdown(drain=False): cancel immediately", fontsize=9, color=INK2)
    save(fig, "engine_loop.png")


if __name__ == "__main__":
    fig_stack(); fig_engine_loop(); fig_attention(); fig_ddp(); fig_ladder(); fig_throughput(); fig_relative(); fig_open_loop()
