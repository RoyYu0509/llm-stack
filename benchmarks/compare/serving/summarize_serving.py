"""Tables from results/final/serving.csv (written by closed_loop.py run / ingest).

Presentation rules (D-7): the headline row is the baseline engine (naive attention); arms are labelled; rows are GROUPED by
attention kernel type and each group also gets the mean of its members, so "what does the attention kernel buy" can be read off
without mixing it with "which system is it".

Usage: python summarize_serving.py [--csv ../results/final/serving.csv]
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# system label in the CSV -> (attention kernel group, display name, who wrote it)
SYSTEMS = {
    "engine_cb_naive_bfloat16":    ("naive (materialises scores)", "baseline engine, naive attention", "base"),
    "hf_generate_eager_bfloat16":  ("naive (materialises scores)", "HF generate, eager", "HF"),
    "engine_cb_sdpa_bfloat16":     ("SDPA (PyTorch fused)", "engine + F.sdpa (wrapper)", "opt"),
    "hf_generate_sdpa_bfloat16":   ("SDPA (PyTorch fused)", "HF generate, sdpa", "HF"),
    "engine_cb_fa2_bfloat16":      ("flash-attn (FA2)", "engine fa2 (prealloc KV + FA2 decode + batched argmax)", "opt"),
    "hf_generate_flash_attention_2_bfloat16": ("flash-attn (FA2)", "HF generate, flash_attention_2", "HF"),
    "sglang_cb_only_flashinfer_bfloat16": ("flashinfer", "SGLang cb_only (flashinfer)", "SGLang"),
    "sglang_default_noradix_flashinfer_bfloat16_gb64": ("flashinfer", "SGLang default minus radix (flashinfer, CUDA graph)", "SGLang"),
    "sglang_cb_only_bfloat16":     ("triton (SGLang)", "SGLang cb_only (triton)", "SGLang"),
}
GROUP_ORDER = ["naive (materialises scores)", "SDPA (PyTorch fused)", "flash-attn (FA2)", "flashinfer", "triton (SGLang)"]
HEADLINE = "engine_cb_naive_bfloat16"
METRICS = [("output_tok_per_s", "tok/s"), ("ttft_ms_p50", "TTFT p50 ms"), ("tpot_ms_p50", "TPOT p50 ms"), ("itl_ms_p99", "ITL p99 ms")]


def load(csv_path: Path):
    data = defaultdict(dict)       # (system, C) -> metric -> (value, std, n)
    for r in csv.DictReader(open(csv_path)):
        if r["module"] != "serving" or r["experiment"] != "I1" or not r["concurrency"]:
            continue
        data[(r["system"], int(r["concurrency"]))][r["metric"]] = (float(r["value"]), float(r["std"] or 0), int(r["n_reps"]))
    return data


def render(data) -> str:
    cs = sorted({c for (_, c) in data})
    unknown = sorted({s for (s, _) in data} - set(SYSTEMS))
    out = ["# Serving, bf16, closed loop, mixed output lengths 32-96, random-init 190M weights (speed only); RTX 3090 Ti",
           "'ratio' = tok/s relative to the baseline engine (naive); opt = optimised-path arm; '~' = std/mean > 10%",
           "Group means mix systems whose non-attention design differs a lot (HF generate is serial; SGLang adds CUDA graphs, a token-level KV pool, "
           "a C++/CUDA scheduler): read them as 'what the group looks like', never as 'what that attention kernel is worth'.", ""]
    for c in cs:
        base = data.get((HEADLINE, c), {}).get("output_tok_per_s", (None,))[0]
        out += [f"## C = {c}", "", "| attention kernel | system | by | tok/s | ratio | TTFT p50 ms | TPOT p50 ms | ITL p99 ms |", "|---|---|---|---|---|---|---|---|"]
        for g in GROUP_ORDER:
            members = [(s, v) for s, v in SYSTEMS.items() if v[0] == g and (s, c) in data]
            for s, (_, name, by) in members:
                m = data[(s, c)]
                tp, sd, _ = m.get("output_tok_per_s", (float("nan"), 0, 0))
                mark = "~" if tp and sd / tp > 0.10 else ""
                ratio = f"{tp / base:.2f}x" if base else "-"
                cells = [f"{m[k][0]:.1f}" if k in m else "-" for k in ("ttft_ms_p50", "tpot_ms_p50", "itl_ms_p99")]
                out.append(f"| {g} | {name} | {by} | {tp:.0f}{mark} | {ratio} | {' | '.join(cells)} |")
            if len(members) > 1:
                vals = [data[(s, c)]["output_tok_per_s"][0] for s, _ in members if "output_tok_per_s" in data[(s, c)]]
                out.append(f"| **{g}: mean of {len(vals)}** | | | **{statistics.fmean(vals):.0f}** | "
                           f"{(statistics.fmean(vals) / base):.2f}x | | | |" if base else "| | | | | | | | |")
        out.append("")
    if unknown:
        out += [f"(systems in the CSV without a display mapping, not shown: {unknown})"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=HERE.parent / "results" / "final" / "serving.csv")
    args = ap.parse_args(argv)
    print(render(load(args.csv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
