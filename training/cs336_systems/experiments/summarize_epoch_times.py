"""Collapse the per-config `epoch_times__*.json` files written by benchmark_lm_matrix.py into
one `facts_v1.json` (mean / std / cv / min / max per config + the comparisons the video shows).

Why this exists: the video may only show numbers that trace back to a real run. This file is
the single hop between the raw per-epoch times and the screen.

Fairness guards (each one has bitten a benchmark before):
  * all OK configs must have the same `samples_per_epoch` -- otherwise an "epoch" means a
    different amount of data in different rows and epoch times are not comparable;
  * a failed config (status "oom" / "error") is reported as failed, never as a number;
  * a difference is called "resolved" only if it exceeds 2 standard errors of the difference
    (Welch); otherwise the verdict is "within noise" and the video must say so.

Usage:
    python summarize_epoch_times.py <dir with epoch_times__*.json> [-o facts_v1.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st

NAIVE_ATTN = "scaled_dot_prod_attention"
FLASH_ATTN = "flash_attention_triton"
LOCAL, NAIVE_DDP, OVERLAP_DDP = "Local No DDP", "Naive DDP", "Bucketed Overlapping DDP"

# (name, kernel_a, wrapper_a, kernel_b, wrapper_b): "b vs a" -> speedup = mean_a / mean_b
COMPARISONS = [
    ("ddp_naive_vs_single_gpu", NAIVE_ATTN, LOCAL, NAIVE_ATTN, NAIVE_DDP),
    ("ddp_overlap_vs_single_gpu", NAIVE_ATTN, LOCAL, NAIVE_ATTN, OVERLAP_DDP),
    ("ddp_overlap_vs_naive_ddp", NAIVE_ATTN, NAIVE_DDP, NAIVE_ATTN, OVERLAP_DDP),
    ("flash_vs_naive_attn_single_gpu", NAIVE_ATTN, LOCAL, FLASH_ATTN, LOCAL),
    ("flash_vs_naive_attn_naive_ddp", NAIVE_ATTN, NAIVE_DDP, FLASH_ATTN, NAIVE_DDP),
    ("flash_vs_naive_attn_overlap_ddp", NAIVE_ATTN, OVERLAP_DDP, FLASH_ATTN, OVERLAP_DDP),
    ("everything_vs_baseline", NAIVE_ATTN, LOCAL, FLASH_ATTN, OVERLAP_DDP),
    # DDP measured with FlashAttention only (the compute-heavy batch where naive attention OOMs)
    ("flash_ddp_naive_vs_single_gpu", FLASH_ATTN, LOCAL, FLASH_ATTN, NAIVE_DDP),
    ("flash_ddp_overlap_vs_single_gpu", FLASH_ATTN, LOCAL, FLASH_ATTN, OVERLAP_DDP),
    ("flash_ddp_overlap_vs_naive_ddp", FLASH_ATTN, NAIVE_DDP, FLASH_ATTN, OVERLAP_DDP),
]


def load_rows(directory: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "epoch_times__*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(f"no epoch_times__*.json in {directory}")
    return rows


def config_stats(row: dict) -> dict:
    times = row.get("epoch_times_s") or []
    status = row.get("status", "error")
    if status == "ok" and len(times) < 2:
        status = "too_few_epochs"  # a std needs >= 2 epochs
    ok = status == "ok"
    out = {
        "kernel": row["kernel"],
        "wrapper": row["ddp"],
        "status": status,
        "gpus": row.get("gpus"),
        "context_length": row.get("context_length"),
        "local_batch_size": row.get("local_batch_size"),
        "steps_per_epoch": row.get("steps_per_epoch"),
        "samples_per_epoch": row.get("samples_per_epoch"),
        "epoch_times_s": times,
    }
    if ok:
        mean = st.mean(times)
        sd = st.stdev(times)
        out.update(
            n=len(times), mean_s=mean, std_s=sd, cv_pct=100 * sd / mean,
            min_s=min(times), max_s=max(times),
            tokens_per_sec=row.get("tokens_per_sec"), peak_gpu_mb=row.get("peak_gpu_mb"),
        )
    return out


def compare(a: dict, b: dict) -> dict:
    """b relative to a. speedup > 1 means b is faster. Welch standard error of the difference."""
    se = math.sqrt(a["std_s"] ** 2 / a["n"] + b["std_s"] ** 2 / b["n"])
    diff = a["mean_s"] - b["mean_s"]
    resolved = se > 0 and abs(diff) > 2 * se
    return {
        "speedup": a["mean_s"] / b["mean_s"],
        "saved_pct": 100 * diff / a["mean_s"],
        "diff_s": diff,
        "se_s": se,
        "verdict": ("resolved" if resolved else "within_noise"),
    }


def summarize(rows: list[dict]) -> dict:
    configs = {f"{c['kernel']}|{c['wrapper']}": c for c in map(config_stats, rows)}

    ok_samples = {c["samples_per_epoch"] for c in configs.values() if c["status"] == "ok"}
    if len(ok_samples) > 1:
        raise SystemExit(
            f"configs disagree on samples_per_epoch {sorted(ok_samples)}: epoch times are not "
            "comparable. Re-run with --samples_per_epoch."
        )

    comparisons = []
    for name, ka, wa, kb, wb in COMPARISONS:
        a, b = configs.get(f"{ka}|{wa}"), configs.get(f"{kb}|{wb}")
        entry = {"name": name, "a": f"{ka}|{wa}", "b": f"{kb}|{wb}"}
        if a and b and a["status"] == "ok" and b["status"] == "ok":
            entry.update(compare(a, b))
        else:
            missing = [k for k, c in (("a", a), ("b", b)) if not c or c["status"] != "ok"]
            entry["unavailable"] = {
                k: (configs.get(entry[k]) or {}).get("status", "not_run") for k in missing
            }
        comparisons.append(entry)

    return {
        "meta": next((r["meta"] for r in rows if r.get("meta")), {}),
        "samples_per_epoch": next(iter(ok_samples), None),
        "configs": configs,
        "comparisons": comparisons,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    facts = summarize(load_rows(args.directory))
    out = args.out or os.path.join(args.directory, "facts_v1.json")
    with open(out, "w") as f:
        json.dump(facts, f, indent=2)
    print(f"wrote {out}")
    for c in facts["configs"].values():
        if c["status"] == "ok":
            print(f"{c['kernel']:<28}{c['wrapper']:<28}{c['mean_s']:>8.3f} ± {c['std_s']:.3f} s  "
                  f"(cv {c['cv_pct']:.1f}%, n={c['n']})")
        else:
            print(f"{c['kernel']:<28}{c['wrapper']:<28}{c['status'].upper()}")
    for cmp_ in facts["comparisons"]:
        if "speedup" in cmp_:
            print(f"{cmp_['name']:<36} x{cmp_['speedup']:.3f}  ({cmp_['verdict']})")
        else:
            print(f"{cmp_['name']:<36} unavailable {cmp_['unavailable']}")


if __name__ == "__main__":
    main()
