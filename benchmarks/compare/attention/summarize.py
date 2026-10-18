"""Tables from results/final/attention.csv: latency per (system, seq), ratios against sdpa_flash, OOM map, memory, noise.

Only rows with experiment A1 are used; forward and forward+bwd are separate tables. A cell with std/mean above
`--noisy` (default 10%) is marked with '~' because a ratio built on it is not a claim.

Usage: python summarize.py [--csv ../results/final/attention.csv] [--dtype bfloat16]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["attention", "torch_compile_attention", "sdpa_flash", "flash_attn_pkg", "triton_fa_mine"]


def load(csv_path: Path, dtype: str = "bfloat16"):
    cells = defaultdict(dict)          # (forward_only, seq) -> system -> dict(ms, std, mem, tflops, status)
    for r in csv.DictReader(open(csv_path)):
        if r["module"] != "attention" or r["experiment"] != "A1" or (dtype and r["dtype"] != dtype):
            continue
        if not r["seq_len"]:      # e.g. the inductor fuse_attention_count rows are A1 rows without a sequence length
            continue
        sysd = cells[(r["forward_only"] == "True", int(r["seq_len"]))].setdefault(r["system"], {})
        m = r["metric"]
        if m == "latency_ms":
            sysd["ms"], sysd["std"], sysd["n"], sysd["batch"] = float(r["value"]), float(r["std"] or 0), int(r["n_reps"]), int(r["batch"])
        elif m == "tflops_counted":
            sysd["tflops"] = float(r["value"])
        elif m == "peak_mem_delta_mb":
            sysd["mem"] = float(r["value"])
        elif m in ("oom", "unavailable", "error", "crash", "timeout"):
            sysd["status"] = m
    return cells


def fmt_cell(c: dict | None, noisy: float) -> str:
    if not c:
        return "-"
    if "ms" in c:
        mark = "~" if c["ms"] > 0 and c["std"] / c["ms"] > noisy else ""
        return f"{c['ms']:.3f}{mark}"
    return c.get("status", "-").upper()


def order_for(cells) -> list[str]:
    """The five CV rows first, then any extra / diagnostic system that has data (e.g. flash_attn_pkg_flat)."""
    present = {k for row in cells.values() for k in row}
    return ORDER + sorted(present - set(ORDER))


def table(cells, forward_only: bool, noisy: float, what: str) -> list[str]:
    ORDER = order_for(cells)
    seqs = sorted(s for (fo, s) in cells if fo == forward_only)
    head = "| S (B) | " + " | ".join(ORDER) + " |"
    lines = [head, "|" + "---|" * (len(ORDER) + 1)]
    for s in seqs:
        row = cells[(forward_only, s)]
        b = next((row[k]["batch"] for k in row if "batch" in row[k]), "?")
        if what == "latency":
            vals = [fmt_cell(row.get(k), noisy) for k in ORDER]
        elif what == "ratio":      # system / sdpa_flash
            base = row.get("sdpa_flash", {}).get("ms")
            vals = [(f"{row[k]['ms'] / base:.2f}x" if base and "ms" in row.get(k, {}) else fmt_cell(row.get(k), noisy)) for k in ORDER]
        elif what == "tflops":
            vals = [(f"{row[k]['tflops']:.1f}" if "tflops" in row.get(k, {}) else fmt_cell(row.get(k), noisy)) for k in ORDER]
        else:                      # mem
            vals = [(f"{row[k]['mem']:.0f}" if "mem" in row.get(k, {}) else fmt_cell(row.get(k), noisy)) for k in ORDER]
        lines.append(f"| {s} ({b}) | " + " | ".join(vals) + " |")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=HERE.parent / "results" / "final" / "attention.csv")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--noisy", type=float, default=0.10)
    args = ap.parse_args(argv)
    cells = load(args.csv, args.dtype)
    out = [f"# Attention A1 ({args.dtype}, causal, 12 heads x hd64, B*S fixed)  '~' = std/mean > {args.noisy:.0%}"]
    for fo, name in ((True, "forward"), (False, "forward + backward")):
        if not any(k[0] == fo for k in cells):
            continue
        for what, title in (("latency", "latency (ms)"), ("ratio", "ratio to sdpa_flash (lower = faster than sdpa_flash)"),
                            ("tflops", "counted TFLOP/s"), ("mem", "peak memory delta (MiB)")):
            out += ["", f"## {name}: {title}", ""] + table(cells, fo, args.noisy, what)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
