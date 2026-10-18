"""One page for the KV-cache size sweep (run_kv_sweep.sh): per model size and prompt length, what the KV cache bought.

    python kv_sweep_summary.py [--csv .../results/final/serving_kv_sweep.csv] [--raw .../serving_raw/engine_kv]

Columns: prefill time (TTFT), per-token decode time (TPOT) without / with the KV cache, the decode speed-up TPOT_nokv / TPOT_kv, and the
peak GPU memory of the KV arm (read from the recordings; the card has 24 GiB). Cells that are in the grid but not in the csv are printed
as MISSING and the exit code is 1; extra cells (the 12288 fall-back) are printed too.

"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZES = ("190m", "1.4b", "2.8b", "6.9b")
LENS = {"190m": (1024, 4096, 16384), "1.4b": (1024, 4096, 16384), "2.8b": (1024, 4096, 16384), "6.9b": (1024, 4096, 12288)}   # 6.9b @ 16384 does not fit
SYSTEM = re.compile(r"engine_kvsweep_(nokv|kv)_(.+)_L(\d+)_bfloat16$")


def load(path: Path) -> dict:
    """{(arm, size, L): {metric: value}}"""
    cells: dict = {}
    if not path.exists():
        return cells
    for r in csv.DictReader(open(path)):
        m = SYSTEM.match(r["system"])
        if m:
            cells.setdefault((m.group(1), m.group(2), int(m.group(3))), {})[r["metric"]] = float(r["value"])
    return cells


def peak_gib(raw: Path | None, arm: str, size: str, L: int) -> float | None:
    """Largest peak_mem_gb over the repetitions of one cell (torch.cuda.max_memory_allocated, so below what nvidia-smi shows)."""
    if raw is None or not raw.exists():
        return None
    peaks = []
    for f in raw.glob(f"{arm}_{size}_L{L}_r*.json"):
        v = json.loads(f.read_text()).get("meta", {}).get("peak_mem_gb")
        if v is not None:
            peaks.append(float(v))
    return max(peaks) if peaks else None


def summarize(csv_path: Path, raw: Path | None = None) -> tuple[list[str], int]:
    cells = load(csv_path)
    wanted = [(a, s, L) for s in SIZES for L in LENS[s] for a in ("nokv", "kv")]
    planned = {(s, L) for s in SIZES for L in LENS[s]}
    extra = sorted(k for k in cells if k not in wanted)
    lines = [f"{'model':>6s} {'prompt':>7s} | {'TTFT ms':>9s} | {'TPOT ms nokv':>12s} {'TPOT ms kv':>11s} {'speed-up':>9s} | {'peak GiB (kv arm)':>17s}"]
    missing = 0
    rows = [(s, L) for s in SIZES for L in LENS[s]] + sorted({(s, L) for _, s, L in extra})
    for size, L in rows:
        no, kv = cells.get(("nokv", size, L)), cells.get(("kv", size, L))
        absent = [a for a, c in (("nokv", no), ("kv", kv)) if c is None or "tpot_ms_p50" not in c]
        if absent:
            if (size, L) in planned:
                missing += len(absent)
                lines.append(f"{size:>6s} {L:>7d} | MISSING {', '.join(absent)}")
            continue
        peak = peak_gib(raw, "kv", size, L)
        speed = no["tpot_ms_p50"] / kv["tpot_ms_p50"]
        note = "" if (size, L) in planned else "   (fall-back cell)"
        lines.append(f"{size:>6s} {L:>7d} | {kv.get('ttft_ms_p50', float('nan')):9.0f} | {no['tpot_ms_p50']:12.1f} {kv['tpot_ms_p50']:11.1f} "
                     f"{speed:8.1f}x | {('%.1f' % peak) if peak is not None else '-':>17s}{note}")
    total = len(wanted)
    lines.append(f"\n{total - missing}/{total} cells present" + ("" if not missing else f"; {missing} MISSING"))
    return lines, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=HERE.parent / "results" / "final" / "serving_kv_sweep.csv")
    ap.add_argument("--raw", type=Path, default=HERE.parent / "results" / "final" / "serving_raw" / "engine_kv")
    a = ap.parse_args(argv)
    lines, missing = summarize(a.csv, a.raw)
    print(f"KV-cache size sweep: {a.csv}")
    print("\n".join(lines))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
