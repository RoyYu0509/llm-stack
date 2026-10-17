"""D-7 gate: is `sdpa_flash` vs `flash_attn_pkg` (the standalone flash-attn package) 'not obviously different'?

Rule (his decision, 2026-10-16): if at EVERY sequence length the ratio of median latencies
sdpa_flash / flash_attn_pkg lies in [0.9, 1.1], STOP and discuss before the serving grid is run.
Only forward rows (forward_only=True) count, and only cells where both rows are ok; a missing or unavailable
flash-attn row means the gate cannot be evaluated (reported, never silently passed).

Usage: python gate.py [--csv ../results/final/attention.csv] [--lo 0.9] [--hi 1.1]
Exit code: 0 = gate NOT triggered (differences are visible, carry on), 3 = TRIGGERED (stop and discuss),
           4 = cannot be evaluated.

"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A, B = "sdpa_flash", "flash_attn_pkg"


def forward_latencies(csv_path: Path, dtype: str | None = "bfloat16") -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for r in csv.DictReader(open(csv_path)):
        if r["module"] != "attention" or r["experiment"] != "A1" or r["metric"] != "latency_ms":
            continue
        if r["forward_only"] != "True" or (dtype and r["dtype"] != dtype):
            continue
        out.setdefault(r["system"], {})[int(r["seq_len"])] = float(r["value"])
    return out


def evaluate(lat: dict[str, dict[int, float]], lo: float = 0.9, hi: float = 1.1) -> dict:
    a, b = lat.get(A, {}), lat.get(B, {})
    seqs = sorted(set(a) & set(b))
    if not seqs:
        return dict(verdict="cannot_evaluate", reason=f"no sequence length has both {A} and {B}", rows=[])
    rows = [dict(seq=s, a_ms=a[s], b_ms=b[s], ratio=a[s] / b[s]) for s in seqs]
    inside = all(lo <= r["ratio"] <= hi for r in rows)
    return dict(verdict="triggered" if inside else "not_triggered", rows=rows, lo=lo, hi=hi,
                missing=sorted(set(a) ^ set(b)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=HERE.parent / "results" / "final" / "attention.csv")
    ap.add_argument("--lo", type=float, default=0.9)
    ap.add_argument("--hi", type=float, default=1.1)
    args = ap.parse_args(argv)
    res = evaluate(forward_latencies(args.csv), args.lo, args.hi)
    if res["verdict"] == "cannot_evaluate":
        print(f"GATE CANNOT BE EVALUATED: {res['reason']}")
        return 4
    print(f"{A} / {B}  median-latency ratio per sequence length (forward, bf16); gate band [{args.lo}, {args.hi}]")
    print("  seq     sdpa_flash ms   flash_attn ms   ratio")
    for r in res["rows"]:
        flag = "" if args.lo <= r["ratio"] <= args.hi else "   <-- outside the band"
        print(f"  {r['seq']:6d}  {r['a_ms']:13.4f}  {r['b_ms']:14.4f}  {r['ratio']:6.3f}{flag}")
    if res["missing"]:
        print(f"  (lengths with only one of the two rows: {res['missing']})")
    if res["verdict"] == "triggered":
        print("GATE TRIGGERED: the two are within the band at every length -> STOP and discuss before the serving grid.")
        return 3
    print("GATE NOT TRIGGERED: at least one length differs by more than the band -> carry on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
