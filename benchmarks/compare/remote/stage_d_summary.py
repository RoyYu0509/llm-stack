"""One-page status of stage D: which cells of the agreed grid (D-7 section 7.13-F) are in serving_stage_d.csv and which are missing.

Meant to be read first when coming back to a run nobody watched:
    python stage_d_summary.py [--csv .../results/final/serving_stage_d.csv]

Prints one line per expected cell (OK / MISSING, throughput, TTFT p50, TPOT p50, number of failed requests), then a verdict.
Exit code 1 if any cell is missing, so a script can branch on it.

"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CS = (1, 8, 32, 64)
RATES = (10, 50, 75)
LADDER = {"seq_nokv": (1,), "seq_kv": (1, 32), "cb_naive": CS, "cb_fa2": CS, "cb_fa2_prefill": CS, "cb_fa2_prefill_compile": CS}
BEST = "engine_cb_fa2_prefill_compile_bfloat16"
SGLANG = ("sglang_cb_only_flashinfer_bfloat16", "sglang_default_noradix_flashinfer_bfloat16_gb64")
VLLM = ("vllm_nograph_noprefix_bfloat16", "vllm_default_noprefix_bfloat16")


def expected():
    """[(group, experiment, system, concurrency, rate or None)]"""
    out = []
    for arm, cs in LADDER.items():
        out += [("ladder", "I1", f"engine_{arm}_bfloat16", c, None) for c in cs]
    for s in SGLANG:
        out += [("sglang", "I1", s, c, None) for c in CS]
    for s in VLLM:
        out += [("vllm", "I1", s, c, None) for c in CS]
    for s in (BEST,) + SGLANG:
        out += [("openloop", "I2", s, 0, r) for r in RATES]
    return out


def load(path: Path) -> dict:
    """{(experiment, system, concurrency, rate): {metric: value}}"""
    cells: dict = {}
    if not path.exists():
        return cells
    for r in csv.DictReader(open(path)):
        rate = None
        if r["experiment"] == "I2" and "rate=" in r["variant"]:
            rate = float(r["variant"].split("rate=")[1].split(";")[0])
        key = (r["experiment"], r["system"], int(r["concurrency"] or 0), rate)
        cells.setdefault(key, {})[r["metric"]] = float(r["value"])
    return cells


def summarize(path: Path) -> tuple[list[str], int]:
    cells = load(path)
    lines, missing, group = [], 0, None
    for g, exp, system, c, rate in expected():
        if g != group:
            lines.append(f"\n[{g}]")
            group = g
        m = next((v for (e, s, cc, rr), v in cells.items() if e == exp and s == system and cc == c and
                  (rate is None or (rr is not None and abs(rr - rate) < 1e-9))), None)
        label = f"{system:48s} " + (f"rate={rate:g}/s" if rate is not None else f"C={c}")
        if m is None or "output_tok_per_s" not in m:
            lines.append(f"  MISSING  {label}")
            missing += 1
            continue
        f = lambda k, fmt: format(m[k], fmt) if k in m else "-"
        failed = int(m.get("n_failed_requests", 0))
        lines.append(f"  OK       {label:70s} {f('output_tok_per_s', '8.0f')} tok/s  TTFT p50 {f('ttft_ms_p50', '7.1f')} ms  "
                     f"TPOT p50 {f('tpot_ms_p50', '6.1f')} ms" + (f"   FAILED REQUESTS: {failed}" if failed else ""))
    total = len(expected())
    lines.append(f"\n{total - missing}/{total} cells present" + ("" if not missing else f"; {missing} MISSING"))
    return lines, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=HERE.parent / "results" / "final" / "serving_stage_d.csv")
    a = ap.parse_args(argv)
    lines, missing = summarize(a.csv)
    print(f"stage D: {a.csv}")
    print("\n".join(lines))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
