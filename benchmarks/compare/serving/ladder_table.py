"""The stage-C ladder as one table: every rung x concurrency, with the gain over the PREVIOUS rung and over rung 0.

Reads results/final/serving.csv (system `engine_cb_<arm>_bfloat16`, written by closed_loop.py ingest) for throughput / TPOT /
ITL / TTFT, and the raw recordings (serving_raw/engine/<arm>_c<C>_r*.json) for the one thing the CSV cannot give: the
STEADY-STATE TTFT, i.e. without the opening burst (the first C requests are all sent at t=0 and, in this engine, all see the
first token at the end of the same round; counting them says more about the start of the run than about serving).

    python ladder_table.py [--csv ../results/final/serving.csv] [--raw ../results/final/serving_raw/engine]

"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

ARMS = ["fa2_base", "fa2_lastlogits", "fa2_prefillbatch", "fa2_compiled"]
LABEL = {"fa2_base": "0 fa2 (stage-B arm, re-run)", "fa2_lastlogits": "1 + last-position logits",
         "fa2_prefillbatch": "2 + batched prefill, 1 group/round", "fa2_compiled": "3 + torch.compile decode"}
HERE = Path(__file__).resolve().parent


def load_csv(path: Path) -> dict:
    """{(arm, C): {metric: (value, std)}} for the ladder arms."""
    out: dict = {}
    for r in csv.DictReader(open(path)):
        s = r["system"]
        if not (s.startswith("engine_cb_") and s.endswith("_bfloat16")):
            continue
        arm = s[len("engine_cb_"):-len("_bfloat16")]
        if arm not in ARMS and arm != "naive_drift":
            continue
        out.setdefault((arm, int(r["concurrency"])), {})[r["metric"]] = (float(r["value"]), float(r["std"] or 0))
    return out


def steady_ttft_ms(raw_dir: Path, arm: str, c: int):
    """(p50, p99) of the TTFT of requests sent after t=0, median over the repetitions; None if no recordings."""
    p50s, p99s = [], []
    for f in sorted(glob.glob(str(raw_dir / f"{arm}_c{c}_r*.json"))):
        j = json.load(open(f))
        if j["meta"].get("status") != "ok":
            continue
        t = sorted(1e3 * (r["tokens"][0][0] - r["submit_t"]) for r in j["requests"] if r["submit_t"] > 1e-3)
        if t:
            p50s.append(statistics.median(t))
            p99s.append(t[min(len(t) - 1, int(round(0.99 * (len(t) - 1))))])
    return (statistics.median(p50s), statistics.median(p99s)) if p50s else None


def table(csv_path: Path, raw_dir: Path, cs=(1, 8, 32, 64)) -> str:
    d = load_csv(csv_path)
    lines = ["| C | rung | tok/s | vs prev | vs rung 0 | TTFT p50 | TTFT p99 | steady TTFT p50 / p99 | TPOT p50 | ITL p99 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cs:
        base = d.get(("fa2_base", c), {}).get("output_tok_per_s", (None,))[0]
        prev = None
        for arm in ARMS:
            m = d.get((arm, c))
            if not m:
                lines.append(f"| {c} | {LABEL[arm]} | missing | | | | | | | |")
                continue
            tps = m["output_tok_per_s"][0]
            noisy = "~" if m["output_tok_per_s"][0] and m["output_tok_per_s"][1] / m["output_tok_per_s"][0] > 0.10 else ""
            st = steady_ttft_ms(raw_dir, arm, c)
            g = lambda k: f"{m[k][0]:.1f}" if k in m else "-"
            lines.append(
                f"| {c} | {LABEL[arm]} | {tps:.0f}{noisy} | {'' if prev is None else f'{tps / prev:.2f}x'} | "
                f"{'' if base is None or arm == 'fa2_base' else f'{tps / base:.2f}x'} | {g('ttft_ms_p50')} | {g('ttft_ms_p99')} | "
                f"{'-' if st is None else f'{st[0]:.0f} / {st[1]:.0f}'} | {g('tpot_ms_p50')} | {g('itl_ms_p99')} |")
            prev = tps
    if any(k[0] == "naive_drift" for k in d):
        m = d[("naive_drift", 32)]
        lines.append(f"\nDrift check (stage-B naive arm, C=32, re-run in this session): {m['output_tok_per_s'][0]:.0f} tok/s, "
                     f"TPOT p50 {m['tpot_ms_p50'][0]:.1f} ms (stage B on the other machine: 828 tok/s, 33.2 ms).")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE.parent / "results" / "final" / "serving.csv"))
    ap.add_argument("--raw", default=str(HERE.parent / "results" / "final" / "serving_raw" / "engine"))
    a = ap.parse_args()
    print(table(Path(a.csv), Path(a.raw)))


if __name__ == "__main__":
    main()
