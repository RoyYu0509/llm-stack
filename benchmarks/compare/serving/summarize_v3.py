"""Tables for the v3 campaign (engine recordings + serving_v3.csv). Everything is recomputed from raw per-token
timestamps; nothing is taken from a summary field except peak memory. Refuses duplicate rows.

  results/engine_scan_v3/<arm>_c<C>_n<N>_q<requests>[_r<rep>].json
  results/serving_v3.csv
"""
from __future__ import annotations

import collections
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results"


def pct(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))] if xs else float("nan")


def engine_cfgs():
    cfgs = collections.defaultdict(list)
    for f in sorted((RES / "engine_scan_v3").glob("*.json")):
        if f.name.startswith("diag_"):
            continue
        d = json.loads(f.read_text())
        m = d["meta"]
        key = (m["arm"].split("_c")[0], m["concurrency"],
               "all" if m.get("max_prefills_per_round") is None else m["max_prefills_per_round"], m["n_requests"])
        if m.get("status") != "ok":
            cfgs[key].append(dict(status=m.get("status"), phase=m.get("phase"), peak=m.get("peak_mem_gb")))
            continue
        C = m["concurrency"]
        reqs = d["requests"]
        first = [(int(q["id"][1:]), q["tokens"][0][0] - q["submit_t"]) for q in reqs]
        steady = [t for i, t in first if i >= C]            # requests that arrived after the first wave
        allf = [t for _, t in first]
        itl = [b - a for q in reqs for a, b in zip([t for t, _ in q["tokens"]], [t for t, _ in q["tokens"]][1:])]
        cfgs[key].append(dict(status="ok", thr=m["throughput_tok_s"], t50=pct(allf, .5), t90=pct(allf, .9),
                              s50=pct(steady, .5), s90=pct(steady, .9), s99=pct(steady, .99),
                              i50=pct(itl, .5), i99=pct(itl, .99), imax=max(itl), peak=m.get("peak_mem_gb"),
                              steps=(m.get("kv_stats") or {}).get("steps")))
    return cfgs


def main() -> None:
    cfgs = engine_cfgs()
    print("## Engine (bf16, closed loop, mixed output lengths 32–96, random-init weights)\n")
    print("| arm | C | N | requests | reps | tok/s (mean ± std) | TTFT p50 / p90 s (all) | TTFT p50 / p90 / p99 s (after 1st wave) "
          "| ITL p50 / p99 / max ms | peak GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    order = lambda k: (k[1], k[0], 10**6 if k[2] == "all" else k[2])
    best = {}
    for k in sorted(cfgs, key=order):
        arm, C, N, q = k
        ok = [r for r in cfgs[k] if r["status"] == "ok"]
        if not ok:
            r0 = cfgs[k][0]
            print(f"| {arm} | {C} | {N} | {q} | 0 | **{r0['status']}** ({r0['phase'] or 'timed'}, peak {r0['peak']:.1f} GiB) | | | | |")
            continue
        m = lambda key: st.mean(r[key] for r in ok)
        thr = [r["thr"] for r in ok]
        best[k] = st.mean(thr)
        print(f"| {arm} | {C} | {N} | {q} | {len(ok)} | {st.mean(thr):.0f} ± {st.pstdev(thr):.1f} | "
              f"{m('t50'):.2f} / {m('t90'):.2f} | {m('s50'):.2f} / {m('s90'):.2f} / {m('s99'):.2f} | "
              f"{m('i50') * 1e3:.1f} / {m('i99') * 1e3:.0f} / {m('imax') * 1e3:.0f} | {max(r['peak'] or 0 for r in ok):.1f} |")

    print("\n## HF generate / SGLang (serving_v3.csv)\n")
    rows = list(csv.DictReader((RES / "serving_v3.csv").open())) if (RES / "serving_v3.csv").exists() else []
    seen = collections.Counter((r["system"], r["concurrency"], r["metric"]) for r in rows)
    dups = [k for k, v in seen.items() if v > 1]
    if dups:
        raise SystemExit(f"duplicate rows in serving_v3.csv (same system / concurrency / metric): {dups[:5]}")
    t = collections.defaultdict(dict)
    for r in rows:
        if r["system"].startswith("engine_cb"):
            continue
        t[(r["system"], int(r["concurrency"]))][r["metric"]] = (float(r["value"]), r["std"], r["n_reps"], r["notes"])
    print("| system | C | requests | tok/s | ± | reps | TTFT p50 / p99 ms | TPOT p50 ms | failed |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (s, c) in sorted(t):
        m = t[(s, c)]
        o = m.get("output_tok_per_s", (float("nan"), "", "", ""))
        nreq = re.search(r"n_requests=(\d+)", o[3])
        ser = " (serial)" if ("cb_off" in s or s.startswith("hf_generate")) else ""
        print(f"| {s}{ser} | {c} | {nreq.group(1) if nreq else ''} | {o[0]:.0f} | {o[1]} | {o[2]} | "
              f"{m.get('ttft_ms_p50', (float('nan'),))[0]:.0f} / {m.get('ttft_ms_p99', (float('nan'),))[0]:.0f} | "
              f"{m.get('tpot_ms_p50', (float('nan'),))[0]:.2f} | {int(m.get('n_failed_requests', (0,))[0])} |")

    print("\n## Same-C ratios engine / SGLang (throughput)\n")
    sgl = {(s, c): v["output_tok_per_s"][0] for (s, c), v in t.items() if "output_tok_per_s" in v}
    print("| C | engine fa2 N=16 | engine fa2 N=all | SGLang cb_only (flashinfer) | ratio N=16 | SGLang default (flashinfer) | ratio N=16 |")
    print("|---|---|---|---|---|---|---|")
    for C in (32, 64, 128, 256, 384, 512):
        e16 = next((v for k, v in best.items() if k[0] == "fa2" and k[1] == C and k[2] == 16), None)
        eall = next((v for k, v in best.items() if k[0] == "fa2" and k[1] == C and k[2] == "all"), None)
        cb = next((v for (s, c), v in sgl.items() if c == C and s.startswith("sglang_cb_only_flashinfer")), None)
        df = next((v for (s, c), v in sgl.items() if c == C and s.startswith("sglang_default_noradix_flashinfer")), None)
        f = lambda x: f"{x:.0f}" if x else "-"
        r = lambda a, b: f"{a / b:.2f}x" if a and b else "-"
        print(f"| {C} | {f(e16)} | {f(eall)} | {f(cb)} | {r(e16, cb)} | {f(df)} | {r(e16, df)} |")


if __name__ == "__main__":
    main()
