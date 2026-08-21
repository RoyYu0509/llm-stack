"""Analyse the replicated DDP-strategy benchmark.

Reports two things, because they answer different questions:

1. **Unpaired** per-wrapper means. This is what you quote as "X tok/s", and its spread shows
   the raw run-to-run variability of the machine (measured at CV ~5% here -- one 2-epoch
   run is ~78s, and TokenStreamDataset draws random windows through mmap, so page-cache
   state differs between runs).

2. **Paired within-rep** differences, which is the sensitive test. The runs were deliberately
   ordered rep-major (one of each wrapper, then the next rep) rather than wrapper-major, so
   the three wrappers inside a rep execute adjacent in time and share whatever drift the
   machine has at that moment. Differencing within a rep cancels that shared component, and
   the paired standard deviation is typically far smaller than the unpaired one. Had the
   runs been blocked by wrapper, this analysis would be invalid -- drift would be confounded
   with condition.

A paired t-test at n=5 is not much power, so the verdict language stays deliberately
cautious: it reports whether the observed difference clears 2 standard errors, and does not
dress a marginal result up as significance.
"""

import csv
import glob
import re
import statistics as st
import sys

RUNS = "artifacts/ddp_replicate/rep*/lm_matrix_results.csv"


def load() -> dict:
    """-> {wrapper: {rep: tok/s}}"""
    out = {}
    for f in glob.glob(RUNS):
        m = re.search(r"rep(\d+)_", f)
        if not m:
            continue
        rep = int(m.group(1))
        for r in csv.DictReader(open(f)):
            t = (r.get("tokens_per_sec") or "").strip()
            if t:
                out.setdefault(r["ddp"], {})[rep] = float(t)
    return out


def main() -> None:
    data = load()
    if not data:
        print("no replication results found"); sys.exit(1)

    print("=== unpaired: per-wrapper throughput across reps ===")
    print(f"{'wrapper':<28}{'n':>3}{'mean':>10}{'sd':>8}{'cv%':>7}{'min':>8}{'max':>8}")
    for w in sorted(data):
        v = list(data[w].values())
        mean = st.mean(v)
        sd = st.stdev(v) if len(v) > 1 else 0.0
        print(f"{w:<28}{len(v):>3}{mean:>10.0f}{sd:>8.0f}{100*sd/mean:>7.1f}{min(v):>8.0f}{max(v):>8.0f}")

    print("\n=== paired within-rep comparisons (the sensitive test) ===")
    names = sorted(data)
    for i in range(len(names)):
        for j in range(len(names)):
            if i >= j:
                continue
            a, b = names[i], names[j]
            reps = sorted(set(data[a]) & set(data[b]))
            if len(reps) < 2:
                print(f"{b} vs {a}: only {len(reps)} paired rep(s) -- cannot test")
                continue
            diffs = [100 * (data[b][r] - data[a][r]) / data[a][r] for r in reps]
            mean = st.mean(diffs)
            median = st.median(diffs)
            sd = st.stdev(diffs)
            se = sd / len(diffs) ** 0.5
            detail = ", ".join(f"r{r}:{d:+.1f}%" for r, d in zip(reps, diffs))
            if abs(mean) > 2 * se:
                t_verdict = f"mean {mean:+.1f}% clears 2SE"
            else:
                t_verdict = (f"NOT resolvable at n={len(diffs)} "
                             f"(would need |mean| > {2*se:.1f}%)")

            # Sign test, reported alongside the t-test because the per-run magnitudes here
            # are wildly heavy-tailed -- one pair differed by 43% while its neighbours
            # differed by 1-7%. A single outlier like that inflates sd enough to hide a
            # real effect from a mean/sd test. "Is B consistently faster than A?" is a
            # different and much more robust question, and the sign is what answers it.
            pos = sum(1 for d in diffs if d > 0)
            n = len(diffs)

            def _comb(n, k):
                num = den = 1
                for i in range(k):
                    num *= n - i
                    den *= i + 1
                return num // den

            tail = sum(_comb(n, k) for k in range(min(pos, n - pos) + 1))
            p = min(1.0, 2 * tail / (2 ** n))

            print(f"\n{b} vs {a}")
            print(f"  per-rep: {detail}")
            print(f"  mean {mean:+.2f}%   median {median:+.2f}%   sd {sd:.2f}   2SE {2*se:.2f}%")
            print(f"  t-test: {t_verdict}")
            winner = b if pos > n - pos else a
            wins = max(pos, n - pos)
            print(f"  sign  : {b} faster in {pos}/{n} reps  (exact two-sided p={p:.3f})")
            print(f"          -> " + (f"{winner} is consistently faster ({wins}/{n})"
                                      if p < 0.10 else
                                      f"direction NOT established at n={n}"
                                      f"{f' (leaning {winner}, {wins}/{n})' if wins > n - wins else ''}"))

    # Whether the pairing actually helped is a measurable claim, so measure it rather than
    # asserting it. (An earlier version of this script just declared that it worked.)
    unpaired_cvs, paired_sds = [], []
    for w in data:
        v = list(data[w].values())
        if len(v) > 1:
            unpaired_cvs.append(100 * st.stdev(v) / st.mean(v))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            reps = sorted(set(data[a]) & set(data[b]))
            if len(reps) > 1:
                paired_sds.append(st.stdev([100 * (data[b][r] - data[a][r]) / data[a][r]
                                            for r in reps]))
    if unpaired_cvs and paired_sds:
        u, p = st.mean(unpaired_cvs), st.mean(paired_sds)
        n_pairs = max(len(sorted(set(data[a]) & set(data[b]))) for a in data for b in data if a != b)
        print(f"\nPairing check: mean unpaired CV {u:.1f}% vs mean paired sd {p:.1f}%.")
        if n_pairs < 4:
            print(f"  With only {n_pairs} paired reps both estimates are themselves very "
                  f"noisy -- this comparison is not yet meaningful either way.")
        elif p < 0.6 * u:
            print("  Paired spread is materially smaller: the rep-major interleaving is "
                  "cancelling shared drift, as intended.")
        else:
            print("  Paired spread is NOT materially smaller than unpaired. The variance is "
                  "then mostly within-run (each measurement is only ~78s), not shared drift "
                  "-- longer measurements per run would help more than more reps.")


if __name__ == "__main__":
    main()
