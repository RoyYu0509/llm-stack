"""Collapse everything the overnight window produced into one readable markdown report.

The window writes a lot: per-arm logs, eval JSONs per checkpoint per domain, benchmark
CSVs, a serving benchmark. Reading that back means grepping several thousand lines of
tqdm output. This gathers the numbers that answer the questions the night was run to
answer, and says plainly when a piece is missing rather than omitting it silently.

    uv run python cs336_systems/experiments/make_night_report.py --out artifacts/NIGHT_REPORT.md
"""

import argparse
import glob
import json
import os
import re

DOMAINS = ["general", "web", "wiki", "instruction"]


def _load_evals() -> dict:
    """Model scores only. `seedcheck_*` and `slice_*` files are diagnostics (the same model
    re-scored under a different seed, or on a sliced stream) -- listing them as rows in the
    model table invites reading them as separate models."""
    out = {}
    for f in sorted(glob.glob("artifacts/eval/*.json")):
        name = os.path.basename(f)[:-5]
        if name.startswith(("seedcheck_", "slice_")):
            continue
        try:
            d = json.load(open(f))
            out[name] = {k: v["loss"] for k, v in d.get("streams", {}).items()}
        except Exception as e:  # a truncated file should not sink the whole report
            out[name] = {"_error": str(e)}
    return out


def _pretrain_val_curve() -> list[tuple[int, float]]:
    """Recover the val curve from the training log rather than W&B, so the report works
    without network access."""
    pts = []
    for path in ("full_pretrain_run.log",):
        if not os.path.exists(path):
            continue
        text = open(path, errors="ignore").read().replace("\r", "\n")
        for m in re.finditer(r"Rank 0 \| step (\d+) \| Val loss: ([\d.]+)", text):
            pts.append((int(m.group(1)), float(m.group(2))))
    return sorted(set(pts))


def _sft_val_curves() -> dict:
    curves = {}
    for path in sorted(glob.glob("sft_*.log")):
        name = os.path.basename(path)[4:-4]
        text = open(path, errors="ignore").read().replace("\r", "\n")
        pts = [(int(a), float(b)) for a, b in
               re.findall(r"Rank 0 \| step (\d+) \| Val loss: ([\d.]+)", text)]
        diverged = "training DIVERGED" in text
        curves[name] = {"points": sorted(set(pts)), "diverged": diverged}
    return curves


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="artifacts/NIGHT_REPORT.md")
    args = p.parse_args()

    L = ["# Overnight run report", ""]

    # ---- pretraining ----
    curve = _pretrain_val_curve()
    L += ["## 1. Pretraining", ""]
    if curve:
        L.append(_table(["step", "val loss"], [(s, f"{v:.4f}") for s, v in curve]))
        L += ["", f"In-training validation loss: **{curve[-1][1]:.4f}** at step {curve[-1][0]}.", "",
              "> **Two caveats on this number — do not quote it as \"validation loss on the "
              "mixed corpus\".**",
              "> 1. The training loop's val sampler has `shuffle=False` and `_run_eval` takes "
              "the *first* `val_bat_num` batches, while `valid.txt` is written domain-ordered "
              "(web → wiki → instruction). So every eval scored roughly the first ~1,800 "
              "tokens of the **web** section — the same documents each time. That makes the "
              "curve a low-variance *paired* progress signal (the decrease is real), but not "
              "a mixed-corpus number.",
              "> 2. `mixed_valid` itself has a contaminated instruction region: "
              "`build_mixed_corpus.build_instruction` cycles the held-out Alpaca pool 1.91× "
              "to hit the 10% ratio, and repeated text is artificially easy.",
              ">",
              "> The clean figures are the per-domain streams in section 3 "
              "(web / wiki / instruction, none of them oversampled).", ""]
    else:
        L += ["_No validation points found in full_pretrain_run.log._", ""]

    # ---- fine-tuning arms ----
    L += ["## 2. Instruction fine-tuning arms", ""]
    curves = _sft_val_curves()
    if curves:
        rows = []
        for name, c in sorted(curves.items()):
            pts = c["points"]
            best = min((v for _, v in pts), default=float("nan"))
            last = pts[-1][1] if pts else float("nan")
            note = "DIVERGED" if c["diverged"] else ("overfit?" if pts and last > best + 1e-3 else "")
            rows.append((name, len(pts), f"{best:.4f}", f"{last:.4f}", note))
        L.append(_table(["arm", "evals", "best val", "final val", "note"], rows))
        L += ["", "_'overfit?' means the final eval is worse than the best one — expected on "
              "a 4.31M-token corpus seen ~3.4 times._", ""]
    else:
        L += ["_No fine-tuning logs found._", ""]

    # ---- per-domain scoring ----
    L += ["## 3. Per-domain loss (paired: identical batches, fixed seed)", ""]
    evals = _load_evals()
    if evals:
        rows = []
        for name in sorted(evals):
            s = evals[name]
            rows.append([name] + [f"{s[d]:.4f}" if d in s else "—" for d in DOMAINS])
        L.append(_table(["model"] + DOMAINS, rows))
        base = evals.get("pretrained")
        if base and len(evals) > 1:
            L += ["", "### Change vs the pretrained model", "",
                  "_Negative = better. This is the forgetting trade-off: instruction should "
                  "improve, general/web/wiki should degrade, and the replay arms should "
                  "degrade less._", ""]
            rows = []
            for name in sorted(evals):
                if name == "pretrained":
                    continue
                s = evals[name]
                rows.append([name] + [f"{s[d]-base[d]:+.4f}" if d in s and d in base else "—"
                                      for d in DOMAINS])
            L.append(_table(["model"] + DOMAINS, rows))
        L.append("")
    else:
        L += ["_No eval JSONs found._", ""]

    # ---- serving ----
    L += ["## 4. Serving", ""]
    for label, path in [("recompute only", "artifacts/serving_benchmark.json"),
                        ("recompute vs KV cache", "artifacts/serving_benchmark_kv.json")]:
        if not os.path.exists(path):
            L += [f"_{label}: not produced ({path} missing)._", ""]
            continue
        d = json.load(open(path))
        L += [f"### {label}", ""]
        for mode, entry in d.get("modes", {}).items():
            rows = [(r["prompt_tokens"], f"{r['ttft_s']*1e3:.1f}",
                     f"{r['decode_p50_s']*1e3:.2f}", f"{r['decode_tokens_per_s']:.1f}")
                    for r in entry.get("runs", [])]
            if rows:
                L += [f"**{mode}**", "",
                      _table(["prompt tokens", "TTFT ms", "decode p50 ms", "tok/s"], rows), ""]
        if "cache_speedup_by_prompt_len" in d:
            sp = d["cache_speedup_by_prompt_len"]
            ordered = [sp[k] for k in sorted(sp, key=int)]
            rows = [(k, f"{sp[k]:.2f}x") for k in sorted(sp, key=int)]
            L += ["**KV cache speedup by prompt length**", "",
                  _table(["prompt tokens", "speedup"], rows), ""]
            # Read the numbers, don't assert a trend. The first real run measured a flat
            # ~1.2x across a 32x prompt-length range, contradicting the "grows with length"
            # story this section used to print unconditionally.
            trend = ordered[-1] / ordered[0] if ordered and ordered[0] else float("nan")
            mean = sum(ordered) / len(ordered) if ordered else float("nan")
            if trend > 1.25:
                L += [f"_Speedup grows with prompt length ({ordered[0]:.2f}x → "
                      f"{ordered[-1]:.2f}x): the prefix-dependent recompute is a material "
                      f"share of per-step cost at these lengths._", ""]
            else:
                L += [f"_Speedup is roughly flat at ~{mean:.2f}x and does **not** grow with "
                      f"prompt length. At this model size per-step cost is dominated by the "
                      f"fixed forward pass over the weights, so the prefix-dependent "
                      f"attention work a cache eliminates is only a small share of it. "
                      f"A KV cache pays off at longer contexts and larger batch, not here._", ""]

    # ---- benchmarks ----
    L += ["## 5. Systems benchmarks", ""]
    found = sorted(glob.glob("artifacts/bench_*/**/*.csv", recursive=True) +
                   glob.glob("artifacts/bench_*/**/*.md", recursive=True))
    if found:
        L += ["Artifacts written:", ""] + [f"- `{f}`" for f in found] + [""]
    else:
        L += ["_No benchmark artifacts found._", ""]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[report] wrote {args.out}")


if __name__ == "__main__":
    main()
