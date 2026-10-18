"""T1 -- turn `benchmark_lm_matrix.py`'s lm_matrix_results.csv into schema rows and compute
weak-scaling efficiency.

Definition (written down once so that "86%" cannot mean two things):

    efficiency(wrapper) = tokens_per_s(wrapper, N GPUs) / (N * tokens_per_s(Local No DDP, 1 GPU))

with the SAME per-GPU batch on both sides (weak scaling). `benchmark_lm_matrix.py` already
defines an epoch as a fixed number of samples, so 1-GPU and 2-GPU epochs cover the same data.

Usage:
  python ingest_lm_matrix.py ../../../training/artifacts/video1/ddp_flash_ctx1024_b8/lm_matrix_results.csv \
      --ctx 1024 --out ../results/ddp.csv --source "video1 ddp_flash (n=12 timed epochs)"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from schema import ResultWriter  # noqa: E402

REPO = HERE.parents[2]
BASELINE = "Local No DDP"


def _step_ms(r: dict) -> float:
    """Wall time of one optimizer step in ms = sec_per_epoch / steps_per_epoch."""
    return 1e3 * float(r["sec_per_epoch"]) / float(r["steps_per_epoch"])


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def efficiencies(rows: list[dict]) -> list[dict]:
    """Add `efficiency` to every multi-GPU row. Rows without a 1-GPU baseline of the same
    kernel (and local batch size) get efficiency=None rather than a guess."""
    base, base_step = {}, {}
    for r in rows:
        if r["ddp"] == BASELINE:
            key = (r["kernel"], r["local_batch_size"])
            base[key] = float(r["tokens_per_sec"])
            base_step[key] = _step_ms(r)
    out = []
    for r in rows:
        key = (r["kernel"], r["local_batch_size"])
        b = base.get(key)
        tps = float(r["tokens_per_sec"])
        n = int(r["gpus"])
        eff = None if (b is None or r["ddp"] == BASELINE) else tps / (n * b)
        step = _step_ms(r)
        # Same per-GPU batch => the compute per step is the same, so the extra wall time per step
        # of the N-GPU run over the 1-GPU run is communication / sync that was NOT hidden.
        over = None if (key not in base_step or r["ddp"] == BASELINE) else step - base_step[key]
        out.append({**r, "tokens_per_sec": tps, "gpus": n, "efficiency": eff,
                    "step_ms": step, "exposed_ms": over})
    return out


def ingest(csv_path: Path, out_path: Path, ctx: int | None, source: str, bucket_mb: str | None = None):
    w = ResultWriter(out_path, REPO)
    rows = efficiencies(load_rows(csv_path))
    for r in rows:
        bmb = r.get("bucket_size_mb") or bucket_mb or ""
        common = dict(
            module="ddp", experiment="T1", system=r["ddp"], n_gpus=r["gpus"],
            batch=int(r["local_batch_size"]), seq_len=ctx, dtype="float32",
            variant=f"kernel={r['kernel']};bucket_mb={bmb};per_gpu_batch={r['local_batch_size']}",
            n_reps=int(r["epochs"]),
        )
        w.write(**common, metric="tokens_per_s", value=r["tokens_per_sec"], unit="tok/s",
                notes=f"source: {source}; {csv_path.name}")
        w.write(**common, metric="peak_gpu_mem_mb", value=float(r["peak_gpu_mb"]), unit="MiB")
        w.write(**common, metric="step_time_ms", value=round(r["step_ms"], 2), unit="ms")
        if r["exposed_ms"] is not None:
            w.write(**common, metric="exposed_overhead_ms_per_step", value=round(r["exposed_ms"], 2),
                    unit="ms", notes="step_N - step_1 at equal per-GPU batch: un-hidden comm / sync")
        if r["efficiency"] is not None:
            w.write(**common, metric="scaling_efficiency_pct", value=round(100 * r["efficiency"], 2),
                    unit="%", notes="weak scaling vs 1-GPU same per-GPU batch")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=HERE.parent / "results" / "ddp.csv")
    ap.add_argument("--ctx", type=int, default=None, help="context length used (not stored in the CSV)")
    ap.add_argument("--bucket-mb", default=None, help="bucket size if the CSV has no bucket_size_mb column")
    ap.add_argument("--source", default="", help="free text: which run / commit this CSV came from")
    args = ap.parse_args(argv)
    rows = ingest(args.csv, args.out, args.ctx, args.source, args.bucket_mb)
    for r in rows:
        eff = "" if r["efficiency"] is None else f"{100 * r['efficiency']:.1f}%"
        ex = "" if r["exposed_ms"] is None else f"  +{r['exposed_ms']:.0f} ms/step"
        print(f"{r['kernel']:<26}{r['ddp']:<28}{r['gpus']}x  {r['tokens_per_sec']:>9.1f} tok/s  "
              f"{r['step_ms']:7.1f} ms/step  {eff}{ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
