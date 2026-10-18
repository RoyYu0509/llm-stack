"""Shared result schema for benchmarks/compare.

Every script in this directory writes its measurements through `ResultWriter`, so that
all rows share one set of columns and carry the metadata a reader needs to judge whether
two numbers are comparable (dtype / causal / forward_only / GPU / library versions / git sha).

Design rules (see README.md, "Fairness traps"):
  * one row = one measurement of one metric (long format);
  * `forward_only` must be stated explicitly on attention rows -- a missing value is an error,
    because a latency number that silently mixes fwd and fwd+bwd is exactly the kind of
    claim that does not survive an interviewer opening the CSV;
  * environment metadata is captured automatically, never typed by hand.

"""
from __future__ import annotations

import csv
import datetime as _dt
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

COLUMNS = [
    # identity
    "module",        # attention | ddp | serving | tokenizer | convert
    "experiment",    # handoff id: A1 / T1 / I1-b / K1 / C3 ...
    "system",        # e.g. triton_fa_mine | sdpa_flash | torch_ddp | sglang_cb_only
    "variant",       # free text for one-off knobs (bucket_mb=25, backend=triton, ...)
    "metric",        # latency_ms | tokens_per_s | peak_mem_mb | ttft_ms_p50 | max_abs_diff ...
    "value",
    "unit",
    "std",           # std over `n_reps` repetitions, blank if n_reps == 1
    "n_reps",
    "warmup",
    "iters",
    # workload
    "dtype",
    "causal",
    "forward_only",  # True: forward kernel only. False: forward + backward measured together.
    "seq_len",
    "batch",
    "concurrency",
    "n_gpus",
    # environment (filled automatically)
    "gpu",
    "torch",
    "triton",
    "cuda",
    "git_sha",
    "git_dirty",
    "timestamp",
    "notes",
]

REQUIRED = ["module", "experiment", "system", "metric", "value", "unit", "n_reps"]


def _git(*args: str, cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def env_metadata(repo_dir: str | os.PathLike | None = None) -> dict[str, Any]:
    """Collect the environment fields. Missing libraries are recorded as '' (not guessed)."""
    repo = Path(repo_dir or Path(__file__).resolve().parent)
    # A rented box has no .git (311 MB): the launcher exports the LOCAL tree's state instead (BENCH_GIT_SHA / BENCH_GIT_DIRTY=0|1),
    # and that value is the one that goes into every row. Never fall back to "clean" just because git is missing.
    sha_env, dirty_env = os.environ.get("BENCH_GIT_SHA"), os.environ.get("BENCH_GIT_DIRTY")
    meta: dict[str, Any] = {
        "git_sha": sha_env or _git("rev-parse", "--short=10", "HEAD", cwd=repo),
        "git_dirty": (dirty_env == "1") if dirty_env is not None else bool(_git("status", "--porcelain", cwd=repo)),
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "torch": "",
        "triton": "",
        "cuda": "",
        "gpu": "",
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda or ""
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
        else:
            meta["gpu"] = f"cpu:{platform.processor() or platform.machine()}"
    except Exception:
        pass
    try:
        import triton

        meta["triton"] = triton.__version__
    except Exception:
        pass
    return meta


class ResultWriter:
    """Append rows to a CSV with the shared header. Header is written once."""

    def __init__(self, path: str | os.PathLike, repo_dir: str | os.PathLike | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._env = env_metadata(repo_dir)

    def validate(self, row: Mapping[str, Any]) -> None:
        missing = [k for k in REQUIRED if row.get(k) in (None, "")]
        if missing:
            raise ValueError(f"result row is missing required fields: {missing}")
        unknown = [k for k in row if k not in COLUMNS]
        if unknown:
            raise ValueError(f"unknown columns {unknown}; add them to schema.COLUMNS first")
        if row["module"] == "attention" and not isinstance(row.get("forward_only"), bool):
            raise ValueError(
                "attention rows must set forward_only=True/False explicitly "
                "(do not leave the fwd / fwd+bwd distinction implicit)"
            )

    def write(self, **row: Any) -> dict[str, Any]:
        full = {**self._env, **row}
        self.validate(full)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                w.writeheader()
            w.writerow({c: full.get(c, "") for c in COLUMNS})
        return full
