"""results/final/manifest.json -- what produced the numbers next to it.

Every benchmark entry point calls `record(stage, ...)` once at the start. The file is a list; each entry holds the
command line, git sha and dirtiness, host, interpreter, the versions of the packages the numbers depend on, and the
GPU. Nothing here is typed by hand, so the CSVs and the manifest cannot disagree about the environment.
(`env_check.py` is a separate, deliberate step on a rented box; its log goes to logs/ and is referenced by stage.)

"""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGES = ["torch", "triton", "flash-attn", "transformers", "sglang", "regex", "numpy", "tokenizers"]


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def package_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


def gpu_info() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip().replace("\n", " | ") if r.returncode == 0 else ""
    except Exception:
        return ""


def record(stage: str, out_dir: str | os.PathLike | None = None, argv: list[str] | None = None,
           extra: dict | None = None) -> Path:
    path = Path(out_dir or (HERE / "results" / "final")) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(path.read_text()) if path.exists() and path.stat().st_size else []
    entries.append(dict(
        stage=stage,
        timestamp=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        argv=argv if argv is not None else sys.argv,
        git_sha=os.environ.get("BENCH_GIT_SHA") or _git("rev-parse", "--short=10", "HEAD"),
        git_dirty=(os.environ["BENCH_GIT_DIRTY"] == "1") if "BENCH_GIT_DIRTY" in os.environ
        else bool(_git("status", "--porcelain")),
        host=socket.gethostname(),
        platform=platform.platform(),
        python=platform.python_version(),
        packages=package_versions(),
        gpu=gpu_info(),
        extra=extra or {},
    ))
    path.write_text(json.dumps(entries, indent=1))
    return path
