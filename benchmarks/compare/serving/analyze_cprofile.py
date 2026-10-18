"""Where does the engine's Python / event-loop time go?  Reads a cProfile dump of one engine run (record_demo.py).

cProfile sees Python functions and the C builtins they call, not what happens on the GPU. What it does show: how much
wall time is Python (the asyncio loop, the engine's scheduling code, the model's Python layer) versus time inside torch
calls, and how much is spent BLOCKED at GPU sync points -- `Tensor.item()`, `.tolist()`, `.cpu()`, `synchronize` -- because a
blocking call is charged the time the GPU needs to finish what it was asked to do. That last number is "the CPU is waiting for
the GPU", not "Python is slow". cProfile also slows Python code, so this run is never part of the throughput tables.

Usage: python analyze_cprofile.py engine.prof [--top 15]
"""
from __future__ import annotations

import argparse
import pstats
from collections import defaultdict

CATEGORIES = [  # first match wins
    ("engine scheduling (inferenceLM/engine)", ("inferenceLM/engine",)),
    ("other serving code (inferenceLM/)", ("inferenceLM/",)),
    ("model Python layer (cs336_basics)", ("cs336_basics",)),
    ("asyncio event loop", ("/asyncio/", "asyncio\\")),
    ("torch Python layer", ("site-packages/torch", "/torch/")),
]
SYNC_NAMES = ("'item'", "'tolist'", "'cpu'", "'numpy'", "synchronize", "'__bool__'", "'__int__'", "'__float__'", "_cuda_synchronize")


def categorize(filename: str, funcname: str) -> str:
    if filename == "~":                      # C builtins and methods
        return "C builtins / torch C++ ops"
    for name, keys in CATEGORIES:
        if any(k in filename for k in keys):
            return name
    return "other Python (stdlib, third party)"


def analyze(path: str, top: int = 15) -> dict:
    st = pstats.Stats(path)
    total = sum(v[2] for v in st.stats.values())
    by_cat: dict[str, float] = defaultdict(float)
    sync = []
    rows = []
    for (fn, line, name), (cc, nc, tt, ct, _) in st.stats.items():
        by_cat[categorize(fn, name)] += tt
        rows.append((tt, nc, f"{fn.split('/')[-1] if fn != '~' else '~'}:{line}({name})"))
        if fn == "~" and any(s in name for s in SYNC_NAMES):
            sync.append((tt, nc, name))
    rows.sort(reverse=True)
    sync.sort(reverse=True)
    return dict(total_s=total, by_category=dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
                top=[dict(tottime_s=t, calls=n, where=w) for t, n, w in rows[:top]],
                sync=[dict(tottime_s=t, calls=n, name=w) for t, n, w in sync],
                sync_total_s=sum(t for t, _, _ in sync))


def render(r: dict) -> str:
    out = [f"total profiled time (sum of tottime): {r['total_s']:.2f} s", "", "by category (tottime):"]
    for k, v in r["by_category"].items():
        out.append(f"  {k:46s} {v:8.2f} s  {100 * v / r['total_s']:5.1f}%")
    out += ["", f"GPU sync points (blocking calls; time here = the CPU waiting for the GPU): {r['sync_total_s']:.2f} s "
                f"({100 * r['sync_total_s'] / r['total_s']:.1f}% of profiled time)"]
    for s in r["sync"][:8]:
        out.append(f"  {s['name']:60s} {s['calls']:8d} calls  {s['tottime_s']:8.2f} s")
    out += ["", "top functions by own time:"]
    for t in r["top"]:
        out.append(f"  {t['tottime_s']:8.2f} s  {t['calls']:9d} calls  {t['where']}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)
    print(render(analyze(args.profile, args.top)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
