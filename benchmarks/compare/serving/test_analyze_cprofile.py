import cProfile
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze_cprofile as ac  # noqa: E402


def _busy(s):
    t = time.perf_counter()
    while time.perf_counter() - t < s:
        pass


def test_categories_and_sync_points_on_a_synthetic_profile(tmp_path):
    import torch

    x = torch.ones(4)

    def work():
        _busy(0.05)
        for _ in range(100):
            x.sum().item()       # a builtin 'item' call: the kind of blocking point the report must surface

    pr = cProfile.Profile()
    pr.runcall(work)
    p = tmp_path / "a.prof"
    pr.dump_stats(str(p))
    r = ac.analyze(str(p))
    assert r["total_s"] >= 0.05
    assert any("item" in s["name"] and s["calls"] == 100 for s in r["sync"]), r["sync"]
    assert "C builtins / torch C++ ops" in r["by_category"]
    assert abs(sum(r["by_category"].values()) - r["total_s"]) < 1e-6, "every function must land in exactly one category"
    assert "GPU sync points" in ac.render(r)


def test_categorize_paths():
    assert ac.categorize("/w/serving/inferenceLM/engine/inference_engine.py", "f").startswith("engine scheduling")
    assert ac.categorize("/w/training/cs336-basics/cs336_basics/lm.py", "f").startswith("model Python")
    assert ac.categorize("/usr/lib/python3.12/asyncio/base_events.py", "f") == "asyncio event loop"
    assert ac.categorize("~", "<built-in method x>") == "C builtins / torch C++ ops"
