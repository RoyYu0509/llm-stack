"""T1 ingest tests. The real-data test uses the committed video1 CSV and is skipped if absent."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ingest_lm_matrix as ing  # noqa: E402

VIDEO1 = (Path(__file__).resolve().parents[3] / "training" / "artifacts" / "video1"
          / "ddp_flash_ctx1024_b8" / "lm_matrix_results.csv")

HEADER = ["kernel", "ddp", "gpus", "epochs", "steps_per_epoch", "global_batch_size",
          "local_batch_size", "samples_per_epoch", "bucket_size_mb", "wall_sec",
          "sec_per_epoch", "tokens_per_sec", "peak_gpu_mb"]


def _write(p, rows):
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADER})


def _r(ddp, gpus, tps, kernel="k", lb=8, steps=1, sec=1.0):
    return dict(kernel=kernel, ddp=ddp, gpus=gpus, epochs=12, steps_per_epoch=steps,
                global_batch_size=lb * gpus, local_batch_size=lb, samples_per_epoch=256,
                wall_sec=1, sec_per_epoch=sec, tokens_per_sec=tps, peak_gpu_mb=100)


def test_weak_scaling_efficiency_definition(tmp_path):
    src = tmp_path / "m.csv"
    _write(src, [_r("Local No DDP", 1, 10000), _r("Pytorch DDP", 2, 17000),
                 _r("Bucketed Overlapping DDP", 2, 16000)])
    rows = {r["ddp"]: r for r in ing.efficiencies(ing.load_rows(src))}
    assert rows["Local No DDP"]["efficiency"] is None
    assert rows["Pytorch DDP"]["efficiency"] == pytest.approx(0.85), (
        "efficiency must be tps_N / (N * tps_1): 17000 / (2 * 10000)"
    )
    assert rows["Bucketed Overlapping DDP"]["efficiency"] == pytest.approx(0.80)


def test_missing_baseline_gives_none_not_a_guess(tmp_path):
    src = tmp_path / "m.csv"
    _write(src, [_r("Pytorch DDP", 2, 17000, kernel="orphan")])
    (row,) = ing.efficiencies(ing.load_rows(src))
    assert row["efficiency"] is None


def test_baseline_is_matched_per_kernel_and_batch(tmp_path):
    src = tmp_path / "m.csv"
    _write(src, [_r("Local No DDP", 1, 10000, kernel="a", lb=8),
                 _r("Local No DDP", 1, 5000, kernel="b", lb=8),
                 _r("Pytorch DDP", 2, 18000, kernel="a", lb=8),
                 _r("Pytorch DDP", 2, 9000, kernel="b", lb=8)])
    eff = {r["kernel"]: r["efficiency"] for r in ing.efficiencies(ing.load_rows(src))
           if r["ddp"] == "Pytorch DDP"}
    assert eff["a"] == pytest.approx(0.9) and eff["b"] == pytest.approx(0.9), (
        "each kernel must be divided by its OWN 1-GPU baseline"
    )


@pytest.mark.skipif(not VIDEO1.exists(), reason="video1 artifacts not on this machine")
def test_video1_numbers_reproduce(tmp_path):
    ing.ingest(VIDEO1, tmp_path / "ddp.csv", 1024, "video1 ddp_flash")
    rows = {r["ddp"]: r for r in ing.efficiencies(ing.load_rows(VIDEO1))}
    assert rows["Bucketed Overlapping DDP"]["efficiency"] == pytest.approx(0.823, abs=0.002)
    assert rows["Naive DDP"]["efficiency"] == pytest.approx(0.771, abs=0.002)


def test_exposed_overhead_is_step_time_difference_at_equal_per_gpu_batch(tmp_path):
    src = tmp_path / "m.csv"
    # 1 GPU: 32 steps/epoch in 28.0 s -> 875 ms/step ; 2 GPUs: 16 steps/epoch in 17.0 s -> 1062.5 ms/step
    _write(src, [_r("Local No DDP", 1, 9000, steps=32, sec=28.0),
                 _r("Bucketed Overlapping DDP", 2, 15000, steps=16, sec=17.0)])
    rows = {r["ddp"]: r for r in ing.efficiencies(ing.load_rows(src))}
    assert rows["Local No DDP"]["exposed_ms"] is None
    assert rows["Local No DDP"]["step_ms"] == pytest.approx(875.0)
    assert rows["Bucketed Overlapping DDP"]["exposed_ms"] == pytest.approx(187.5), (
        "exposed overhead per step = step_ms(N GPUs) - step_ms(1 GPU); both use the same per-GPU batch"
    )


@pytest.mark.skipif(not VIDEO1.exists(), reason="video1 artifacts not on this machine")
def test_video1_exposed_overhead_matches_hand_computation():
    raw = {r["ddp"]: r for r in ing.load_rows(VIDEO1)}
    step1 = 1e3 * float(raw["Local No DDP"]["sec_per_epoch"]) / float(raw["Local No DDP"]["steps_per_epoch"])
    b = raw["Bucketed Overlapping DDP"]
    step2 = 1e3 * float(b["sec_per_epoch"]) / float(b["steps_per_epoch"])
    got = {r["ddp"]: r for r in ing.efficiencies(ing.load_rows(VIDEO1))}["Bucketed Overlapping DDP"]
    assert got["exposed_ms"] == pytest.approx(step2 - step1)
    assert 100 < got["exposed_ms"] < 300, got["exposed_ms"]   # sanity: the order of magnitude we expect
