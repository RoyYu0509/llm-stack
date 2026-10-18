import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import gate  # noqa: E402

COLS = ["module", "experiment", "system", "metric", "value", "forward_only", "dtype", "seq_len"]


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for system, seq, ms, fwd in rows:
            w.writerow(dict(module="attention", experiment="A1", system=system, metric="latency_ms", value=ms,
                            forward_only=fwd, dtype="bfloat16", seq_len=seq))


def test_triggers_when_every_length_is_inside_the_band(tmp_path):
    p = tmp_path / "a.csv"
    _write(p, [("sdpa_flash", 1024, 1.00, True), ("flash_attn_pkg", 1024, 1.05, True),
               ("sdpa_flash", 8192, 6.4, True), ("flash_attn_pkg", 8192, 6.1, True)])
    assert gate.main(["--csv", str(p)]) == 3


def test_not_triggered_if_any_length_is_outside_the_band(tmp_path):
    p = tmp_path / "a.csv"
    _write(p, [("sdpa_flash", 1024, 1.00, True), ("flash_attn_pkg", 1024, 1.02, True),
               ("sdpa_flash", 8192, 6.4, True), ("flash_attn_pkg", 8192, 4.9, True)])   # ratio 1.31
    assert gate.main(["--csv", str(p)]) == 0


def test_band_edges_are_inclusive_and_ratio_direction_is_sdpa_over_pkg():
    lat = {"sdpa_flash": {128: 1.1, 256: 0.9}, "flash_attn_pkg": {128: 1.0, 256: 1.0}}
    assert gate.evaluate(lat)["verdict"] == "triggered"
    lat = {"sdpa_flash": {128: 1.0}, "flash_attn_pkg": {128: 0.8}}      # sdpa 1.25x SLOWER than the package
    r = gate.evaluate(lat)
    assert r["verdict"] == "not_triggered" and r["rows"][0]["ratio"] == pytest.approx(1.25)


def test_missing_package_row_is_reported_not_silently_passed(tmp_path):
    p = tmp_path / "a.csv"
    _write(p, [("sdpa_flash", 1024, 1.0, True)])
    assert gate.main(["--csv", str(p)]) == 4


def test_only_forward_rows_count(tmp_path):
    p = tmp_path / "a.csv"
    _write(p, [("sdpa_flash", 1024, 1.0, True), ("flash_attn_pkg", 1024, 1.0, True),
               ("sdpa_flash", 1024, 5.0, False), ("flash_attn_pkg", 1024, 1.0, False)])    # fwd+bwd would differ a lot
    assert gate.main(["--csv", str(p)]) == 3
