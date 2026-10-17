"""Schema guards. Pure CPU, no GPU needed."""
import csv

import pytest

from schema import COLUMNS, ResultWriter


def _row(**over):
    base = dict(
        module="ddp", experiment="T1", system="torch_ddp", metric="tokens_per_s",
        value=1.0, unit="tok/s", n_reps=12,
    )
    base.update(over)
    return base


def test_header_written_once_and_rows_roundtrip(tmp_path):
    p = tmp_path / "r.csv"
    w = ResultWriter(p)
    w.write(**_row(value=100.0))
    w.write(**_row(value=200.0))
    rows = list(csv.DictReader(p.open()))
    assert [r["value"] for r in rows] == ["100.0", "200.0"]
    assert list(rows[0].keys()) == COLUMNS, "CSV header must be exactly schema.COLUMNS"


def test_missing_required_field_is_an_error(tmp_path):
    w = ResultWriter(tmp_path / "r.csv")
    with pytest.raises(ValueError, match="missing required"):
        w.write(**_row(unit=""))


def test_unknown_column_is_an_error(tmp_path):
    w = ResultWriter(tmp_path / "r.csv")
    with pytest.raises(ValueError, match="unknown columns"):
        w.write(**_row(my_new_knob=3))


def test_attention_rows_must_state_forward_only(tmp_path):
    """A latency number that silently mixes fwd and fwd+bwd does not survive an interviewer
    opening the CSV, so the schema refuses to record one."""
    w = ResultWriter(tmp_path / "r.csv")
    with pytest.raises(ValueError, match="forward_only"):
        w.write(**_row(module="attention", experiment="A1"))
    w.write(**_row(module="attention", experiment="A1", forward_only=True))
    w.write(**_row(module="attention", experiment="A1", forward_only=False))


def test_environment_columns_are_filled_automatically(tmp_path):
    w = ResultWriter(tmp_path / "r.csv")
    full = w.write(**_row())
    assert full["timestamp"], "timestamp should be captured automatically"
    assert "torch" in full and "gpu" in full and "git_sha" in full


def test_git_state_can_be_injected_for_boxes_without_dot_git(tmp_path, monkeypatch):
    """A rented box has no .git; without an override the row would claim 'clean, sha unknown'. The launcher exports
    the LOCAL tree's state, and that is what must land in the CSV."""
    monkeypatch.setenv("BENCH_GIT_SHA", "abc1234567")
    monkeypatch.setenv("BENCH_GIT_DIRTY", "1")
    p = tmp_path / "r.csv"
    ResultWriter(p, repo_dir=tmp_path).write(**_row())
    row = next(csv.DictReader(p.open()))
    assert row["git_sha"] == "abc1234567" and row["git_dirty"] == "True"
    monkeypatch.setenv("BENCH_GIT_DIRTY", "0")
    p2 = tmp_path / "r2.csv"
    ResultWriter(p2, repo_dir=tmp_path).write(**_row())
    assert next(csv.DictReader(p2.open()))["git_dirty"] == "False"
