import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import summarize as sm  # noqa: E402

COLS = ["module", "experiment", "system", "metric", "value", "std", "n_reps", "dtype", "forward_only", "seq_len", "batch"]


def _csv(tmp_path, rows):
    p = tmp_path / "a.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for system, seq, metric, value, std, fwd in rows:
            w.writerow(dict(module="attention", experiment="A1", system=system, metric=metric, value=value, std=std,
                            n_reps=5, dtype="bfloat16", forward_only=fwd, seq_len=seq, batch=32768 // seq))
    return p


def test_ratio_oom_and_noise_marks(tmp_path, capsys):
    p = _csv(tmp_path, [("sdpa_flash", 1024, "latency_ms", 1.0, 0.01, True),
                        ("triton_fa_mine", 1024, "latency_ms", 2.5, 0.01, True),
                        ("attention", 1024, "oom", 1, "", True),
                        ("flash_attn_pkg", 1024, "latency_ms", 0.5, 0.2, True)])           # std/mean 40% -> noisy
    assert sm.main(["--csv", str(p)]) == 0
    out = capsys.readouterr().out
    lat = [ln for ln in out.splitlines() if ln.startswith("| 1024 (32)")]
    assert "0.500~" in lat[0], "a noisy cell must carry the '~' mark"
    ratio = [ln for ln in out.splitlines() if ln.startswith("| 1024 (32)")][1]
    assert "2.50x" in ratio and "OOM" in ratio and "1.00x" in ratio


def test_only_the_asked_dtype_is_read(tmp_path):
    p = tmp_path / "a.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerow(dict(module="attention", experiment="A1", system="sdpa_flash", metric="latency_ms", value=1, std=0,
                        n_reps=5, dtype="float16", forward_only=True, seq_len=128, batch=256))
    assert sm.load(p, "bfloat16") == {}


def test_rows_without_a_sequence_length_are_skipped(tmp_path):
    """Regression: the inductor check writes A1 rows (fuse_attention_count) with an empty seq_len; they crashed int()."""
    p = tmp_path / "a.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerow(dict(module="attention", experiment="A1", system="torch_compile_attention", metric="fuse_attention_count",
                        value=0, std="", n_reps=1, dtype="bfloat16", forward_only=True, seq_len="", batch=""))
        w.writerow(dict(module="attention", experiment="A1", system="sdpa_flash", metric="latency_ms", value=1.0, std=0.0,
                        n_reps=5, dtype="bfloat16", forward_only=True, seq_len=128, batch=256))
    cells = sm.load(p)
    assert list(cells) == [(True, 128)] and list(cells[(True, 128)]) == ["sdpa_flash"]
