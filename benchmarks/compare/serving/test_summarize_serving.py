import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import summarize_serving as ss  # noqa: E402

COLS = ["module", "experiment", "system", "metric", "value", "std", "n_reps", "concurrency"]


def _csv(tmp_path, rows):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for system, c, metric, value, std in rows:
            w.writerow(dict(module="serving", experiment="I1", system=system, metric=metric, value=value, std=std, n_reps=3, concurrency=c))
    return p


def test_groups_ratios_and_group_means(tmp_path, capsys):
    rows = []
    for system, tp in (("engine_cb_naive_bfloat16", 1000), ("hf_generate_eager_bfloat16", 100),
                       ("sglang_cb_only_flashinfer_bfloat16", 4000), ("engine_cb_fa2_bfloat16", 1500)):
        rows += [(system, 32, "output_tok_per_s", tp, tp * 0.01), (system, 32, "ttft_ms_p50", 20, 0)]
    p = _csv(tmp_path, rows)
    assert ss.main(["--csv", str(p)]) == 0
    out = capsys.readouterr().out
    lines = {ln.split("|")[2].strip(): ln for ln in out.splitlines() if ln.startswith("| ") and ln.count("|") > 6}
    assert "4.00x" in lines["SGLang cb_only (flashinfer)"], "ratio is relative to the baseline engine (naive)"
    assert "1.00x" in lines["baseline engine, naive attention"]
    assert "naive (materialises scores): mean of 2" in out and "550" in out        # (1000 + 100) / 2
    assert "| opt |" in lines["engine fa2 (prealloc KV + FA2 decode + batched argmax)"], "arms must be labelled"


def test_noise_mark_and_unknown_systems(tmp_path, capsys):
    p = _csv(tmp_path, [("engine_cb_naive_bfloat16", 8, "output_tok_per_s", 100, 30), ("something_new", 8, "output_tok_per_s", 1, 0)])
    ss.main(["--csv", str(p)])
    out = capsys.readouterr().out
    assert "100~" in out and "something_new" in out
