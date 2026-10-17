"""ladder_table.py: the steady-state TTFT must leave the opening burst out, and the ratios must be against the right rung."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ladder_table as lt  # noqa: E402


def _csv(path: Path, rows):
    cols = ["module", "experiment", "system", "variant", "metric", "value", "unit", "std", "n_reps", "concurrency"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, c, metric, value, std in rows:
            w.writerow(dict(module="serving", experiment="I1", system=f"engine_cb_{arm}_bfloat16", variant="v", metric=metric,
                            value=value, unit="", std=std, n_reps=3, concurrency=c))


def _raw(d: Path, arm: str, c: int, ttfts_ms):
    """First `c` requests sent at t=0 (the burst), the rest later; ttfts_ms lists the TTFT of each, burst first."""
    reqs = []
    for i, ms in enumerate(ttfts_ms):
        sent = 0.0 if i < c else 1.0 + i
        reqs.append(dict(id=f"r{i}", submit_t=sent, tokens=[[sent + ms / 1e3, 1]]))
    (d / f"{arm}_c{c}_r1.json").write_text(json.dumps(dict(meta=dict(status="ok"), requests=reqs)))


def test_steady_ttft_excludes_the_opening_burst(tmp_path):
    _raw(tmp_path, "fa2_base", 2, [900, 900, 100, 110, 120, 130, 140])          # burst 900 ms, steady ~120 ms
    p50, p99 = lt.steady_ttft_ms(tmp_path, "fa2_base", 2)
    assert 100 <= p50 <= 140 and p99 <= 140, (p50, p99)
    assert lt.steady_ttft_ms(tmp_path, "fa2_compiled", 2) is None


def test_table_ratios_are_against_the_previous_rung_and_against_rung_zero(tmp_path):
    rows = []
    for arm, tps in (("fa2_base", 100.0), ("fa2_lastlogits", 110.0), ("fa2_prefillbatch", 220.0), ("fa2_compiled", 330.0)):
        rows += [(arm, 32, "output_tok_per_s", tps, 1.0), (arm, 32, "ttft_ms_p50", 50, 0), (arm, 32, "ttft_ms_p99", 80, 0),
                 (arm, 32, "tpot_ms_p50", 20, 0), (arm, 32, "itl_ms_p99", 40, 0)]
    _csv(tmp_path / "s.csv", rows)
    md = lt.table(tmp_path / "s.csv", tmp_path, cs=(32,))
    line = {r.split("|")[2].strip(): r for r in md.splitlines() if r.startswith("| 32")}
    assert "1.10x" in line["1 + last-position logits"] and "1.10x" in line["1 + last-position logits"]
    assert "2.00x" in line["2 + batched prefill, 1 group/round"] and "2.20x" in line["2 + batched prefill, 1 group/round"]
    assert "1.50x" in line["3 + torch.compile decode"] and "3.30x" in line["3 + torch.compile decode"]
    assert "x" not in line["0 fa2 (stage-B arm, re-run)"].split("|")[5], "rung 0 has nothing to be compared with"


def test_noisy_cells_are_marked(tmp_path):
    _csv(tmp_path / "s.csv", [("fa2_base", 1, "output_tok_per_s", 100.0, 30.0)])
    assert "100~" in lt.table(tmp_path / "s.csv", tmp_path, cs=(1,))
