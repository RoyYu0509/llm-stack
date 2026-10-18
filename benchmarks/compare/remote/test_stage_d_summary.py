"""stage_d_summary.py is what is read first after an unwatched run: it must list exactly the cells run_stage_d.sh produces,
find them in the csv by the same keys the writers use, and say MISSING loudly."""
import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path

import stage_d_summary as sm

HERE = Path(__file__).parent
SCRIPT = HERE / "run_stage_d.sh"


def dry(phase):
    root = Path(tempfile.mkdtemp()) / "llm-stack"
    e = {**os.environ, "DRY_RUN": "1", "ROOT": str(root)}
    out = subprocess.run(["bash", str(SCRIPT), phase], capture_output=True, text=True, env=e, check=True).stderr
    return [ln[2:] for ln in out.splitlines() if ln.startswith("+ ")]


def write_csv(path, cells, drop=None, failed=None):
    cols = ["experiment", "system", "variant", "metric", "value", "concurrency"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for g, exp, system, c, rate in cells:
            if (system, c, rate) == drop:
                continue
            variant = "new_tokens=mixed(mean=64);" + (f"open_loop;rate={rate:g};cv=1;duration=30s;cap=64" if rate else "closed_loop")
            for metric, v in (("output_tok_per_s", 1000.0), ("ttft_ms_p50", 40.0), ("tpot_ms_p50", 10.0)):
                w.writerow(dict(experiment=exp, system=system, variant=variant, metric=metric, value=v, concurrency=c))
            if failed == (system, c, rate):
                w.writerow(dict(experiment=exp, system=system, variant=variant, metric="n_failed_requests", value=3, concurrency=c))


def test_the_expected_cells_are_exactly_the_agreed_grid():
    exp = sm.expected()
    assert len(exp) == 19 + 16 + 9, "19 ladder + (2 sglang + 2 vllm) x 4 closed loop + 3 systems x 3 rates"
    assert len({e[1:] for e in exp}) == len(exp), "no duplicate cells"


def test_the_ladder_cells_match_what_run_stage_d_actually_runs():
    ran = set()
    for c in dry("ladder"):
        if "record_demo.py" in c:
            m = re.search(r"--arm-name (\w+?)_c(\d+) ", c)
            ran.add((f"engine_{m.group(1)}_bfloat16", int(m.group(2))))
    assert ran == {(system, c) for g, _, system, c, _ in sm.expected() if g == "ladder"}


def test_a_complete_csv_has_nothing_missing_and_a_hole_is_reported(tmp_path):
    p = tmp_path / "s.csv"
    write_csv(p, sm.expected())
    lines, missing = sm.summarize(p)
    assert missing == 0 and "44/44 cells present" in lines[-1]
    write_csv(p, sm.expected(), drop=("vllm_nograph_noprefix_bfloat16", 32, None))
    lines, missing = sm.summarize(p)
    assert missing == 1 and any("MISSING" in ln and "vllm_nograph_noprefix_bfloat16" in ln and "C=32" in ln for ln in lines)


def test_open_loop_cells_are_found_by_rate_and_failures_are_flagged(tmp_path):
    p = tmp_path / "s.csv"
    write_csv(p, sm.expected(), drop=(sm.BEST, 0, 50), failed=(sm.SGLANG[0], 0, 75))
    lines, missing = sm.summarize(p)
    assert missing == 1 and any("MISSING" in ln and "rate=50/s" in ln for ln in lines)
    assert any("FAILED REQUESTS: 3" in ln and "rate=75/s" in ln for ln in lines)


def test_a_missing_file_is_all_missing_and_exit_code_is_nonzero(tmp_path):
    assert sm.main(["--csv", str(tmp_path / "nope.csv")]) == 1
