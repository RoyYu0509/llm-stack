"""run_stage_c.sh is shell, so its completeness is tested through DRY_RUN: every rung x concurrency cell must appear exactly once,
with exactly the cumulative flags of its rung (an arm that silently lacks a flag would be reported as a rung it is not)."""
import os
import re
import subprocess
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).parent / "run_stage_c.sh"
CS = (1, 8, 32, 64)


def dry(phase="all", **env):
    root = Path(tempfile.mkdtemp()) / "llm-stack"
    e = {**os.environ, "DRY_RUN": "1", "ROOT": str(root), **env}
    out = subprocess.run(["bash", str(SCRIPT), phase], capture_output=True, text=True, env=e, check=True).stderr
    return [ln[2:].replace(str(root), "<ROOT>") for ln in out.splitlines() if ln.startswith("+ ")]


def arm_of(cmd):
    return re.search(r"--arm-name (\w+?)_c(\d+)", cmd).group(1), int(re.search(r"--concurrency (\d+)", cmd).group(1))


def test_ladder_has_every_rung_and_concurrency_once_with_the_cumulative_flags():
    cmds = [c for c in dry("ladder") if "record_demo.py" in c]
    assert len(cmds) == 16
    assert sorted((arm_of(c)[0], arm_of(c)[1]) for c in cmds) == sorted(
        (a, c) for a in ("fa2_base", "fa2_lastlogits", "fa2_prefillbatch", "fa2_compiled") for c in CS)
    want = {
        "fa2_base": (set(), {"--prefill-last-logits", "--prefill-batch-tokens", "--compile-decode"}),
        "fa2_lastlogits": ({"--prefill-last-logits"}, {"--prefill-batch-tokens", "--compile-decode"}),
        "fa2_prefillbatch": ({"--prefill-last-logits", "--prefill-batch-tokens 16384"}, {"--compile-decode"}),
        "fa2_compiled": ({"--prefill-last-logits", "--prefill-batch-tokens 16384", "--compile-decode"}, set()),
    }
    for c in cmds:
        arm, _ = arm_of(c)
        must, mustnt = want[arm]
        assert all(m in c for m in must), f"{arm}: missing {[m for m in must if m not in c]}"
        assert not any(m in c for m in mustnt), f"{arm}: has a switch that belongs to a later rung"
        assert "--attn fa2" in c and "--dtype bfloat16" in c and "--reps 3" in c and "workload_mixed4096.json" in c
    n = {arm_of(c)[1]: int(re.search(r"--n-requests (\d+)", c).group(1)) for c in cmds}
    w = {arm_of(c)[1]: int(re.search(r"--warmup-requests (\d+)", c).group(1)) for c in cmds}
    assert n == {1: 32, 8: 64, 32: 256, 64: 512} and w == {1: 8, 8: 16, 32: 64, 64: 128}, "same request / warm-up rule as stage B"
    assert len([c for c in dry("ladder") if "closed_loop.py ingest" in c]) == 16


def test_prefill_budget_is_a_knob_and_defaults_to_the_big_value():
    assert all("--prefill-batch-tokens 16384" in c for c in dry("ladder") if "fa2_compiled_c" in c and "record_demo.py" in c)
    assert all("--prefill-batch-tokens 4096" in c for c in dry("ladder", PBT="4096") if "fa2_prefillbatch_c" in c and "record_demo.py" in c)


def test_check_phase_has_the_gate_and_both_decode_step_profiles():
    cmds = dry("check")
    assert sum("check_stage_c.py" in c and "--checkpoint" in c and "--dtype bfloat16" in c for c in cmds) == 1
    prof = [c for c in cmds if "profile_decode_step.py" in c]
    assert len(prof) == 2 and sum("--compile-decode" in c for c in prof) == 1 and all("--attn fa2" in c for c in prof)


def test_drift_phase_reruns_the_stage_b_naive_arm_at_c32():
    cmds = [c for c in dry("drift") if "record_demo.py" in c]
    assert len(cmds) == 1 and "--attn naive" in cmds[0] and "--concurrency 32" in cmds[0] and "--compile-decode" not in cmds[0]


def test_everything_writes_under_results_final_and_nothing_touches_the_instance():
    for c in dry("all"):
        for m in re.findall(r"(?:--out|-o) (\S+)", c):
            assert "results/final" in m, f"output outside results/final: {m}"
        assert not re.search(r"vastai|rent|destroy|nvidia-smi -pl", c), f"a stage script must never start / stop / rent: {c}"


def test_smoke_covers_every_rung_and_both_checks_at_toy_size_and_writes_nothing_to_results():
    cmds = dry("smoke")
    rec = [c for c in cmds if "record_demo.py" in c]
    assert sorted(re.search(r"--arm-name (\w+)", c).group(1) for c in rec) == [
        "smoke_fa2_base", "smoke_fa2_compiled", "smoke_fa2_lastlogits", "smoke_fa2_prefillbatch", "smoke_naive"]
    assert sum("--compile-decode" in c for c in rec) == 1 and sum("--prefill-batch-tokens" in c for c in rec) == 2
    assert sum("closed_loop.py ingest" in c for c in cmds) == 4
    assert sum("check_stage_c.py" in c for c in cmds) == 1
    prof = [c for c in cmds if "profile_decode_step.py" in c]
    assert len(prof) == 2 and sum("--compile-decode" in c for c in prof) == 1
    for c in cmds:
        for m in re.findall(r"(?:--out|-o) (\S+)", c):
            assert "results/final" not in m, f"the smoke test must not write into results/final: {m}"


def test_sglang_phase_runs_the_two_variants_on_this_machine_and_never_writes_to_the_ladder_csv():
    cmds = dry("sglang")
    runs = [c for c in cmds if "run_sglang.sh" in c]
    assert len(runs) == 2
    assert any("ATTN=flashinfer" in c and " cb_only" in c and "MEM_FRACTION=0.8" in c for c in runs)
    assert any("ATTN=flashinfer" in c and "default_noradix" in c and "CUDA_GRAPH_MAX_BS=64" in c for c in runs)
    for c in runs:
        assert "CONC=1 8 32 64" in c and "DTYPE=bfloat16" in c and "N_MIN=32" in c and "N_MULT=8" in c and "REPS=3" in c
        assert "SERVING_CSV=<ROOT>/benchmarks/compare/results/final/serving_sglang_samemachine.csv" in c, "separate CSV: stage B has rows with the same system names"
        assert "SAME MACHINE" in c and "450 W" in c
