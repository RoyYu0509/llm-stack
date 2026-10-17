"""run_stage_d.sh is shell, so its completeness is tested through DRY_RUN: every cell of the agreed grid (D-7 section 7.13-F) must
appear exactly once with exactly its own switches (an arm that silently carries a later rung's flag would be reported as a
rung it is not), and every flag it passes must exist in the real parser of the script it is passed to."""
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SCRIPT = HERE / "run_stage_d.sh"
REPO = HERE.parents[2]
CS = (1, 8, 32, 64)
RUNGS = ["seq_nokv", "seq_kv", "cb_naive", "cb_fa2", "cb_fa2_prefill", "cb_fa2_prefill_compile"]


def dry(phase="all", **env):
    root = Path(tempfile.mkdtemp()) / "llm-stack"
    e = {**os.environ, "DRY_RUN": "1", "ROOT": str(root), **env}
    out = subprocess.run(["bash", str(SCRIPT), phase], capture_output=True, text=True, env=e, check=True).stderr
    return [ln[2:].replace(str(root), "<ROOT>") for ln in out.splitlines() if ln.startswith("+ ")]


def rec_cmds(cmds):
    return [c for c in cmds if "record_demo.py" in c]


def arm_c(cmd):
    m = re.search(r"--arm-name (\w+?)_c(\d+) ", cmd)
    return m.group(1), int(m.group(2))


def num(cmd, flag):
    return int(re.search(rf"{flag} (\d+)", cmd).group(1))


FLAGS = {   # what each rung carries on top of --preset/--prompts-json/--device/--dtype, and nothing else
    "seq_nokv": {"--attn naive", "--mode seq", "--no-kv-cache"},
    "seq_kv": {"--attn naive", "--mode seq"},
    "cb_naive": {"--attn naive", "--mode cb"},
    "cb_fa2": {"--attn fa2", "--mode cb"},
    "cb_fa2_prefill": {"--attn fa2", "--mode cb", "--prefill-last-logits", "--prefill-batch-tokens 16384"},
    "cb_fa2_prefill_compile": {"--attn fa2", "--mode cb", "--prefill-last-logits", "--prefill-batch-tokens 16384", "--compile-decode"},
}
OPTIONAL = ["--no-kv-cache", "--prefill-last-logits", "--prefill-batch-tokens", "--compile-decode"]


def test_ladder_is_the_19_agreed_cells_each_with_exactly_its_own_switches():
    cmds = rec_cmds(dry("ladder"))
    assert len(cmds) == 19
    cells = sorted(arm_c(c) for c in cmds)
    want = sorted([("seq_nokv", 1), ("seq_kv", 1), ("seq_kv", 32)] + [(a, c) for a in RUNGS[2:] for c in CS])
    assert cells == want
    for c in cmds:
        arm, _ = arm_c(c)
        for f in FLAGS[arm]:
            assert f in c, f"{arm}: missing {f}"
        for f in OPTIONAL:
            if not any(g.startswith(f) for g in FLAGS[arm]):
                assert f not in c, f"{arm}: carries {f}, which belongs to a later rung"
        assert "--dtype bfloat16" in c and "--reps 3" in c and "workload_mixed4096_cont.json" in c and "--device cuda" in c
        assert "workload_mixed4096.json" not in c, "stage D uses the continuous-length workload"


def test_request_and_warmup_rules_and_the_two_documented_exceptions():
    for c in rec_cmds(dry("ladder")):
        arm, C = arm_c(c)
        n, w = num(c, "--n-requests"), num(c, "--warmup-requests")
        if (arm, C) == ("seq_kv", 32):
            assert n == 64, "serial C=32 uses 64 requests (256 would take ~15 min per repetition)"
        else:
            assert n == max(8 * C, 32), (arm, C, n)
        want_w = max(2 * C, 8)
        assert w == (min(want_w, 16) if arm.startswith("seq_") else want_w), (arm, C, w)
        assert num(c, "--concurrency") == C


def test_every_ladder_cell_is_ingested_into_the_one_stage_d_csv_under_its_own_system_name():
    cmds = dry("ladder")
    ing = [c for c in cmds if "closed_loop.py ingest" in c]
    assert len(ing) == 19
    assert sorted(set(re.search(r"--system (\S+)", c).group(1) for c in ing)) == sorted(f"engine_{a}_bfloat16" for a in RUNGS)
    assert all("--out <ROOT>/benchmarks/compare/results/final/serving_stage_d.csv" in c for c in ing)


def test_sglang_and_vllm_rows_are_the_agreed_ones_on_the_continuous_workload_and_hf_is_not_there():
    sg = [c for c in dry("sglang") if "run_sglang.sh" in c]
    assert len(sg) == 2
    assert any("cb_only" in c and "MEM_FRACTION=0.8" in c and "ATTN=flashinfer" in c for c in sg)
    assert any("default_noradix" in c and "CUDA_GRAPH_MAX_BS=64" in c for c in sg)
    vl = [c for c in dry("vllm") if "run_vllm.sh" in c]
    assert [c.rsplit(" ", 1)[1] for c in vl] == ["nograph_noprefix", "default_noprefix"]
    assert all("/venv-vllm/bin/python" in c for c in vl)
    for c in sg + vl:
        assert "workload_mixed4096_cont.json" in c and "DTYPE=bfloat16" in c and "REPS=3" in c
        assert "SERVING_CSV=<ROOT>/benchmarks/compare/results/final/serving_stage_d.csv" in c
        assert "CONC=1 8 32 64" in c and "N_MULT=8" in c and "N_MIN=32" in c
    for phase in ("all", "smoke"):
        assert not [c for c in dry(phase) if "HF_ATTN" in c or "hf_cb" in c or "run_sglang.sh hf" in c], \
            "HF is not in the throughput comparison; it is only the c5 correctness reference"


def test_open_loop_has_three_rates_for_the_engine_and_two_sglang_variants_all_capped_at_64():
    cmds = dry("openloop")
    eng = rec_cmds(cmds)
    assert sorted(num(c, "--arrival-rate") for c in eng) == [10, 50, 75]
    for c in eng:
        assert num(c, "--concurrency") == 64 and num(c, "--duration") == 30 and num(c, "--reps") == 3 and "--arrival-cv 1" in c
        for f in FLAGS["cb_fa2_prefill_compile"]:
            assert f in c, f"open loop must use the best configuration: missing {f}"
        assert "--n-requests" not in c, "the request count comes from rate x duration"
    ing = [c for c in cmds if "closed_loop.py ingest" in c]
    assert len(ing) == 3 and all("--open-loop" in c and "--slo-ttft-s 0.5" in c and "--slo-tpot-ms 50" in c for c in ing)
    sg = [c for c in cmds if "run_sglang.sh" in c]
    assert len(sg) == 2
    for c in sg:
        assert "ARRIVAL_RATES=10 50 75" in c and "MAX_INFLIGHT=64" in c and "DURATION=30" in c and "ARRIVAL_CV=1" in c
        assert "SLO_TTFT_S=0.5" in c and "SLO_TPOT_MS=50" in c and "CONC=" not in c
    assert not [c for c in cmds if "run_vllm.sh" in c or "hf_cb" in c], "vLLM and HF are not part of the open-loop experiment"


def test_correctness_is_the_best_configuration_in_bf16_against_hf_report_only():
    (c,) = [c for c in dry("c5") if "c5_greedy_equal.py" in c]
    for f in ("--dtype bfloat16", "--attn fa2", "--prefill-last-logits", "--prefill-batch-tokens 16384", "--compile-decode",
              "--n-prompts 64", "--modes cb", "--device cuda"):
        assert f in c, f
    assert "--out <ROOT>/benchmarks/compare/results/final/serving_correctness_stage_d.csv" in c


def test_all_runs_every_phase_once_in_the_agreed_order_and_never_touches_the_instance():
    cmds = dry("all")
    first = lambda s: next(i for i, c in enumerate(cmds) if s in c)
    order = [first("--arm-name seq_nokv_c1"), first("run_sglang.sh cb_only"), first("run_vllm.sh nograph_noprefix"),
             first("--arrival-rate 10"), first("c5_greedy_equal.py")]
    assert order == sorted(order), "ladder, sglang, vllm, openloop, c5"
    for c in cmds:
        assert not re.search(r"vastai|rent|destroy|nvidia-smi -pl", c), f"a stage script must never start / stop / rent: {c}"
        if "convert_190m_to_hf.py" in c:   # only when the random-init HF model dir is missing; it is a model, not a result
            continue
        for m in re.findall(r"(?:(?<!\S)(?:--out|-o) |(?<!\S)SERVING_CSV=)(\S+)", c):
            assert "results/final" in m, f"output outside results/final: {m}"


def test_smoke_covers_every_component_and_writes_nothing_to_results():
    cmds = dry("smoke")
    rec = rec_cmds(cmds)
    assert sorted(re.search(r"--arm-name (\w+)", c).group(1) for c in rec) == sorted([f"smoke_{a}" for a in RUNGS] + ["smoke_openloop"])
    assert sum("--no-kv-cache" in c for c in rec) == 1 and sum("--compile-decode" in c for c in rec) == 2   # best config + its open-loop twin
    assert sum("closed_loop.py ingest" in c for c in cmds) == 7 and sum("--open-loop" in c for c in cmds) == 1
    assert sum("c5_greedy_equal.py" in c for c in cmds) == 1
    assert sum("run_sglang.sh cb_only" in c and "ARRIVAL_RATES=5" in c for c in cmds) == 1
    assert sum("run_vllm.sh nograph_noprefix" in c for c in cmds) == 1
    for c in cmds:
        for m in re.findall(r"(?:(?<!\S)(?:--out|-o) |(?<!\S)SERVING_CSV=)(\S+)", c):
            assert "results/final" not in m, f"the smoke test must not write into results/final: {m}"


def _help(script, *sub):
    r = subprocess.run([sys.executable, str(script), *sub, "--help"], capture_output=True, text=True, cwd=REPO / "serving")
    assert r.returncode == 0, r.stderr[-300:]
    return r.stdout


@pytest.mark.parametrize("phase", ["ladder", "openloop", "c5", "smoke"])
def test_every_flag_passed_to_a_python_script_exists_in_that_scripts_real_parser(phase):
    helps = {"record_demo.py": _help(REPO / "serving" / "benchmarks" / "record_demo.py"),
             "c5_greedy_equal.py": _help(REPO / "benchmarks" / "compare" / "serving" / "c5_greedy_equal.py"),
             "ingest": _help(REPO / "benchmarks" / "compare" / "serving" / "closed_loop.py", "ingest")}
    for c in dry(phase):
        key = "ingest" if "closed_loop.py ingest" in c else next((k for k in helps if k in c), None)
        if key is None:
            continue
        tail = c.split(key if key != "ingest" else "ingest", 1)[1]
        for flag in set(re.findall(r"(?<!\S)(--[a-z0-9][a-z0-9-]*)", tail)):
            assert flag in helps[key], f"{key} has no flag {flag}: {c[:160]}"


def test_runner_scripts_get_only_env_vars_they_read():
    """run_sglang.sh / run_vllm.sh read their configuration from the environment: a misspelt name is silently ignored."""
    text = {n: (REPO / "benchmarks" / "compare" / "serving" / n).read_text() for n in ("run_sglang.sh", "run_vllm.sh")}
    for c in dry("all"):
        for script in text:
            if f"{script} " not in c + " ":
                continue
            for name in re.findall(r"(?<![A-Za-z_])([A-Z][A-Z0-9_]+)=", c.split("env ", 1)[1].split(f"bash ", 1)[0]):
                if name in ("PY",):
                    continue
                assert re.search(rf"\b{name}\b", text[script]), f"{script} does not read ${name}"


def _stub(path, body):
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _live(tmp_path, nvidia_mib):
    root = tmp_path / "llm-stack"
    (root / "serving").mkdir(parents=True)
    log = tmp_path / "py.log"
    py = _stub(tmp_path / "py", f'echo "$@" >> {log}\n')
    smi = _stub(tmp_path / "smi", f"echo {nvidia_mib}\n")
    env = {**os.environ, "ROOT": str(root), "PY": str(py), "NVSMI": str(smi), "ST": str(tmp_path / "status.txt"), "REPS": "1", "CS": "1",
           "IDLE_WAIT_S": "2", "IDLE_POLL_S": "1"}
    env.pop("DRY_RUN", None)
    r = subprocess.run(["bash", str(SCRIPT), "ladder"], capture_output=True, text=True, env=env, timeout=120)
    return r, log


def test_a_busy_gpu_stops_the_phase_before_anything_is_started(tmp_path):
    r, log = _live(tmp_path, 18000)
    assert r.returncode == 3 and "GPU is not idle" in r.stderr
    assert not log.exists(), "nothing may be launched while another job holds the GPU"


def test_an_idle_gpu_lets_the_phase_start(tmp_path):
    r, log = _live(tmp_path, 3)
    assert r.returncode == 0, r.stderr[-300:]
    assert "record_demo.py" in log.read_text()


def test_a_gpu_that_is_released_a_few_seconds_late_is_waited_for_not_treated_as_busy(tmp_path):
    """After SGLang / vLLM stop, their memory goes away a moment later; aborting there would kill an unattended run."""
    root = tmp_path / "llm-stack"
    (root / "serving").mkdir(parents=True)
    log, counter = tmp_path / "py.log", tmp_path / "n"
    py = _stub(tmp_path / "py", f'echo "$@" >> {log}\n')
    smi = _stub(tmp_path / "smi", f'n=$(cat {counter} 2>/dev/null || echo 0); echo $((n+1)) > {counter}; [ $n -lt 2 ] && echo 18000 || echo 3\n')
    env = {**os.environ, "ROOT": str(root), "PY": str(py), "NVSMI": str(smi), "ST": str(tmp_path / "st.txt"), "REPS": "1", "CS": "1",
           "IDLE_WAIT_S": "20", "IDLE_POLL_S": "1"}
    env.pop("DRY_RUN", None)
    r = subprocess.run(["bash", str(SCRIPT), "ladder"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and "record_demo.py" in log.read_text(), r.stderr[-300:]
    assert int(counter.read_text()) >= 3, "it must have polled until the memory dropped"


def test_every_long_running_command_has_a_timeout_so_one_hung_cell_cannot_eat_the_night():
    for phase in ("ladder", "sglang", "vllm", "openloop", "c5"):
        for c in dry(phase):
            if any(k in c for k in ("record_demo.py", "run_sglang.sh", "run_vllm.sh", "c5_greedy_equal.py")):
                assert re.match(r"timeout \d+ ", c), f"{phase}: no timeout around: {c[:120]}"
                assert int(c.split()[1]) <= 2400


def test_all_ends_with_the_tarball_and_the_status_page():
    cmds = dry("all")
    assert "stage_d_summary.py" in cmds[-2] and "stage_d_results.tgz" in cmds[-1], cmds[-2:]
    assert cmds.index(next(c for c in cmds if "c5_greedy_equal.py" in c)) < len(cmds) - 2
