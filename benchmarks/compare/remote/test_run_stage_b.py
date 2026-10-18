"""The matrix script is shell, so its completeness is tested through DRY_RUN: every cell of the D-7 design must appear exactly once."""
import os
import re
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent / "run_stage_b.sh"


def dry(phase="all", **env):
    import tempfile

    root = Path(tempfile.mkdtemp()) / "llm-stack"          # the script mkdirs its log / raw dirs under ROOT
    e = {**os.environ, "DRY_RUN": "1", "ROOT": str(root), **env}
    out = subprocess.run(["bash", str(SCRIPT), phase], capture_output=True, text=True, env=e, check=True).stderr
    return [ln[2:].replace(str(root), "<ROOT>") for ln in out.splitlines() if ln.startswith("+ ")]


def test_engine_phase_has_every_arm_and_concurrency_once():
    cmds = [c for c in dry("engine") if "record_demo.py" in c]
    assert len(cmds) == 12
    got = {(re.search(r"--attn (\w+)", c).group(1), int(re.search(r"--concurrency (\d+)", c).group(1))) for c in cmds}
    assert got == {(a, c) for a in ("naive", "sdpa", "fa2") for c in (1, 8, 32, 64)}
    for c in cmds:
        assert "--dtype bfloat16" in c and "--reps 3" in c and "workload_mixed4096.json" in c
        assert "--prefill-last-logits" not in c and "--batched-sampling" not in c, \
            "the ladder must be naive -> sdpa -> fa2 only: no extra AI switches"
    n = {int(re.search(r"--concurrency (\d+)", c).group(1)): int(re.search(r"--n-requests (\d+)", c).group(1)) for c in cmds}
    w = {int(re.search(r"--concurrency (\d+)", c).group(1)): int(re.search(r"--warmup-requests (\d+)", c).group(1)) for c in cmds}
    assert n == {1: 32, 8: 64, 32: 256, 64: 512}, "requests = max(8 x C, 32)"
    assert w == {1: 8, 8: 16, 32: 64, 64: 128}, "warm-up = max(2 x C, 8)"
    assert len([c for c in dry("engine") if "closed_loop.py ingest" in c]) == 12


def test_hf_phase_covers_the_three_attention_implementations_at_c1_and_c32():
    cmds = [c for c in dry("hf") if "closed_loop.py run" in c]
    assert sorted(re.search(r"--attn (\w+)", c).group(1) for c in cmds) == ["eager", "flash_attention_2", "sdpa"]
    assert all("--concurrency 1 32" in c and "--n 64" in c and "--dtype bfloat16" in c for c in cmds)


def test_sglang_phase_has_the_three_variants_and_no_torch_native():
    cmds = [c for c in dry("sglang") if "run_sglang.sh" in c]
    assert len(cmds) == 3
    assert any("ATTN=flashinfer" in c and c.rstrip().endswith("cb_only") or ("ATTN=flashinfer" in c and " cb_only" in c) for c in cmds)
    assert any("ATTN=triton" in c and " cb_only" in c for c in cmds)
    assert any("ATTN=flashinfer" in c and "default_noradix" in c and "CUDA_GRAPH_MAX_BS=64" in c for c in cmds)
    assert all("DTYPE=bfloat16" in c and 'CONC=1 8 32 64' in c and "N_MIN=32" in c and "N_MULT=8" in c for c in cmds)
    assert not any("torch_native" in c for c in cmds)


def test_attrib_phase_has_cprofile_decode_steps_c5_and_the_fa2_check():
    cmds = dry("attrib")
    assert sum("--cprofile" in c and "record_demo.py" in c for c in cmds) == 1
    assert not any(" -m cProfile" in c for c in cmds), "profile only the timed trial, not interpreter start-up + warm-up"
    assert sorted(re.search(r"--attn (\w+)", c).group(1) for c in cmds if "profile_decode_step.py" in c) == ["fa2", "naive", "sdpa"]
    assert sum("c5_greedy_equal.py" in c for c in cmds) == 2
    assert any("--dtype float32" in c and "c5_greedy_equal.py" in c for c in cmds)
    assert any("check_fa2_decode.py" in c and "--checkpoint" in c for c in cmds)


def test_everything_writes_under_results_final():
    for c in dry("all"):
        if "convert_190m_to_hf" in c:       # the HF model directory is an input for the other phases, not a result
            continue
        for m in re.findall(r"(?:--out|-o) (\S+)", c):
            assert "results/final" in m, f"output outside results/final: {m}"


def test_smoke_covers_every_component_of_the_real_phases_at_toy_size():
    cmds = dry("smoke")
    assert sum("record_demo.py" in c and "--cprofile" not in c for c in cmds) == 3
    assert sorted(re.search(r"--attn (\w+)", c).group(1) for c in cmds if "closed_loop.py run" in c) == ["eager", "flash_attention_2", "sdpa"]
    assert sum("closed_loop.py ingest" in c for c in cmds) == 3
    assert sum("--cprofile" in c for c in cmds) == 1 and sum("analyze_cprofile.py" in c for c in cmds) == 1
    assert sum("profile_decode_step.py" in c for c in cmds) == 3
    assert sum("c5_greedy_equal.py" in c for c in cmds) == 2 and sum("check_fa2_decode.py" in c for c in cmds) == 1
    for c in cmds:
        for m in re.findall(r"(?:--out|-o) (\S+)", c):
            assert "results/final" not in m, f"the smoke test must not write into results/final: {m}"
