"""run_kv_sweep.sh is shell, so it is tested through DRY_RUN: the grid must be exactly sizes x lengths x {nokv, kv}, the two arms of a cell
must differ ONLY by --no-kv-cache (anything else would make the ratio mean something other than 'what the KV cache buys'), every flag
must exist in the real record_demo parser, and kv_sweep_summary.py must read back what the script writes."""
import csv
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import kv_sweep_summary as sm

HERE = Path(__file__).parent
SCRIPT = HERE / "run_kv_sweep.sh"
REPO = HERE.parents[2]
SIZES = ["190m", "1.4b", "2.8b", "6.9b"]
LENS = {"190m": [1024, 4096, 16384], "1.4b": [1024, 4096, 16384], "2.8b": [1024, 4096, 16384], "6.9b": [1024, 4096, 12288]}   # 6.9b @ 16384 OOMs


def dry(phase="sweep", **env):
    root = Path(tempfile.mkdtemp()) / "llm-stack"
    e = {**os.environ, "DRY_RUN": "1", "ROOT": str(root), **env}
    out = subprocess.run(["bash", str(SCRIPT), phase], capture_output=True, text=True, env=e, check=True).stderr
    return [ln[2:].replace(str(root), "<ROOT>") for ln in out.splitlines() if ln.startswith("+ ")]


def rec(cmds):
    return [c for c in cmds if "record_demo.py" in c]


def cell(cmd):
    m = re.search(r"--arm-name (nokv|kv)_(\S+?)_L(\d+) ", cmd)
    return m.group(1), m.group(2), int(m.group(3))


def val(cmd, flag):
    return re.search(rf"{flag} (\d+)", cmd).group(1)


def test_grid_is_every_size_by_every_length_with_both_arms_exactly_once():
    cmds = rec(dry())
    assert sorted(cell(c) for c in cmds) == sorted((a, s, L) for s in SIZES for L in LENS[s] for a in ("nokv", "kv"))


def test_the_two_arms_of_a_cell_differ_only_by_the_kv_cache_and_its_prefill_logits_switch():
    by = {}
    for c in rec(dry()):
        arm, s, L = cell(c)
        norm = re.sub(r"--arm-name \S+|--out \S+", "", c).replace("--no-kv-cache", "").replace("--prefill-last-logits", "")
        norm = re.sub(r"--n-requests \d+|--reps \d+|--warmup-requests \d+", "", norm)   # the slow no-KV arm may use fewer requests
        assert ("--no-kv-cache" in c) == (arm == "nokv")
        assert ("--prefill-last-logits" in c) == (arm == "kv"), "last-position logits belong to the kv arm only (the no-KV path has no such switch)"
        assert "--fast-rope-init" in c
        by.setdefault((s, L), set()).add(" ".join(norm.split()))
    assert all(len(v) == 1 for v in by.values()), "the arms of a cell must share model, prompt, new tokens, attention and dtype"


def test_every_cell_is_single_request_bf16_sdpa_seq_with_an_exact_length_prompt_that_fits_the_context():
    for c in rec(dry()):
        _, _, L = cell(c)
        assert "--mode seq" in c and "--attn sdpa" in c and "--dtype bfloat16" in c and "--device cuda" in c
        assert val(c, "--concurrency") == "1" and "--synthetic" in c
        assert f"--prompt-len {L} {L}" in c
        assert int(val(c, "--context-length")) >= L + int(val(c, "--new-tokens"))


def test_both_arms_generate_the_same_number_of_tokens():
    assert {val(c, "--new-tokens") for c in rec(dry())} == {"8"}


def test_only_the_slow_big_nokv_cells_use_fewer_requests():
    for c in rec(dry()):
        arm, s, L = cell(c)
        slow = arm == "nokv" and (s, L) in (("2.8b", 16384), ("6.9b", 12288))
        assert (int(val(c, "--n-requests")), int(val(c, "--reps"))) == ((2, 2) if slow else (4, 3)), (arm, s, L)


def test_every_recording_is_ingested_under_the_system_name_the_summary_reads():
    ingests = [c for c in dry() if "closed_loop.py ingest" in c]
    assert len(ingests) == 24
    for c in ingests:
        assert sm.SYSTEM.match(re.search(r"--system (\S+)", c).group(1))
        assert "serving_kv_sweep.csv" in c


def test_the_flags_exist_in_the_real_record_demo_parser():
    src = (REPO / "serving" / "benchmarks" / "record_demo.py").read_text()
    for c in rec(dry()):
        for flag in re.findall(r"(--[a-z0-9-]+)", c.split("record_demo.py", 1)[1]):
            assert f'"{flag}"' in src, f"{flag} is not a record_demo.py option"


def test_script_reads_the_env_vars_it_documents():
    cmds = dry(SIZES="1.4b 6.9b", LENS="1024", LENS_69="2048", NEW="4")
    assert sorted(cell(c) for c in rec(cmds)) == [("kv", "1.4b", 1024), ("kv", "6.9b", 2048), ("nokv", "1.4b", 1024), ("nokv", "6.9b", 2048)]
    assert all("--new-tokens 4" in c for c in rec(cmds))


def test_smoke_has_tiny_cells_and_the_card_filling_probe():
    cmds = rec(dry("smoke"))
    assert len(cmds) == 3
    probe = [c for c in cmds if "--preset 6.9b" in c]
    assert len(probe) == 1 and "--prompt-len 12288 12288" in probe[0] and "--no-kv-cache" not in probe[0]


def test_wrapup_summarises_then_packs_and_is_last_in_all():
    cmds = dry("all")
    assert any("kv_sweep_summary.py" in c for c in cmds) and any(c.startswith("tar czf") for c in cmds)
    assert max(i for i, c in enumerate(cmds) if "record_demo.py" in c) < min(i for i, c in enumerate(cmds) if "kv_sweep_summary.py" in c)


def test_busy_gpu_stops_the_script_before_anything_runs(tmp_path):
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/usr/bin/env bash\necho 20000\n")
    fake.chmod(0o755)
    e = {**os.environ, "NVSMI": str(fake), "IDLE_WAIT_S": "0", "ROOT": str(tmp_path / "llm-stack"), "ST": str(tmp_path / "status.txt"),
         "PY": "/bin/false"}
    r = subprocess.run(["bash", str(SCRIPT), "sweep"], capture_output=True, text=True, env=e)
    assert r.returncode == 3 and "not idle" in r.stderr
    assert "ABORT" in (tmp_path / "status.txt").read_text()


# ---- the summary ------------------------------------------------------------------------------------------------------------------

def write_csv(path, drop=None, tpot=None):
    tpot = tpot or {}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "system", "variant", "metric", "value", "concurrency"])
        w.writeheader()
        for s in SIZES:
            for L in LENS[s]:
                for a in ("nokv", "kv"):
                    if (a, s, L) == drop:
                        continue
                    for metric, v in (("tpot_ms_p50", tpot.get((a, s, L), 100.0 if a == "nokv" else 10.0)), ("ttft_ms_p50", 50.0)):
                        w.writerow(dict(experiment="I1", system=f"engine_kvsweep_{a}_{s}_L{L}_bfloat16", variant="new_tokens=8;closed_loop",
                                        metric=metric, value=v, concurrency=1))


def test_summary_expects_the_same_cells_the_script_runs():
    ran = {cell(c) for c in rec(dry())}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.csv"
        write_csv(p)
        lines, missing = sm.summarize(p)
    assert missing == 0 and ran == {(a, s, L) for s in SIZES for L in LENS[s] for a in ("nokv", "kv")}
    assert len([ln for ln in lines if ln.strip().split()[0] in SIZES]) == sum(len(v) for v in LENS.values())


def test_summary_says_missing_and_computes_the_speedup_from_the_two_arms(tmp_path):
    p = tmp_path / "x.csv"
    write_csv(p, drop=("kv", "6.9b", 12288), tpot={("nokv", "190m", 1024): 25.0, ("kv", "190m", 1024): 20.0})
    lines, missing = sm.summarize(p)
    text = "\n".join(lines)
    assert missing == 1 and "MISSING kv" in text and "23/24 cells present" in text.replace("\n", " ")
    row = next(ln for ln in lines if ln.strip().startswith("190m") and " 1024 " in ln)
    assert "1.2x" in row


def test_summary_reads_peak_memory_from_the_recordings_and_prints_the_fallback_cell(tmp_path):
    p, raw = tmp_path / "x.csv", tmp_path / "raw"
    raw.mkdir()
    write_csv(p, drop=("kv", "6.9b", 12288))
    (raw / "kv_6.9b_L8192_r1.json").write_text(json.dumps({"meta": {"peak_mem_gb": 20.5}}))
    (raw / "kv_6.9b_L8192_r2.json").write_text(json.dumps({"meta": {"peak_mem_gb": 21.5}}))
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        for a, t in (("nokv", 1000.0), ("kv", 50.0)):
            w.writerow(["I1", f"engine_kvsweep_{a}_6.9b_L8192_bfloat16", "v", "tpot_ms_p50", t, 1])
    lines, _ = sm.summarize(p, raw)
    row = next(ln for ln in lines if " 8192 " in ln)
    assert "21.5" in row and "fall-back" in row and "20.0x" in row
