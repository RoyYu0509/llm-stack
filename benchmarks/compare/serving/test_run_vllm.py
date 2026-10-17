"""run_vllm.sh is shell, so it is tested through DRY_RUN: the two commands it would run (the `vllm serve` line and the
closed-loop client line) are checked flag by flag, and every client flag must exist in closed_loop.py's real parser."""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPT = HERE / "run_vllm.sh"


def dry(*args, **env):
    e = {**os.environ, "DRY_RUN": "1", "HF_DIR": "/x/hf", "PY": "/v/bin/python", **env}
    p = subprocess.run(["bash", str(SCRIPT), *(args or ("default_noprefix",))], capture_output=True, text=True, env=e)
    return p, [ln[2:] for ln in p.stderr.splitlines() if ln.startswith("+ ")]


def test_server_line_is_vllm_defaults_minus_prefix_caching_and_sends_no_usage_stats():
    p, cmds = dry()
    assert p.returncode == 0 and len(cmds) == 2
    server = cmds[0]
    for want in ("VLLM_NO_USAGE_STATS=1", "DO_NOT_TRACK=1", "/v/bin/vllm serve /x/hf", "--dtype bfloat16", "--port 8000",
                 "--host 127.0.0.1", "--served-model-name default", "--gpu-memory-utilization 0.8",
                 "--no-enable-prefix-caching"):
        assert want in server, f"missing {want!r} in: {server}"
    for forbidden in ("--enforce-eager", "--enable-prefix-caching", "--attention-backend", "--max-num-seqs",
                      "--no-enable-chunked-prefill", "--max-model-len", "--compilation-config"):
        assert forbidden not in server, f"{forbidden} would change the 'default' row: {server}"


def test_client_line_matches_the_sglang_rows_grid_and_rules():
    _, cmds = dry(N_MULT="8", N_MIN="32")
    client = cmds[1]
    for want in ("closed_loop.py run --backend vllm", "--system vllm_default_noprefix_bfloat16", "--url http://127.0.0.1:8000",
                 "--served-model-name default", "--warm-mult 2", "--n-mult 8", "--n-min 32", "--concurrency 1 8 32 64",
                 "--reps 3", "results/final/serving.csv"):
        assert want in client, f"missing {want!r} in: {client}"
    notes = client.split("--notes ", 1)[1]
    assert notes.startswith("flags: --dtype bfloat16") and "--no-enable-prefix-caching" in notes and "/x/hf" not in notes


def test_every_client_flag_exists_in_the_real_parser():
    _, cmds = dry()
    helptext = subprocess.run([sys.executable, str(HERE / "closed_loop.py"), "run", "--help"],
                              capture_output=True, text=True).stdout
    client_flags_only = cmds[1].split(" --notes ")[0]   # the notes text quotes the vllm-serve flags, which are not client flags
    flags = set(re.findall(r"(?<=\s)--[a-z][a-z-]+", client_flags_only))
    missing = sorted(f for f in flags if f not in helptext)
    assert not missing, f"run_vllm.sh passes flags closed_loop.py does not know: {missing}"
    assert "vllm" in re.search(r"--backend \{([^}]*)\}", helptext).group(1)


def test_env_overrides_reach_the_commands():
    _, cmds = dry(PORT="8100", SERVING_CSV="/tmp/o.csv", CONC="8", REPS="1", MODEL_NAME="m", GPU_MEM_UTIL="0.5",
                  ATTN="flashinfer", DTYPE="float16", NOTES_EXTRA="SAME MACHINE")
    server, client = cmds
    assert "--port 8100" in server and "--served-model-name m" in server and "--gpu-memory-utilization 0.5" in server
    assert "--attention-backend flashinfer" in server and "--dtype float16" in server
    assert "--system vllm_default_noprefix_flashinfer_float16" in client, "a backend override must not overwrite the default row"
    assert "--url http://127.0.0.1:8100" in client and "--out /tmp/o.csv" in client
    assert "--concurrency 8 " in client and "--reps 1" in client and client.rstrip().endswith("SAME MACHINE")


def test_nograph_variant_keeps_compile_and_only_switches_the_cuda_graph_off():
    p, cmds = dry("nograph_noprefix")
    assert p.returncode == 0 and len(cmds) == 2
    server, client = cmds
    assert '--compilation-config {"cudagraph_mode":"NONE"}' in server and "--no-enable-prefix-caching" in server
    assert "--enforce-eager" not in server, "--enforce-eager also switches torch.compile off: not the engine's configuration"
    assert "--system vllm_nograph_noprefix_bfloat16" in client
    assert '--compilation-config {"cudagraph_mode":"NONE"}' in client.split("--notes ", 1)[1], "the row's notes must say what was switched off"


def test_the_json_flag_stays_one_argument_in_the_real_command():
    """DRY_RUN prints with spaces; the real launch uses the array, so the JSON must not be word-split."""
    src = SCRIPT.read_text()
    assert "'{\"cudagraph_mode\":\"NONE\"}'" in src and 'env "${SERVER_ENV[@]}" "$VLLM" "${SERVE[@]}"' in src


def test_unknown_variant_is_refused():
    p, cmds = dry("cb_only")
    assert p.returncode == 2 and not cmds and "ABORT" in p.stderr


def test_dry_run_touches_nothing_and_never_manages_the_instance():
    env_file = HERE.parent / "results" / "final" / "logs" / "serving" / "env_vllm_default_noprefix_bfloat16.txt"
    before = env_file.exists()
    p, cmds = dry()
    assert env_file.exists() == before, "DRY_RUN must not write into results/final"
    assert not re.search(r"vastai|destroy|nvidia-smi -pl|pkill", p.stderr)
