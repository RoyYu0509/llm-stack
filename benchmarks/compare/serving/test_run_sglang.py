"""run_sglang.sh is shell: the hf_cb variant and the open-loop switch are tested by replacing $PY with a stub that
records its arguments (the SGLang variants need a server and are covered on the GPU box)."""
import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent / "run_sglang.sh"


def call(tmp_path, variant, **env):
    stub, log = tmp_path / "py", tmp_path / "args.log"
    stub.write_text('#!/usr/bin/env bash\nIFS=$\'\\037\'\necho "$*" >> "$ARGLOG"\n')   # one call per line, args split by \x1f (JSON survives)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    e = {**os.environ, "PY": str(stub), "HF_DIR": "/model", "ARGLOG": str(log), "SERVING_CSV": str(tmp_path / "o.csv"), **env}
    for k in ("ARRIVAL_RATES", "CB_OPTS", "CB_TAG", "HF_ATTN", "CONC"):
        if k not in env:
            e.pop(k, None)
    subprocess.run(["bash", str(SCRIPT), variant], env=e, check=True, capture_output=True, text=True)
    return [ln.split("\x1f") for ln in log.read_text().splitlines()]


def test_hf_cb_closed_loop_uses_the_concurrency_grid_and_flash_attention_by_default(tmp_path):
    (a,) = call(tmp_path, "hf_cb", DTYPE="bfloat16", CONC="1 8 32 64")
    s = " ".join(a)
    assert "closed_loop.py run --backend hf_cb --system hf_cb_flash_attention_2_bfloat16 " in s
    assert "--attn flash_attention_2" in s and "--concurrency 1 8 32 64" in s and "--arrival-rate" not in s
    assert "--model-dir /model" in s and "--dtype bfloat16" in s


def test_arrival_rates_switch_every_variant_to_open_loop(tmp_path):
    (a,) = call(tmp_path, "hf_cb", DTYPE="bfloat16", ARRIVAL_RATES="10 20 40", ARRIVAL_CV="2", DURATION="45", MAX_INFLIGHT="64")
    s = " ".join(a)
    assert "--arrival-rate 10 20 40 --arrival-cv 2 --duration 45 --max-inflight 64" in s
    assert "--concurrency" not in s and "--slo-ttft-s 2 --slo-tpot-ms 100" in s


def test_cb_opts_travel_as_one_json_argument_and_cb_tag_names_the_variant(tmp_path):
    (a,) = call(tmp_path, "hf_cb", DTYPE="bfloat16", CB_OPTS='{"use_cuda_graph": true}', CB_TAG="graph")
    assert a[a.index("--cb-opts") + 1] == '{"use_cuda_graph": true}'
    assert a[a.index("--system") + 1] == "hf_cb_flash_attention_2_bfloat16_graph"


def test_serial_hf_variant_is_untouched(tmp_path):
    (a,) = call(tmp_path, "hf", DTYPE="bfloat16", CONC="1")
    s = " ".join(a)
    assert "--backend hf --system hf_generate_bfloat16" in s and "hf_cb" not in s
