#!/usr/bin/env bash
# SGLang in its own venv (never touch venv-fa: the attention results were measured there).
#   SGLANG_VERSION (default 0.5.10) | progress: $ROOT/SGLANG_SETUP_STATUS.txt (last line SGLANG_SETUP_DONE | SGLANG_SETUP_FAILED)
#
# Why a pinned version: the newest SGLang (0.5.22) wants torch 2.14 + the CUDA 13 runtime = driver >= 580. This box has driver 550
# (CUDA 12.4). The newest release that still resolves to a CUDA 12.x runtime is 0.5.10 (torch 2.9.1 + CUDA 12.8 runtime, which works
# on 550 only through CUDA 12.x minor-version compatibility -- so it is smoke-tested here instead of assumed).
set -u
ROOT=${ROOT:-/workspace}; VENV=$ROOT/venv-sglang; VER=${SGLANG_VERSION:-0.5.10}; ST=$ROOT/SGLANG_SETUP_STATUS.txt
: > "$ST"; st() { echo "$(date -Is) $*" | tee -a "$ST"; }
st "uv venv"; uv venv "$VENV" --python 3.12 >/dev/null 2>&1 || { st "SGLANG_SETUP_FAILED uv venv"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
st "install sglang==$VER (timeout 25 min)"
timeout 1500 uv pip install "sglang==$VER" >"$ROOT/install_sglang_$VER.log" 2>&1; rc=$?
tail -3 "$ROOT/install_sglang_$VER.log" | tee -a "$ST"
[ $rc = 0 ] || { st "SGLANG_SETUP_FAILED install rc=$rc"; exit 1; }
python - <<'PY' 2>&1 | tee -a "$ST"
import sglang, torch
print("sglang", sglang.__version__, "| torch", torch.__version__, "| cuda runtime", torch.version.cuda, "| cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16); torch.cuda.synchronize(); print("matmul ok", float((x @ x).float().abs().mean()) > 0)
PY
uv pip list 2>/dev/null | grep -i -E "^(sglang|sglang-kernel|sgl-kernel|torch|flashinfer-python|transformers|triton|nvidia-cuda-runtime) " | tee -a "$ST"
st "SGLANG_SETUP_DONE"
