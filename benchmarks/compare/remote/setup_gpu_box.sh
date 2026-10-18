#!/usr/bin/env bash
# Prepare a freshly rented Vast single-GPU box for benchmarks/compare (idempotent, ~5-10 min, nothing here spends money).
#
#   bash setup_gpu_box.sh                     # ROOT=/workspace
#   progress: $ROOT/SETUP_STATUS.txt  (last line SETUP_DONE | SETUP_FAILED)
#
# One environment for every attention row (D-7): venv-fa = torch 2.6.0+cu126 + its bundled triton + flash-attn 2.8.3 (prebuilt
# wheel; needs the C++11-ABI=True torch build, which is why the cu126 wheel index is used). NO source compile of flash-attn.
# Python 3.12 because the repo uses 3.12-only syntax. Nsight Systems comes from the CUDA apt repo if the image has none.
# The box has no .git: the launcher exports BENCH_GIT_SHA / BENCH_GIT_DIRTY from the local tree (schema.py reads them).
set -u
ROOT=${ROOT:-/workspace}; VENV=$ROOT/venv-fa; ST=$ROOT/SETUP_STATUS.txt
mkdir -p "$ROOT"; : > "$ST"
st() { echo "$(date -Is) $*" | tee -a "$ST"; }
fail() { st "SETUP_FAILED: $*"; exit 1; }

st "uv venv (python 3.12)"
uv venv "$VENV" --python 3.12 >/dev/null 2>&1 || fail "uv venv"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

st "torch 2.6.0 (cu126 index)"
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2 | tee -a "$ST"
python - <<'PY' 2>&1 | tee -a "$ST"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cxx11_abi", torch._C._GLIBCXX_USE_CXX11_ABI,
      "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY

st "project dependencies"
uv pip install einops jaxtyping regex tqdm numpy pytest transformers==5.15.1 safetensors pydantic 2>&1 | tail -2 | tee -a "$ST"

st "project packages (editable; cs336_systems/__init__.py reads importlib.metadata.version('llm-training') with NO fallback)"
# the 2026-10-15 'attempt 1' (whole pipeline re-run) and the 2026-10-16 first stage-A try both died on exactly this
if [ -d "$ROOT/llm-stack/training/cs336-basics" ] && [ -f "$ROOT/llm-stack/training/pyproject.toml" ]; then
  uv pip install --no-deps -e "$ROOT/llm-stack/training/cs336-basics" -e "$ROOT/llm-stack/training" 2>&1 | tail -4 | tee -a "$ST"
  PYTHONPATH="" python - <<'PY' 2>&1 | tee -a "$ST"
from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNEL_REGISTRY as R
import cs336_systems
print("registry ok:", sorted(R), "| cs336_systems", cs336_systems.__version__)
PY
else
  st "WARN: upload the code first (training/cs336-basics, training/pyproject.toml), then re-run this script"
fi

st "flash-attn 2.8.3 prebuilt wheel"
FA_OK=0
for abi in TRUE FALSE; do
  uv pip uninstall flash-attn >/dev/null 2>&1
  uv pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abi${abi}-cp312-cp312-linux_x86_64.whl" >/dev/null 2>&1
  r=$(python -c "import flash_attn; from flash_attn import flash_attn_func, flash_attn_with_kvcache; print('OK', flash_attn.__version__)" 2>&1 | tail -1 | cut -c1-140)
  st "flash-attn abi${abi} -> $r"
  case "$r" in OK*) FA_OK=1; break;; esac
done
[ $FA_OK = 1 ] || st "WARN flash-attn wheel not usable: flash_attn_pkg rows will be 'unavailable', SDPA flash stands in for FA2"

st "nsight systems (pinned to a release that matches the driver's CUDA)"
# 2026-10-16: nsys 2026.3.2 + driver 550 (CUDA 12.4) recorded ONE kernel out of ten and no CUDA API calls; 2024.5.1 recorded all.
# So: install the 2024.x line (CUDA 12.x era) and PROVE it captures a 10-kernel loop before anything depends on it.
NSYS_PKG=${NSYS_PKG:-nsight-systems-2024.5.1}
NSYS=/opt/nvidia/nsight-systems/${NSYS_PKG#nsight-systems-}/target-linux-x64/nsys
if [ ! -x "$NSYS" ]; then
  apt-get install -y --no-install-recommends "$NSYS_PKG" >"$ROOT/apt_nsys.log" 2>&1 || st "WARN apt install $NSYS_PKG failed (see apt_nsys.log)"
fi
if [ -x "$NSYS" ]; then
  cat > "$ROOT/nsys_selftest.py" <<'PY'
import torch
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
for _ in range(3): x @ x
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(10): x @ x
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
PY
  (cd "$ROOT" && "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
      --capture-range-end=stop --force-overwrite=true -o nsys_selftest python nsys_selftest.py >nsys_selftest.out 2>&1)
  N=$("$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-export=true "$ROOT/nsys_selftest.nsys-rep" 2>/dev/null \
      | awk -F, '$3 ~ /^[0-9]+$/ {n+=$3} END {print n+0}')
  if [ "${N:-0}" -ge 10 ]; then st "nsys ok: $("$NSYS" --version | head -1); self-test captured $N kernels (expected 10)"; echo "export NSYS_BIN=$NSYS" > "$ROOT/nsys_env.sh"
  else st "WARN nsys self-test captured ${N:-0}/10 kernels: its CUDA trace is truncated on this driver; profile_prefill will refuse nsys cells"; fi
else
  st "WARN no usable nsys: profile_prefill will run torch.profiler only"
fi
st "SETUP_DONE"
