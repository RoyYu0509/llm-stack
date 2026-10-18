#!/usr/bin/env bash
# Stage A, part 2 (after the main sweep, gate NOT triggered): diagnostics + profiling. Same rules as run_stage_a.sh.
#   1. layout diagnostic: the flash-attn package called with the SDPA-identical memory layout (flash_attn_pkg_flat)
#   2. short sequences re-measured with 15 repetitions (S<=512 cells of the main sweep were noisy: std/mean up to 30%)
#   3. profile_prefill: torch.profiler + Nsight Systems, all lengths x all five systems
# progress: /workspace/STAGE_A2_STATUS.txt (last line STAGE_A2_DONE)
set -u
ROOT=${ROOT:-/workspace/llm-stack}; CMP=$ROOT/benchmarks/compare; FINAL=$CMP/results/final; LOGS=$FINAL/logs
mkdir -p "$LOGS"
# shellcheck disable=SC1091
source /workspace/venv-fa/bin/activate
export PYTHONPATH=$ROOT/training/cs336-basics:$ROOT/training
[ -f /workspace/nsys_env.sh ] && source /workspace/nsys_env.sh
export NSYS_BIN=${NSYS_BIN:-/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys}
ST=/workspace/STAGE_A2_STATUS.txt; : > "$ST"
st() { echo "$(date -Is) $*" >> "$ST"; }
cd "$CMP" || exit 1
SYS="sdpa_flash flash_attn_pkg flash_attn_pkg_flat triton_fa_mine"
st "layout diagnostic (fwd, all lengths)"
python attention/bench_attention.py --mode fwd --systems $SYS --out "$FINAL/attention_diagnostic.csv" > "$LOGS/stage_a2_layout.log" 2>&1; st "layout rc=$?"
st "short sequences, 15 reps"
python attention/bench_attention.py --mode fwd --systems $SYS --seqs 128 256 512 --reps 15 --out "$FINAL/attention_diagnostic_shortS.csv" > "$LOGS/stage_a2_shortS.log" 2>&1; st "shortS rc=$?"
st "profile_prefill (torch + nsys)"
python attention/profile_prefill.py --tool both > "$LOGS/stage_a2_profile.log" 2>&1; st "profile rc=$?"
st "STAGE_A2_DONE"
