#!/usr/bin/env bash
# Stage A on a prepared box (setup_gpu_box.sh done, env_check PASS): bf16 correctness checks -> attention timing sweep -> gate.
# It does NOT run profile_prefill: that waits for the gate (D-7). It does not spend money or touch the instance.
#   progress: $ROOT/STAGE_A_STATUS.txt (last line STAGE_A_DONE)   logs: results/final/logs/stage_a_*.log
# Export BENCH_GIT_SHA / BENCH_GIT_DIRTY from the LOCAL tree before calling (the box has no .git).
set -u
ROOT=${ROOT:-/workspace/llm-stack}; CMP=$ROOT/benchmarks/compare; FINAL=$CMP/results/final; LOGS=$FINAL/logs
mkdir -p "$LOGS"
# shellcheck disable=SC1091
source /workspace/venv-fa/bin/activate
export PYTHONPATH=$ROOT/training/cs336-basics:$ROOT/training
ST=/workspace/STAGE_A_STATUS.txt; : > "$ST"
st() { echo "$(date -Is) $*" >> "$ST"; }
cd "$CMP" || exit 1
CSV=$FINAL/attention.csv
# import guard: the registry import needs the editable install; without it every cell of three systems fails after minutes
python -c "from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNEL_REGISTRY as R; import flash_attn; print('guard ok', sorted(R), flash_attn.__version__)" \
  > "$LOGS/stage_a_import_guard.log" 2>&1 || { st "ABORT import guard failed (see logs/stage_a_import_guard.log); run setup_gpu_box.sh again"; exit 2; }

st "gpu state"; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > "$LOGS/stage_a_nvidia_smi_q.log" 2>&1
st "C1 bf16";            python attention/bench_attention.py --check --out "$CSV"               > "$LOGS/stage_a_c1.log" 2>&1;       st "C1 rc=$?"
st "C6 flash raises";    python attention/bench_attention.py --check-flash-raises --out "$CSV"  > "$LOGS/stage_a_c6.log" 2>&1;       st "C6 rc=$?"
st "inductor";           python attention/bench_attention.py --check-inductor --out "$CSV"      > "$LOGS/stage_a_inductor.log" 2>&1; st "inductor rc=$?"
st "sweep (5 systems x 8 seqs x fwd,fwd+bwd; one fresh process per cell)"
python attention/bench_attention.py --mode both --reps 5 --out "$CSV"                           > "$LOGS/stage_a_sweep.log" 2>&1;    st "sweep rc=$?"
python attention/gate.py --csv "$CSV"                                                           > "$LOGS/stage_a_gate.log" 2>&1;     st "gate rc=$? (0 carry on, 3 triggered -> stop and discuss, 4 cannot evaluate)"
st "STAGE_A_DONE"
