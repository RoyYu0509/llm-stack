#!/usr/bin/env bash
# I1-f -- vLLM under the same closed-loop workload as the self-written engine and SGLang (run_sglang.sh is the model).
#
#   bash serving/run_vllm.sh default_noprefix   # `vllm serve` defaults, minus prefix caching (CUDA graph ON: the "ceiling" row)
#   bash serving/run_vllm.sh nograph_noprefix   # the same, plus -cc '{"cudagraph_mode":"NONE"}': torch.compile on, CUDA graph OFF
#                                               # (the same-feature-set row: the self-written engine has torch.compile but no CUDA graph)
#
# vLLM cannot switch continuous batching off, so there is no "cb_only" row. "default_noprefix" is what a user gets from
# `vllm serve` with ONE change, --no-enable-prefix-caching (the workload repeats prompts; the SGLang counterpart is
# default_noradix). What stays on and is part of the number: chunked prefill (max_num_batched_tokens 8192), torch.compile
# + CUDA graphs, the attention backend vLLM picks by itself (FLASH_ATTN = FA2 on an sm86 card in 0.31.0; the log line is
# copied into logs/serving/env_<system>.txt), and the HTTP front end with tokenizer / detokenizer (the SGLang rows pay for
# HTTP as well; the self-written engine is timed in-process, which favours the engine).
#
# Needs: vllm in its OWN venv (never touch venv-fa / venv-sglang), the HF-format 190M model dir that run_sglang.sh uses
# (convert_hf/convert_190m_to_hf.py --tokenizer gpt2), and the workload json.
#
# Env: HF_DIR (required) PY (the vLLM venv's python; also runs the client, which only needs aiohttp) VLLM (default: vllm next to PY)
#      DTYPE(bfloat16) PORT(8000) CONC("1 8 32 64") REPS(3) WORKLOAD (default workloads/workload_wiki128.json)
#      SERVING_CSV (default results/final/serving.csv) N_MULT / N_MIN (requests per level = max(N_MIN, N_MULT x C))
#      GPU_MEM_UTIL(0.8) MODEL_NAME(default) ATTN (optional --attention-backend; unset = vLLM's own choice)
#      READY_TIMEOUT(1200 s) NOTES_EXTRA (appended to every row's notes)
#      DRY_RUN=1  print the server command and the client command instead of running them (tested on the laptop)
set -euo pipefail
VARIANT=${1:?variant: default_noprefix}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$HERE/../results/final"; LOGS="$RESULTS/logs/serving"
HF_DIR=${HF_DIR:?set HF_DIR to the converted HF model directory}
DTYPE=${DTYPE:-bfloat16}; PORT=${PORT:-8000}; CONC=${CONC:-"1 8 32 64"}; REPS=${REPS:-3}
PY=${PY:-python}; VLLM=${VLLM:-$(dirname "$PY")/vllm}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}; MODEL_NAME=${MODEL_NAME:-default}; READY_TIMEOUT=${READY_TIMEOUT:-1200}
WORKLOAD=${WORKLOAD:-$HERE/../workloads/workload_wiki128.json}
SERVING_CSV=${SERVING_CSV:-$RESULTS/serving.csv}
SYSTEM="vllm_${VARIANT}${ATTN:+_$ATTN}_${DTYPE}"
dry() { [ "${DRY_RUN:-0}" = 1 ]; }

case $VARIANT in
  default_noprefix) FLAGS=(--no-enable-prefix-caching);;
  # same-feature-set row: torch.compile stays on, CUDA graph off -- the self-written engine's own configuration.
  # (--enforce-eager would switch torch.compile off as well: vllm/config/vllm.py sets mode=NONE and cudagraph_mode=NONE.)
  nograph_noprefix) FLAGS=(--no-enable-prefix-caching --compilation-config '{"cudagraph_mode":"NONE"}');;
  *) echo "ABORT: unknown variant '$VARIANT' (vLLM cannot turn continuous batching off: default_noprefix | nograph_noprefix)" >&2; exit 2;;
esac
SERVE=(serve "$HF_DIR" --dtype "$DTYPE" --host 127.0.0.1 --port "$PORT" --served-model-name "$MODEL_NAME"
       --gpu-memory-utilization "$GPU_MEM_UTIL" "${FLAGS[@]}")
[[ -n ${ATTN:-} ]] && SERVE+=(--attention-backend "$ATTN")

read -ra CONC_ARR <<<"$CONC"
NOTES="flags: ${SERVE[*]:2}${NOTES_EXTRA:+; $NOTES_EXTRA}"      # skip "serve <dir>"
CLIENT=("$PY" "$HERE/closed_loop.py" run --backend vllm --system "$SYSTEM" --url "http://127.0.0.1:$PORT"
        --served-model-name "$MODEL_NAME" --workload "$WORKLOAD" --warm-mult 2)
[[ -n ${N_MULT:-} ]] && CLIENT+=(--n-mult "$N_MULT")
[[ -n ${N_MIN:-} ]] && CLIENT+=(--n-min "$N_MIN")
CLIENT+=(--concurrency "${CONC_ARR[@]}" --reps "$REPS" --out "$SERVING_CSV" --notes "$NOTES")
# no usage report to the vLLM project (anonymous, but it is still data leaving the box)
SERVER_ENV=(VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1)

if dry; then
  echo "+ env ${SERVER_ENV[*]} $VLLM ${SERVE[*]}" >&2
  echo "+ ${CLIENT[*]}" >&2
  exit 0
fi

mkdir -p "$RESULTS" "$LOGS"
LOG="$LOGS/${SYSTEM}.log"   # SYSTEM carries variant + dtype (+ backend): runs must not overwrite each other
ENVF="$LOGS/env_${SYSTEM}.txt"
{ echo "# $(date)"; "$PY" -c "import importlib.metadata as m, torch; print('vllm', m.version('vllm')); print('torch', torch.__version__)" 2>/dev/null || true; \
  nvidia-smi -L; echo "# flags: ${SERVE[*]}"; } > "$ENVF"
env "${SERVER_ENV[@]}" "$VLLM" "${SERVE[@]}" >"$LOG" 2>&1 &
SERVER=$!
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$LOGS/${SYSTEM}.gpumem" 2>/dev/null &
SAMPLER=$!

cleanup() {
  kill "$SAMPLER" 2>/dev/null || true
  kill -TERM "$SERVER" 2>/dev/null || true          # the API server stops its EngineCore child itself (timeout 0: it is force-killed, by design)
  # Return only when the GPU memory is really free: the next variant / the next system starts on this card right away.
  local used=999999 i
  for i in $(seq 1 45); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-999999}" -lt 1500 ] && break
    sleep 2
  done
  if [ "${used:-999999}" -ge 1500 ]; then
    echo "WARN: GPU still holds ${used} MiB 90 s after SIGTERM; SIGKILL" >&2
    pkill -KILL -P "$SERVER" 2>/dev/null || true; kill -KILL "$SERVER" 2>/dev/null || true; sleep 5
  fi
  wait "$SERVER" 2>/dev/null || true
  echo "# peak_gpu_mem_MiB $(sort -n "$LOGS/${SYSTEM}.gpumem" | tail -1)" >> "$ENVF"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "waiting for server (log: $LOG)"
ready=0
for ((i = 0; i < READY_TIMEOUT / 5; i++)); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then ready=1; break; fi
  kill -0 "$SERVER" 2>/dev/null || { echo "server died:"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
[ "$ready" = 1 ] || { echo "ABORT: server not healthy after ${READY_TIMEOUT}s"; tail -30 "$LOG"; exit 1; }
# what the server really chose (attention backend, chunked prefill, CUDA graph, prefix caching): the log is the evidence
grep -m1 -i "attention backend" "$LOG" >> "$ENVF" || true
grep -m1 -i "chunked prefill" "$LOG" >> "$ENVF" || true
grep -m3 -i "cuda graph\|enable_prefix_caching\|prefix cach" "$LOG" >> "$ENVF" || true
grep -m1 -o "cudagraph_mode[^,]*" "$LOG" >> "$ENVF" || true      # NONE / FULL_AND_PIECEWISE: what the variant really ran with
grep -m1 -i "torch.compile took" "$LOG" >> "$ENVF" || true       # present = torch.compile was on

"${CLIENT[@]}"
