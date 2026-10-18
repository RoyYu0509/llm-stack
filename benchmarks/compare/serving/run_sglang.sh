#!/usr/bin/env bash
# I1-d / I1-e -- SGLang under the same closed-loop workload as the self-written engine.
#
#   bash serving/run_sglang.sh cb_only   # "same feature set": everything switchable is switched off
#   bash serving/run_sglang.sh cb_off    # cb_only + --max-running-requests 1: SGLang's own "seq" arm
#   bash serving/run_sglang.sh default   # all optimisations on (the industrial ceiling)
#   bash serving/run_sglang.sh default_noradix   # default minus radix cache (repeated prompts would hit it)
#   bash serving/run_sglang.sh hf        # HuggingFace generate(), one request at a time
#   bash serving/run_sglang.sh hf_cb     # HuggingFace transformers' own continuous batching (flash_attention_2 by default)
#
# Needs: the HF-format 190M model dir (benchmarks/compare/convert_hf/convert_190m_to_hf.py --tokenizer
# gpt2), `pip install sglang` on the 3090 box, and workloads/workload_wiki128.json
# (python serving/closed_loop.py make-workload --recording <demo_c32_cb_r1.json>).
#
# SGLang's scheduler IS continuous batching; it cannot be turned off. "cb_only" therefore means
# "off: radix cache, overlap scheduler, chunked prefill, CUDA graph". What stays on (token-level KV
# pool, fused kernels, process split) is part of the gap -- README says so. Flag names drift between
# versions: this script greps `--help` and refuses to guess.
#
# Env: HF_DIR (required) DTYPE(float32|bfloat16, default float32) ATTN(default triton) PORT(30000)
#      CONC("8 16 32") REPS(3) HF_ATTN(sdpa|eager, hf variant only)
#      SERVING_CSV (output csv, default results/final/serving.csv), N_MULT (requests per level = N_MULT x C), N_MIN (floor)
#      WORKLOAD (json with 'prompts'; default workloads/workload_wiki128.json)
#      CUDA_GRAPH_MAX_BS (optional --cuda-graph-max-bs-decode; label suffix _gb<N>)
#      MEM_FRACTION (optional --mem-fraction-static override)
#      NOTES_EXTRA (appended to every row's notes, e.g. "random-init weights; speed-only")
#      OPEN LOOP instead of CONC: ARRIVAL_RATES ("10 20 40" req/s) [ARRIVAL_CV=1 (Poisson)] [DURATION=30 s]
#        [MAX_INFLIGHT=0 (no client cap)] [SLO_TTFT_S=2] [SLO_TPOT_MS=100]; rows go to experiment I2, concurrency 0
#      hf_cb only: CB_OPTS (JSON for ContinuousBatchingConfig, e.g. '{"use_cuda_graph": true}'), CB_TAG (suffix for the system name)
set -euo pipefail
VARIANT=${1:?variant: cb_only | cb_off | default | default_noradix | hf | hf_cb}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$HERE/../results/final"; LOGS="$RESULTS/logs/serving"; mkdir -p "$RESULTS" "$LOGS"
HF_DIR=${HF_DIR:?set HF_DIR to the converted HF model directory}
DTYPE=${DTYPE:-float32}; ATTN=${ATTN:-triton}; PORT=${PORT:-30000}
CONC=${CONC:-"8 16 32"}; REPS=${REPS:-3}
PY=${PY:-python}
ATTN_SUFFIX=""; [[ $ATTN == triton ]] || ATTN_SUFFIX="_${ATTN}"   # legacy names stay valid for triton
SYSTEM="sglang_${VARIANT}${ATTN_SUFFIX}_${DTYPE}"
if [[ -n ${ARRIVAL_RATES:-} ]]; then   # open loop: requests arrive on a clock; replaces the closed-loop concurrency grid
  LOAD=(--arrival-rate $ARRIVAL_RATES --arrival-cv "${ARRIVAL_CV:-1}" --duration "${DURATION:-30}" --max-inflight "${MAX_INFLIGHT:-0}"
        --slo-ttft-s "${SLO_TTFT_S:-2}" --slo-tpot-ms "${SLO_TPOT_MS:-100}")
else
  LOAD=(--concurrency $CONC)
fi
WORKLOAD=${WORKLOAD:-$HERE/../workloads/workload_wiki128.json}
[[ -n ${CUDA_GRAPH_MAX_BS:-} ]] && SYSTEM="${SYSTEM}_gb${CUDA_GRAPH_MAX_BS}"

if [[ $VARIANT == hf_cb ]]; then
  HFA=${HF_ATTN:-flash_attention_2}
  $PY "$HERE/closed_loop.py" run --backend hf_cb --system "hf_cb_${HFA}_${DTYPE}${CB_TAG:+_$CB_TAG}" --model-dir "$HF_DIR" \
      --dtype "$DTYPE" --attn "$HFA" --workload "$WORKLOAD" --warm-mult 2 ${N_MULT:+--n-mult $N_MULT} ${N_MIN:+--n-min $N_MIN} \
      "${LOAD[@]}" --reps "$REPS" ${CB_OPTS:+--cb-opts "$CB_OPTS"} --out "${SERVING_CSV:-$RESULTS/serving.csv}" \
      --notes "transformers continuous batching (ContinuousBatchingManager); attn=$HFA; cb_opts=${CB_OPTS:-defaults}; eos disabled${NOTES_EXTRA:+; $NOTES_EXTRA}"
  exit 0
fi

if [[ $VARIANT == hf ]]; then
  $PY "$HERE/closed_loop.py" run --backend hf --system "hf_generate_${DTYPE}" --model-dir "$HF_DIR" \
      --dtype "$DTYPE" --attn "${HF_ATTN:-sdpa}" --workload "$WORKLOAD" --warm-mult 2 ${N_MULT:+--n-mult $N_MULT} ${N_MIN:+--n-min $N_MIN} \
      "${LOAD[@]}" --reps "$REPS" --out "${SERVING_CSV:-$RESULTS/serving.csv}" \
      --notes "generate() one request at a time; attn=${HF_ATTN:-sdpa}; min_new_tokens=new_tokens; TF32 matmul${NOTES_EXTRA:+; $NOTES_EXTRA}"
  exit 0
fi

HELP=$($PY -m sglang.launch_server --help 2>&1 || true)
need() { grep -q -- "$1" <<<"$HELP" || { echo "ABORT: this SGLang has no flag $1 -- check 'launch_server --help' and update this script"; exit 1; }; }

FLAGS=(--model-path "$HF_DIR" --dtype "$DTYPE" --attention-backend "$ATTN" --port "$PORT")
if [[ $VARIANT == cb_only || $VARIANT == cb_off ]]; then
  need --disable-radix-cache;   FLAGS+=(--disable-radix-cache)
  need --disable-overlap-schedule; FLAGS+=(--disable-overlap-schedule)
  need --chunked-prefill-size;  FLAGS+=(--chunked-prefill-size -1)   # -1 disables chunked prefill
  if   grep -q -- "--cuda-graph-backend-decode" <<<"$HELP"; then
    FLAGS+=(--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
  elif grep -q -- "--disable-cuda-graph" <<<"$HELP"; then
    FLAGS+=(--disable-cuda-graph)
  else echo "ABORT: cannot find a CUDA-graph off switch in --help"; exit 1; fi
fi
if [[ $VARIANT == cb_off ]]; then
  need --max-running-requests; FLAGS+=(--max-running-requests 1)       # one request at a time
fi
if [[ $VARIANT == default_noradix ]]; then   # default, but without prefix reuse: the workload repeats the same prompts
  need --disable-radix-cache; FLAGS+=(--disable-radix-cache)
fi
# With CUDA graph + chunked prefill switched off SGLang 0.5.21 auto-derived mem-fraction-static = -0.026 on a
# 24 GB card ("Loaded weights leave no GPU memory for the KV cache"); MEM_FRACTION overrides the auto value.
# SGLang captures decode CUDA graphs only up to an auto-chosen max batch size (48 on this 24 GB card). Above it a
# decode batch silently runs without a graph, which looks like the server "collapsing" at C > 48.
if [[ -n ${CUDA_GRAPH_MAX_BS:-} ]]; then   # the flag was renamed between releases: 0.5.2x has -decode, 0.5.10 has the plain one
  if   grep -q -- "--cuda-graph-max-bs-decode" <<<"$HELP"; then FLAGS+=(--cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS")
  elif grep -q -- "--cuda-graph-max-bs" <<<"$HELP";        then FLAGS+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
  else echo "ABORT: no CUDA-graph max batch size flag in --help"; exit 1; fi
fi
if [[ -n ${MEM_FRACTION:-} ]]; then
  need --mem-fraction-static; FLAGS+=(--mem-fraction-static "$MEM_FRACTION")
fi

LOG="$LOGS/${SYSTEM}.log"   # SYSTEM carries variant + backend + dtype (+ graph bs): runs must not overwrite each other
# version via the interpreter itself: `pip show | grep` aborts the whole script under `set -e -o pipefail`
# in a uv venv (no pip module), which silently killed every variant before the server was launched.
{ echo "# $(date)"; $PY -c "import sglang, torch; print('sglang', sglang.__version__); print('torch', torch.__version__)" 2>/dev/null || true; \
  nvidia-smi -L; echo "# flags: ${FLAGS[*]}"; } > "$LOGS/env_${SYSTEM}.txt"
$PY -m sglang.launch_server "${FLAGS[@]}" >"$LOG" 2>&1 &
SERVER=$!
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$LOGS/${SYSTEM}.gpumem" 2>/dev/null &
SAMPLER=$!
trap 'kill $SERVER $SAMPLER 2>/dev/null || true; wait $SERVER 2>/dev/null || true; echo "# peak_gpu_mem_MiB $(sort -n "$LOGS/${SYSTEM}.gpumem" | tail -1)" >> "$LOGS/env_${SYSTEM}.txt"' EXIT

echo "waiting for server (log: $LOG)"
for i in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server never became healthy"; exit 1; }
grep -m1 -i "attention.*backend" "$LOG" >> "$LOGS/env_${SYSTEM}.txt" || true   # which backend it really used

$PY "$HERE/closed_loop.py" run --backend sglang --system "$SYSTEM" --url "http://127.0.0.1:$PORT" \
    --workload "$WORKLOAD" --warm-mult 2 ${N_MULT:+--n-mult $N_MULT} ${N_MIN:+--n-min $N_MIN} \
    "${LOAD[@]}" --reps "$REPS" --out "${SERVING_CSV:-$RESULTS/serving.csv}" \
    --notes "flags: ${FLAGS[*]:2}${NOTES_EXTRA:+; $NOTES_EXTRA}"
