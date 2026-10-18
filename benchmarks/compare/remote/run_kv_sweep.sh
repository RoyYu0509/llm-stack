#!/usr/bin/env bash
# KV-cache size sweep -- "does the KV cache matter on a bigger model and a longer context?" (D-7 section 7.13-I; his request
# 2026-10-17: a much bigger model, long context, one request, ideally filling the 24 GB card). Nothing here starts, stops or rents
# an instance. The engine is not touched: only the model SHAPE (random-init presets, record_demo.py --preset) and the prompt length change.
#
# One request at a time (C = 1), the engine's single-request path, so the ONLY difference between the two arms is the KV cache:
#   nokv   adapter.use_kv_cache = False: every generated token re-runs the whole prefix through the model
#   kv     the model's own forward_with_cache: the new token's K/V is appended (torch.cat) to the cached K/V of the prefix; its prefill
#          computes the LM head for the last position only (--prefill-last-logits; see arm_flags for why)
# Both arms: same random weights, same synthetic prompt of exactly L tokens, same $NEW new tokens, attention = torch SDPA
# (FlashAttention-2 kernel for the unmasked causal passes; naive attention would materialise L x L scores and run out of memory at 16k).
#
#   sizes   190m (the model of the ladder)  1.4b  2.8b  6.9b      (MHA, head_dim 128; 6.9b = the shape of a 7B model)
#   L       1024  4096  16384  prompt tokens; 6.9b: 1024  4096  12288
#   6.9b @ 16384 does NOT fit (smoke 2026-10-17: bf16 weights 12.8 GiB + KV cache 8.0 GiB + prefill activations -> OOM at 22 GiB of 23.55).
#   6.9b @ 12288 is the cell that fills the card (weights 12.8 + KV 6.0 GiB; the peak is recorded); if even that does not fit the sweep
#   falls back to 6.9b @ 8192 and says so in the status file.
#
#   bash run_kv_sweep.sh smoke     two tiny cells + the 6.9b @ 12288 memory probe, output in /tmp. RUN FIRST
#   bash run_kv_sweep.sh sweep | wrapup | all
#   DRY_RUN=1 bash run_kv_sweep.sh all    print every command instead of running it (tested on the laptop)
# Results: results/final/serving_kv_sweep.csv (experiment I1, concurrency 1), raw recordings in serving_raw/engine_kv/.
set -u
ROOT=${ROOT:-/workspace/llm-stack}; CMP=$ROOT/benchmarks/compare; SERV=$ROOT/serving
FINAL=$CMP/results/final; LOGS=$FINAL/logs/serving; RAW=$FINAL/serving_raw/engine_kv; CSV=$FINAL/serving_kv_sweep.csv
PY=${PY:-/workspace/venv-fa/bin/python}
SIZES=${SIZES:-"190m 1.4b 2.8b 6.9b"}; LENS=${LENS:-"1024 4096 16384"}; LENS_69=${LENS_69:-"1024 4096 12288"}; NEW=${NEW:-8}
lens_for() { if [ "$1" = 6.9b ]; then echo $LENS_69; else echo $LENS; fi; }   # LENS applies to every size except 6.9b
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # both arms: less allocator fragmentation when a cell fills the card
export PYTHONPATH=$ROOT/training/cs336-basics:$ROOT/training
export BENCH_GIT_DIRTY=${BENCH_GIT_DIRTY:-True}   # the box holds an rsynced, uncommitted tree
ST=${ST:-/workspace/KV_SWEEP_STATUS.txt}; NVSMI=${NVSMI:-nvidia-smi}
RI="random-init weights; speed-only; RTX 3090 Ti (450 W, driver 580); bf16; one request at a time; synthetic prompt of exactly L tokens; torch SDPA attention; one machine, one session"
exec 3>&2
x() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ $*" >&3; else "$@"; fi; }
st() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "# $*" >&3; else echo "$(date -Is) $*" >> "$ST"; fi; }
mkdir -p "$LOGS" "$RAW" 2>/dev/null || true
[ "${DRY_RUN:-0}" = 1 ] && mkdir -p "$SERV" 2>/dev/null
mf() { x env PYTHONPATH=$CMP:$PYTHONPATH $PY -c "import manifest; manifest.record('kv_sweep_$1', out_dir='$FINAL', argv=['run_kv_sweep.sh','$1'], extra=dict(csv='serving_kv_sweep.csv'))"; }

need_idle() {   # a job from another session on this GPU would corrupt every timing: refuse to start
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "# need_idle: GPU memory.used <= ${IDLE_MIB:-1500} MiB" >&3; return 0; fi
  local used waited=0 poll=${IDLE_POLL_S:-5}
  while :; do
    used=$($NVSMI --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -le "${IDLE_MIB:-1500}" ]; then return 0; fi
    [ $waited -ge "${IDLE_WAIT_S:-240}" ] && break
    sleep "$poll"; waited=$((waited + poll))
  done
  echo "ABORT: the GPU is not idle (memory.used=${used:-unknown} MiB after ${waited}s): another job is running; nothing was started" >&2
  st "ABORT gpu busy (${used:-unknown} MiB)"; exit 3
}

# The no-KV arm of a big cell costs one full forward over the prefix PER TOKEN (6.9b @ 12288: seconds per token), so it gets fewer
# requests; every other cell: 4 requests x 3 repetitions after 2 untimed warm-up requests.
cell_n()    { if [ "$1" = nokv ] && { [ "$2" = 6.9b ] || [ "$2" = 2.8b ]; } && [ "$3" -ge 12288 ]; then echo 2; else echo 4; fi; }
cell_reps() { if [ "$1" = nokv ] && { [ "$2" = 6.9b ] || [ "$2" = 2.8b ]; } && [ "$3" -ge 12288 ]; then echo 2; else echo 3; fi; }
cell_warm() { if [ "$1" = nokv ] && { [ "$2" = 6.9b ] || [ "$2" = 2.8b ]; } && [ "$3" -ge 12288 ]; then echo 1; else echo 2; fi; }
# kv arm = the KV cache AND last-position-only logits at prefill (without it the 6.9b @ 12288 prefill ran out of memory computing the
# LM head for all 12288 positions: smoke 2026-10-17). It touches prefill only; the decode step the table is about is identical.
# --fast-rope-init (both arms): same RoPE table, built in a few tensor ops instead of minutes of Python for a long context.
arm_flags() { if [ "$1" = nokv ]; then echo "--no-kv-cache --fast-rope-init"; else echo "--prefill-last-logits --fast-rope-init"; fi; }

ok_recs() {   # recordings of a cell that finished with status ok
  local f recs=""
  for f in $RAW/$1_r*.json; do [ -f "$f" ] && grep -q '"status": "ok"' "$f" && recs="$recs $f"; done
  echo $recs
}

run_cell() {   # arm size L [out_dir] -> 0 if at least one repetition was recorded
  local arm=$1 size=$2 L=$3 name n reps w recs
  name=${arm}_${size}_L${L}; n=$(cell_n $arm $size $L); reps=$(cell_reps $arm $size $L); w=$(cell_warm $arm $size $L)
  st "kv_sweep $name"
  ( cd $SERV && x timeout 3000 $PY benchmarks/record_demo.py --preset $size --context-length $((L + NEW + 8)) --synthetic --prompt-len $L $L \
      --device cuda --dtype bfloat16 --attn sdpa --mode seq $(arm_flags $arm) --concurrency 1 --new-tokens $NEW \
      --n-requests $n --reps $reps --warmup-requests $w --arm-name $name --out $RAW/$name.json ) > $LOGS/kv_$name.log 2>&1
  st "kv_sweep $name rc=$?"
  recs=$(ok_recs $name)
  if [ "${DRY_RUN:-0}" = 1 ] || [ -n "$recs" ]; then
    x $PY $CMP/serving/closed_loop.py ingest --system engine_kvsweep_${arm}_${size}_L${L}_bfloat16 --recordings ${recs:-"<recordings>"} --out $CSV \
        --notes "KV-cache sweep: $arm, model $size, prompt $L tokens, $NEW new tokens, requests=$n x reps=$reps, warm-up $w; $RI" > $LOGS/kv_ingest_$name.log 2>&1
    return 0
  fi
  st "no ok recording for $name (OOM / failed): not in the table"; return 1
}

phase_sweep() {
  need_idle; st "sweep"; mf sweep
  local size L arm
  for size in $SIZES; do
    for L in $(lens_for $size); do
      for arm in nokv kv; do run_cell $arm $size $L; done
    done
  done
  # the card-filling cell did not fit: fall back one notch so the biggest model still has a long-context row
  if [ "${DRY_RUN:-0}" != 1 ] && [ -z "$(ok_recs kv_6.9b_L12288)" ] && echo " $SIZES " | grep -q " 6.9b "; then
    st "6.9b @ 12288 did not fit in 24 GB: falling back to 8192"
    run_cell nokv 6.9b 8192; run_cell kv 6.9b 8192
  fi
  st "STAGE_KV_sweep_DONE"
}

phase_wrapup() {
  st "wrapup"
  x $PY $CMP/remote/kv_sweep_summary.py > $FINAL/serving_kv_sweep_summary.txt 2>&1
  x tar czf $ROOT/kv_sweep_results.tgz -C $FINAL serving_kv_sweep.csv serving_kv_sweep_summary.txt serving_raw/engine_kv logs/serving manifest.json
  st "KV_SWEEP_ALL_DONE (summary: $FINAL/serving_kv_sweep_summary.txt; tarball: $ROOT/kv_sweep_results.tgz)"
}

phase_smoke() {   # tiny cells prove the plumbing; the 6.9b probe proves the one cell that can run out of memory
  need_idle
  local S=${SMOKE_DIR:-/tmp/smoke_kv_sweep} arm out
  mkdir -p "$S" 2>/dev/null
  chk() {
    local name=$1; shift
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ $*" >&3; return 0; fi
    if "$@" > "$S/$name.log" 2>&1; then echo "PASS $name"; else echo "FAIL $name (see $S/$name.log)"; fi
  }
  for arm in nokv kv; do
    ( cd $SERV && chk record_small_$arm $PY benchmarks/record_demo.py --preset small --context-length 300 --synthetic --prompt-len 256 256 \
        --device cuda --dtype bfloat16 --attn sdpa --mode seq $(arm_flags $arm) --concurrency 1 --new-tokens 4 --n-requests 2 --reps 2 \
        --warmup-requests 1 --arm-name smoke_$arm --out $S/smoke_$arm.json )
    chk ingest_small_$arm $PY $CMP/serving/closed_loop.py ingest --system smoke_$arm --recordings $S/smoke_${arm}_r1.json $S/smoke_${arm}_r2.json --out $S/kv.csv
  done
  # the card-filling cell, one request, two new tokens: does it fit, and what is the peak?
  ( cd $SERV && chk probe_6.9b_L12288_kv $PY benchmarks/record_demo.py --preset 6.9b --context-length 12304 --synthetic --prompt-len 12288 12288 \
      --device cuda --dtype bfloat16 --attn sdpa --mode seq $(arm_flags kv) --concurrency 1 --new-tokens 2 --n-requests 1 --reps 1 --warmup-requests 0 \
      --arm-name smoke_probe_kv --out $S/probe_kv.json )
  grep -h "peak" "$S/probe_6.9b_L12288_kv.log" | tail -2
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  echo "smoke finished: every line above must say PASS; the probe's peak must be below the card's 24 GiB"
}

case "${1:-}" in
  smoke) phase_smoke;;
  sweep) phase_sweep;;
  wrapup) phase_wrapup;;
  all) phase_sweep; phase_wrapup;;
  *) echo "usage: $0 smoke|sweep|wrapup|all"; exit 2;;
esac
