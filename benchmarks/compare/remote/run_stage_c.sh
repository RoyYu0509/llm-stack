#!/usr/bin/env bash
# Stage C (serving, optimised engine). Decision record: docs/sprints/week7-discussion.md D-7 section 7.11.
# He chose (2026-10-16 night): keep batched prefill + torch.compile, run ONE prefill group per round followed by ONE decode
# step, and make the prefill groups big. Everything is measured as a LADDER on top of the stage-B fa2 arm, same workload, same
# C grid, so each rung's gain (or lack of it) is a number, not a story:
#
#   fa2_base          the stage-B fa2 arm re-run in THIS session (same machine state as the rungs below)
#   fa2_lastlogits    + --prefill-last-logits   (existing flag: the LM head runs on the last prompt position only)
#   fa2_prefillbatch  + --prefill-batch-tokens  (waiting prompts packed into ONE right-padded forward, then one decode step)
#   fa2_compiled      + --compile-decode        (the whole decode forward as one torch.compile'd graph, attention = custom op)
#
#   bash run_stage_c.sh smoke    every component at toy size, output in /tmp. RUN FIRST (stage B died three times on things a
#                                1-minute smoke would have caught)
#   bash run_stage_c.sh check    correctness gate (check_stage_c.py, real weights) + one decode step profiled: base vs compiled
#   bash run_stage_c.sh ladder   the arms above x C in {1, 8, 32, 64}, 3 repetitions
#   bash run_stage_c.sh sglang   SGLang 0.5.10 cb_only + default on THIS machine (needs setup_sglang.sh first), into serving_sglang_samemachine.csv:
#                                stage B's SGLang numbers come from another machine, so the ratios against them carry +-15-20%
#   bash run_stage_c.sh drift    stage-B naive arm at C=32 again: how far has this session drifted from stage B?
#   DRY_RUN=1 bash run_stage_c.sh all     print every command instead of running it (tested on the laptop)
#
# Same rules as stage B: bf16, mixed output 32-96, requests = max(8 x C, 32), warm-up max(2 x C, 8), random-init weights
# (speed only), results appended to results/final/serving.csv. Nothing here starts, stops or rents an instance.
set -u
ROOT=${ROOT:-/workspace/llm-stack}; CMP=$ROOT/benchmarks/compare; SERV=$ROOT/serving
FINAL=$CMP/results/final; LOGS=$FINAL/logs/serving; RAW=$FINAL/serving_raw; CSV=$FINAL/serving.csv
W=$CMP/workloads/workload_mixed4096.json; W128=$CMP/workloads/workload_wiki128.json
PY=${PY:-/workspace/venv-fa/bin/python}; CKPT=${CKPT:-/workspace/ckpt/model_only.pt}; CS=${CS:-"1 8 32 64"}; REPS=${REPS:-3}
SGLV=${SGLV:-/workspace/venv-sglang}; HFDIR=${HFDIR:-/workspace/llm190m-hf-random}; SGCSV=$FINAL/serving_sglang_samemachine.csv
PBT=${PBT:-16384}     # padded tokens per prefill forward: as big as the 24 GB card takes (16 prompts x 900 tokens ~ 14k)
export PYTHONPATH=$ROOT/training/cs336-basics:$ROOT/training
ST=${ST:-/workspace/STAGE_C_STATUS.txt}
RI="random-init weights; speed-only; RTX 3090 Ti (450 W, driver 580; NOT the 300 W / driver 550 machine of stage B); bf16"
ARMS="fa2_base fa2_lastlogits fa2_prefillbatch fa2_compiled"
exec 3>&2
x() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ $*" >&3; else "$@"; fi; }
st() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "# $*" >&3; else echo "$(date -Is) $*" >> "$ST"; fi; }
mkdir -p "$LOGS" "$RAW/engine" 2>/dev/null || true
[ "${DRY_RUN:-0}" = 1 ] && mkdir -p "$SERV" 2>/dev/null
mf() { x env PYTHONPATH=$CMP:$PYTHONPATH $PY -c "import manifest; manifest.record('serving_c_$1', out_dir='$FINAL', argv=['run_stage_c.sh','$1'], extra=dict(csv='serving.csv'))"; }
nreq() { local n=$((8 * $1)); [ $n -lt 32 ] && n=32; echo $n; }
nwarm() { local n=$((2 * $1)); [ $n -lt 8 ] && n=8; echo $n; }

arm_flags() {   # the cumulative switches of each rung (NOT --attn / --dtype: those are the same for every rung)
  case $1 in
    fa2_base)         echo "";;
    fa2_lastlogits)   echo "--prefill-last-logits";;
    fa2_prefillbatch) echo "--prefill-last-logits --prefill-batch-tokens $PBT";;
    fa2_compiled)     echo "--prefill-last-logits --prefill-batch-tokens $PBT --compile-decode";;
  esac
}
arm_desc() {
  case $1 in
    fa2_base)         echo "rung 0 = stage-B fa2 arm (preallocated KV + flash_attn_with_kvcache + one batched argmax), re-run in this session";;
    fa2_lastlogits)   echo "rung 1 = rung 0 + prefill computes the LM head for the last prompt position only (existing flag)";;
    fa2_prefillbatch) echo "rung 2 = rung 1 + waiting prompts packed into one right-padded prefill forward (<= $PBT padded tokens), one prefill group per round followed by one decode step";;
    fa2_compiled)     echo "rung 3 = rung 2 + the decode forward as one torch.compile'd graph, attention through a registered custom op";;
  esac
}

ingest_engine() { # arm name notes
  local arm=$1 name=$2 notes=$3 recs="" f
  for f in $RAW/engine/${name}_r*.json; do [ -f "$f" ] && grep -q '"status": "ok"' "$f" && recs="$recs $f"; done
  if [ "${DRY_RUN:-0}" = 1 ] || [ -n "$recs" ]; then
    x $PY $CMP/serving/closed_loop.py ingest --system engine_cb_${arm}_bfloat16 --recordings ${recs:-"<recordings>"} --out $CSV --notes "$notes" > $LOGS/ingest_$name.log 2>&1
  else st "no ok recording for $name (OOM / failed): not in the table"; fi
}

phase_check() {
  st "check"; mf check
  local prof=$FINAL/serving_profiles; mkdir -p "$prof" 2>/dev/null || true
  # 1. the correctness gate (thresholds are fixed inside the script, before any number is seen)
  ( cd $SERV && x $PY benchmarks/check_stage_c.py --workload $W128 --checkpoint $CKPT --dtype bfloat16 --rows 8 --steps 16 ) > $LOGS/check_stage_c.log 2>&1
  st "check_stage_c rc=$?"
  # 2. one decode step, B=32: launches, GPU busy, blocking CUDA calls, eager vs compiled
  ( cd $SERV && x $PY benchmarks/profile_decode_step.py --workload $W --batch 32 --attn fa2 --dtype bfloat16 ) > $prof/decode_step_fa2_base.txt 2>&1
  ( cd $SERV && x $PY benchmarks/profile_decode_step.py --workload $W --batch 32 --attn fa2 --dtype bfloat16 --compile-decode ) > $prof/decode_step_fa2_compiled.txt 2>&1
  st "STAGE_C_check_DONE"
}

phase_ladder() {
  st "ladder"; mf ladder
  local arm C name flags
  for arm in $ARMS; do
    flags=$(arm_flags $arm)
    for C in $CS; do
      name=${arm}_c${C}
      st "ladder $name"
      ( cd $SERV && x timeout 2400 $PY benchmarks/record_demo.py --preset 190m --prompts-json $W --device cuda --attn fa2 --dtype bfloat16 \
          --mode cb --concurrency $C --n-requests $(nreq $C) --reps $REPS --warmup-requests $(nwarm $C) $flags --arm-name $name --out $RAW/engine/$name.json ) \
          > $LOGS/engine_$name.log 2>&1
      st "ladder $name rc=$?"
      ingest_engine $arm $name "$(arm_desc $arm); mixed output lengths 32-96; n_requests=$(nreq $C); warm-up $(nwarm $C); $RI"
    done
  done
  st "STAGE_C_ladder_DONE"
}

phase_drift() {
  st "drift"; mf drift
  local name=naive_drift_c32
  ( cd $SERV && x timeout 1800 $PY benchmarks/record_demo.py --preset 190m --prompts-json $W --device cuda --attn naive --dtype bfloat16 \
      --mode cb --concurrency 32 --n-requests $(nreq 32) --reps $REPS --warmup-requests $(nwarm 32) --arm-name $name --out $RAW/engine/$name.json ) \
      > $LOGS/engine_$name.log 2>&1
  st "drift rc=$?"
  ingest_engine naive_drift $name "stage-B naive arm re-run in the stage-C session, to see how far the machine state drifted; $RI"
  st "STAGE_C_drift_DONE"
}

phase_sglang() {   # same machine as the ladder: the ratio to SGLang is only quotable when both ran on the same box
  st "sglang"; mf sglang
  [ -d "$HFDIR" ] || x $PY $CMP/convert_hf/convert_190m_to_hf.py --random-init --out "$HFDIR" --tokenizer gpt2 > $LOGS/convert_random_init_stage_c.log 2>&1
  local spec variant attn
  for spec in "cb_only flashinfer MEM_FRACTION=0.8" "default_noradix flashinfer CUDA_GRAPH_MAX_BS=64"; do
    set -- $spec; variant=$1; attn=$2; shift 2
    st "sglang $variant $attn"
    x timeout 3600 env PY=$SGLV/bin/python HF_DIR=$HFDIR ATTN=$attn DTYPE=bfloat16 WORKLOAD=$W CONC="$CS" REPS=$REPS N_MULT=8 N_MIN=32 SERVING_CSV=$SGCSV \
        NOTES_EXTRA="SAME MACHINE as the stage-C ladder; $RI; mixed output lengths 32-96; requests = max(8 x C, 32); untimed warm-up 2 x C" "$@" \
        bash $CMP/serving/run_sglang.sh $variant > $LOGS/sglang_samemachine_${variant}_${attn}.log 2>&1
    st "sglang $variant $attn rc=$?"
  done
  st "STAGE_C_sglang_DONE"
}

phase_smoke() {   # toy sizes; every line below must PASS before the real phases are started
  local S=${SMOKE:-/tmp/smoke_stage_c} fail=0 arm flags
  rm -rf "$S"; mkdir -p "$S"
  chk() { local name=$1; shift
    if x "$@" > "$S/$name.log" 2>&1; then echo "SMOKE PASS $name"; else echo "SMOKE FAIL $name  (see $S/$name.log: $(tail -1 "$S/$name.log" | cut -c1-150))"; fail=1; fi; }
  for arm in $ARMS; do
    flags=$(arm_flags $arm)
    ( cd $SERV && chk engine_$arm $PY benchmarks/record_demo.py --preset 190m --prompts-json $W --device cuda --attn fa2 --dtype bfloat16 --mode cb \
        --concurrency 4 --n-requests 8 --reps 2 --warmup-requests 4 $flags --arm-name smoke_$arm --out $S/smoke_$arm.json )
    chk ingest_$arm $PY $CMP/serving/closed_loop.py ingest --system smoke_$arm --recordings $S/smoke_${arm}_r1.json $S/smoke_${arm}_r2.json --out $S/serving.csv
  done
  ( cd $SERV && chk naive_drift $PY benchmarks/record_demo.py --preset 190m --prompts-json $W --device cuda --attn naive --dtype bfloat16 --mode cb \
      --concurrency 4 --n-requests 8 --reps 1 --warmup-requests 4 --arm-name smoke_naive --out $S/smoke_naive.json )
  ( cd $SERV && chk check_stage_c $PY benchmarks/check_stage_c.py --workload $W128 --checkpoint $CKPT --dtype bfloat16 --rows 4 --steps 4 )
  ( cd $SERV && chk decode_step_base $PY benchmarks/profile_decode_step.py --workload $W --batch 8 --attn fa2 --dtype bfloat16 --steps 3 )
  ( cd $SERV && chk decode_step_compiled $PY benchmarks/profile_decode_step.py --workload $W --batch 8 --attn fa2 --dtype bfloat16 --steps 3 --compile-decode )
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  if [ $fail = 0 ]; then st "STAGE_C_smoke_DONE (all components passed)"; else st "STAGE_C_smoke_FAILED"; return 1; fi
}

case "${1:-}" in
  smoke) phase_smoke;;
  check) phase_check;; ladder) phase_ladder;; drift) phase_drift;; sglang) phase_sglang;;
  all) phase_check; phase_ladder; phase_drift;;
  *) echo "usage: $0 smoke|check|ladder|drift|sglang|all"; exit 2;;
esac
