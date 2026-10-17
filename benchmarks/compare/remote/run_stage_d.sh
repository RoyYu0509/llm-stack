#!/usr/bin/env bash
# Stage D -- the complete serving benchmark, one machine, one session. Decision record: docs/sprints/week7-discussion.md
# D-7 section 7.13 (E = open-loop parameters, F = the grid below). Nothing here starts, stops or rents an instance.
#
# Story A: the engine's optimisation LADDER (cumulative; every rung adds one thing to the one before)
#   seq_nokv               no KV cache, one request at a time               C = 1
#   seq_kv                 + KV cache                                       C = 1, 32 (C=32 with 64 requests: serial is slow)
#   cb_naive               + continuous batching (naive attention, ragged KV)   C = 1 8 32 64
#   cb_fa2                 + pre-allocated KV + flash-attn decode + batched argmax (one bundle)
#   cb_fa2_prefill         + batched prefill (<= $PBT padded tokens, one group per round) + last-position logits
#   cb_fa2_prefill_compile + torch.compile'd decode  (= the best configuration)
# Story B: the best configuration against the others, SAME-FEATURE-SET rows first, ceiling rows as reference
#   sglang   cb_only (flashinfer, no radix / overlap / chunked prefill / CUDA graph)  |  default_noradix (CUDA graph, bs <= 64)
#   vllm     nograph_noprefix (torch.compile on, CUDA graph off)                      |  default_noprefix (CUDA graph on)
#   (HF is NOT in the throughput comparison -- his decision, 2026-10-16: vLLM and SGLang are the mainstream engines. HF stays
#   as the correctness reference of the c5 phase. The hf / hf_cb harness code is still in closed_loop.py, unused here.)
# Open loop: engine best config + SGLang cb_only + SGLang default, Poisson arrivals at 10 / 50 / 75 req/s, in-flight cap 64
# Correctness: best configuration vs HF generate, bf16, real weights, 64 prompts (report only; FlashAttention has no fp32 kernel)
#
#   bash run_stage_d.sh smoke      EVERY component at toy size, output in /tmp. RUN FIRST, and read every line
#   bash run_stage_d.sh ladder | sglang | vllm | openloop | c5
#   bash run_stage_d.sh all        ladder sglang vllm openloop c5, one after the other (never in parallel)
#   DRY_RUN=1 bash run_stage_d.sh all    print every command instead of running it (tested on the laptop)
#
# Rules: bf16, random-init weights (speed only), workload_mixed4096_cont.json (output lengths: any integer 32-96),
# requests = max(8 x C, 32), untimed warm-up max(2 x C, 8), 3 repetitions, C in {1 8 32 64}. All results go to ONE csv,
# results/final/serving_stage_d.csv (closed loop = experiment I1, open loop = I2). Every phase first checks that the GPU
# is idle: a job from another session on the same card would corrupt every number. Set BENCH_GIT_SHA for the csv rows.
set -u
ROOT=${ROOT:-/workspace/llm-stack}; CMP=$ROOT/benchmarks/compare; SERV=$ROOT/serving
FINAL=$CMP/results/final; LOGS=$FINAL/logs/serving; RAW=$FINAL/serving_raw/engine_d; CSV=$FINAL/serving_stage_d.csv
WC=$CMP/workloads/workload_mixed4096_cont.json; W128=$CMP/workloads/workload_wiki128.json
PY=${PY:-/workspace/venv-fa/bin/python}; CKPT=${CKPT:-/workspace/ckpt/model_only.pt}; CS=${CS:-"1 8 32 64"}; REPS=${REPS:-3}
SGLV=${SGLV:-/workspace/venv-sglang}; VLLV=${VLLV:-/workspace/venv-vllm}; HFDIR=${HFDIR:-/workspace/llm190m-hf-random}
PBT=${PBT:-16384}; RATES=${RATES:-"10 50 75"}; CAP=${CAP:-64}; DURATION=${DURATION:-30}
SLO_TTFT_S=${SLO_TTFT_S:-0.5}; SLO_TPOT_MS=${SLO_TPOT_MS:-50}
export PYTHONPATH=$ROOT/training/cs336-basics:$ROOT/training
export BENCH_GIT_DIRTY=${BENCH_GIT_DIRTY:-True}   # the box holds an rsynced, uncommitted tree
ST=${ST:-/workspace/STAGE_D_STATUS.txt}; NVSMI=${NVSMI:-nvidia-smi}
RI="random-init weights; speed-only; RTX 3090 Ti (450 W, driver 580); bf16; mixed output lengths 32-96 (any integer); one machine, one session"
exec 3>&2
x() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ $*" >&3; else "$@"; fi; }
st() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "# $*" >&3; else echo "$(date -Is) $*" >> "$ST"; fi; }
mkdir -p "$LOGS" "$RAW" 2>/dev/null || true
[ "${DRY_RUN:-0}" = 1 ] && mkdir -p "$SERV" 2>/dev/null
mf() { x env PYTHONPATH=$CMP:$PYTHONPATH $PY -c "import manifest; manifest.record('serving_d_$1', out_dir='$FINAL', argv=['run_stage_d.sh','$1'], extra=dict(csv='serving_stage_d.csv'))"; }
nreq() { local n=$((8 * $1)); [ $n -lt 32 ] && n=32; echo $n; }
nwarm() { local n=$((2 * $1)); [ $n -lt 8 ] && n=8; echo $n; }

need_idle() {   # a job from another session on this GPU would corrupt every timing: refuse to start
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "# need_idle: GPU memory.used <= ${IDLE_MIB:-1500} MiB" >&3; return 0; fi
  # a server that was just stopped (SGLang, vLLM) frees its memory a few seconds later: wait for that, give up on a real foreign job
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

RUNGS="seq_nokv seq_kv cb_naive cb_fa2 cb_fa2_prefill cb_fa2_prefill_compile"
rung_flags() {   # everything record_demo needs besides the sizes (cumulative from cb_fa2 on)
  case $1 in
    seq_nokv)               echo "--attn naive --mode seq --no-kv-cache";;
    seq_kv)                 echo "--attn naive --mode seq";;
    cb_naive)               echo "--attn naive --mode cb";;
    cb_fa2)                 echo "--attn fa2 --mode cb";;
    cb_fa2_prefill)         echo "--attn fa2 --mode cb --prefill-last-logits --prefill-batch-tokens $PBT";;
    cb_fa2_prefill_compile) echo "--attn fa2 --mode cb --prefill-last-logits --prefill-batch-tokens $PBT --compile-decode";;
  esac
}
rung_cs() { case $1 in seq_nokv) echo "1";; seq_kv) echo "1 32";; *) echo "$CS";; esac; }
rung_desc() {
  case $1 in
    seq_nokv)               echo "rung 1 = no KV cache: every token re-runs the whole prefix, one request at a time";;
    seq_kv)                 echo "rung 2 = rung 1 + KV cache (still one request at a time)";;
    cb_naive)               echo "rung 3 = rung 2 + continuous batching, naive attention, ragged left-padded KV (his original engine)";;
    cb_fa2)                 echo "rung 4 = rung 3 + pre-allocated KV + flash_attn_with_kvcache decode + one batched argmax (the three cannot be separated)";;
    cb_fa2_prefill)         echo "rung 5 = rung 4 + waiting prompts packed into one right-padded prefill forward (<= $PBT padded tokens), one group per round then one decode step, LM head on the last prompt position only";;
    cb_fa2_prefill_compile) echo "rung 6 = rung 5 + the decode forward as one torch.compile'd graph, attention through a registered custom op; the best configuration";;
  esac
}
cell_n()    { if [ "$1" = seq_kv ] && [ "$2" = 32 ]; then echo 64; else nreq "$2"; fi; }          # serial C=32 would take ~15 min per repetition
cell_warm() { case $1 in seq_*) local w; w=$(nwarm "$2"); [ "$w" -gt 16 ] && w=16; echo $w;; *) nwarm "$2";; esac; }   # a serial warm-up of 64 requests buys nothing

ingest_engine() { # arm name notes
  local arm=$1 name=$2 notes=$3 recs="" f
  for f in $RAW/${name}_r*.json; do [ -f "$f" ] && grep -q '"status": "ok"' "$f" && recs="$recs $f"; done
  if [ "${DRY_RUN:-0}" = 1 ] || [ -n "$recs" ]; then
    x $PY $CMP/serving/closed_loop.py ingest --system engine_${arm}_bfloat16 --recordings ${recs:-"<recordings>"} --out $CSV --notes "$notes" > $LOGS/d_ingest_$name.log 2>&1
  else st "no ok recording for $name (OOM / failed): not in the table"; fi
}

phase_ladder() {
  need_idle; st "ladder"; mf ladder
  local arm C name n w
  for arm in $RUNGS; do
    for C in $(rung_cs $arm); do
      name=${arm}_c${C}; n=$(cell_n $arm $C); w=$(cell_warm $arm $C)
      st "ladder $name"
      ( cd $SERV && x timeout 1500 $PY benchmarks/record_demo.py --preset 190m --prompts-json $WC --device cuda --dtype bfloat16 \
          $(rung_flags $arm) --concurrency $C --n-requests $n --reps $REPS --warmup-requests $w --arm-name $name --out $RAW/$name.json ) \
          > $LOGS/d_engine_$name.log 2>&1
      st "ladder $name rc=$?"
      ingest_engine $arm $name "$(rung_desc $arm); n_requests=$n; warm-up $w; $RI"
    done
  done
  st "STAGE_D_ladder_DONE"
}

sg_variant() {   # variant attn extra-env...
  local variant=$1 attn=$2; shift 2
  x timeout 2400 env PY=$SGLV/bin/python HF_DIR=$HFDIR ATTN=$attn DTYPE=bfloat16 WORKLOAD=$WC CONC="$CS" REPS=$REPS N_MULT=8 N_MIN=32 SERVING_CSV=$CSV \
      NOTES_EXTRA="stage D; $RI; requests = max(8 x C, 32); untimed warm-up 2 x C" "$@" \
      bash $CMP/serving/run_sglang.sh $variant > $LOGS/d_sglang_${variant}_${attn}.log 2>&1
  st "sglang $variant rc=$?"
}
phase_sglang() {
  need_idle; st "sglang"; mf sglang
  [ -d "$HFDIR" ] || x $PY $CMP/convert_hf/convert_190m_to_hf.py --random-init --out "$HFDIR" --tokenizer gpt2 > $LOGS/d_convert_random_init.log 2>&1
  sg_variant cb_only flashinfer MEM_FRACTION=0.8
  sg_variant default_noradix flashinfer CUDA_GRAPH_MAX_BS=64
  st "STAGE_D_sglang_DONE"
}

vl_variant() {
  local variant=$1
  x timeout 2400 env PY=$VLLV/bin/python HF_DIR=$HFDIR DTYPE=bfloat16 WORKLOAD=$WC CONC="$CS" REPS=$REPS N_MULT=8 N_MIN=32 SERVING_CSV=$CSV \
      NOTES_EXTRA="stage D; $RI; requests = max(8 x C, 32); untimed warm-up 2 x C" \
      bash $CMP/serving/run_vllm.sh $variant > $LOGS/d_vllm_${variant}.log 2>&1
  st "vllm $variant rc=$?"
}
phase_vllm() {   # two independent server starts, one after the other (each takes ~19.5 GB)
  need_idle; st "vllm"; mf vllm
  vl_variant nograph_noprefix
  vl_variant default_noprefix
  st "STAGE_D_vllm_DONE"
}

phase_openloop() {   # arrivals on a clock; the in-flight cap is the HARNESS's admission control (the engine has none)
  need_idle; st "openloop"; mf openloop
  local R name flags=${OL_FLAGS:-"$(rung_flags cb_fa2_prefill_compile)"}
  for R in $RATES; do
    name=openloop_r${R}; st "openloop engine rate=$R"
    ( cd $SERV && x timeout 1800 $PY benchmarks/record_demo.py --preset 190m --prompts-json $WC --device cuda --dtype bfloat16 $flags \
        --concurrency $CAP --arrival-rate $R --arrival-cv 1 --duration $DURATION --reps $REPS --arm-name $name --out $RAW/$name.json ) \
        > $LOGS/d_engine_$name.log 2>&1
    st "openloop engine rate=$R rc=$?"
    local recs="" f
    for f in $RAW/${name}_r*.json; do [ -f "$f" ] && grep -q '"status": "ok"' "$f" && recs="$recs $f"; done
    if [ "${DRY_RUN:-0}" = 1 ] || [ -n "$recs" ]; then
      x $PY $CMP/serving/closed_loop.py ingest --open-loop --system engine_cb_fa2_prefill_compile_bfloat16 --recordings ${recs:-"<recordings>"} \
          --slo-ttft-s $SLO_TTFT_S --slo-tpot-ms $SLO_TPOT_MS --out $CSV \
          --notes "open loop; in-flight cap $CAP is applied by the harness (the engine has no admission control); $RI" > $LOGS/d_ingest_$name.log 2>&1
    fi
  done
  local spec variant attn
  for spec in "cb_only flashinfer MEM_FRACTION=0.8" "default_noradix flashinfer CUDA_GRAPH_MAX_BS=64"; do
    set -- $spec; variant=$1; attn=$2; shift 2
    st "openloop sglang $variant"
    x timeout 2400 env PY=$SGLV/bin/python HF_DIR=$HFDIR ATTN=$attn DTYPE=bfloat16 WORKLOAD=$WC REPS=$REPS SERVING_CSV=$CSV \
        ARRIVAL_RATES="$RATES" ARRIVAL_CV=1 DURATION=$DURATION MAX_INFLIGHT=$CAP SLO_TTFT_S=$SLO_TTFT_S SLO_TPOT_MS=$SLO_TPOT_MS \
        NOTES_EXTRA="stage D open loop; in-flight cap $CAP applied by the client; $RI" "$@" \
        bash $CMP/serving/run_sglang.sh $variant > $LOGS/d_sglang_openloop_${variant}.log 2>&1
    st "openloop sglang $variant rc=$?"
  done
  st "STAGE_D_openloop_DONE"
}

phase_c5() {   # best configuration vs HF generate: bf16 (FlashAttention has no fp32 kernel), report only
  need_idle; st "c5"; mf c5
  ( cd $SERV && x timeout 1800 $PY $CMP/serving/c5_greedy_equal.py --checkpoint $CKPT --workload $W128 --n-prompts 64 --new-tokens 64 --dtype bfloat16 --device cuda \
      --attn fa2 --prefill-last-logits --prefill-batch-tokens $PBT --compile-decode --label cb_fa2_prefill_compile --concurrency 32 --modes cb \
      --out $FINAL/serving_correctness_stage_d.csv ) > $LOGS/d_c5_best_config.log 2>&1
  st "c5 rc=$?"; st "STAGE_D_c5_DONE"
}

phase_wrapup() {   # nobody may be watching: leave one tarball and one status page behind
  st "wrapup"
  x $PY $CMP/remote/stage_d_summary.py --csv $CSV > $FINAL/serving_stage_d_summary.txt 2>&1
  ( cd $FINAL && x tar czf $ROOT/stage_d_results.tgz serving_stage_d.csv serving_stage_d_summary.txt serving_correctness_stage_d.csv \
      serving_raw/engine_d logs/serving manifest.json ) > $LOGS/d_wrapup_tar.log 2>&1 || st "tar: some files were missing (see $LOGS/d_wrapup_tar.log)"
  st "STAGE_D_ALL_DONE (status page: $FINAL/serving_stage_d_summary.txt; tarball: $ROOT/stage_d_results.tgz)"
}

phase_smoke() {   # toy sizes; every line below must PASS before any real phase is started
  need_idle
  local S=${SMOKE:-/tmp/smoke_stage_d} fail=0 arm flags C
  [ -z "${ONLY:-}" ] && rm -rf "$S"; mkdir -p "$S"
  chk() { local name=$1; shift
    if [ -n "${ONLY:-}" ] && ! echo " $ONLY " | grep -q " $name "; then return 0; fi   # ONLY="name ...": re-run just these items
    if x "$@" > "$S/$name.log" 2>&1; then echo "SMOKE PASS $name"; else echo "SMOKE FAIL $name  (see $S/$name.log: $(tail -1 "$S/$name.log" | cut -c1-150))"; fail=1; fi; }
  for arm in $RUNGS; do   # one tiny cell per rung
    case $arm in seq_*) C=1;; *) C=4;; esac
    ( cd $SERV && chk engine_$arm $PY benchmarks/record_demo.py --preset 190m --prompts-json $WC --device cuda --dtype bfloat16 $(rung_flags $arm) \
        --concurrency $C --n-requests 8 --reps 2 --warmup-requests 4 --arm-name smoke_$arm --out $S/smoke_$arm.json )
    chk ingest_$arm $PY $CMP/serving/closed_loop.py ingest --system smoke_$arm --recordings $S/smoke_${arm}_r1.json $S/smoke_${arm}_r2.json --out $S/serving.csv
  done
  ( cd $SERV && chk engine_openloop $PY benchmarks/record_demo.py --preset 190m --prompts-json $WC --device cuda --dtype bfloat16 \
      $(rung_flags cb_fa2_prefill_compile) --concurrency 8 --arrival-rate 5 --duration 4 --reps 2 --warmup-requests 8 --arm-name smoke_openloop --out $S/smoke_openloop.json )
  chk ingest_openloop $PY $CMP/serving/closed_loop.py ingest --open-loop --system smoke_openloop --recordings $S/smoke_openloop_r1.json $S/smoke_openloop_r2.json \
      --slo-ttft-s $SLO_TTFT_S --slo-tpot-ms $SLO_TPOT_MS --out $S/serving.csv
  ( cd $SERV && chk c5_best_config $PY $CMP/serving/c5_greedy_equal.py --checkpoint $CKPT --workload $W128 --n-prompts 8 --new-tokens 8 --dtype bfloat16 --device cuda \
      --attn fa2 --prefill-last-logits --prefill-batch-tokens $PBT --compile-decode --label smoke --concurrency 8 --modes cb --out $S/c5.csv )
  chk sglang_openloop env PY=$SGLV/bin/python HF_DIR=$HFDIR ATTN=flashinfer DTYPE=bfloat16 WORKLOAD=$WC MEM_FRACTION=0.8 ARRIVAL_RATES="5" DURATION=3 REPS=1 MAX_INFLIGHT=8 SERVING_CSV=$S/sg_ol.csv bash $CMP/serving/run_sglang.sh cb_only
  chk vllm_closed env PY=$VLLV/bin/python HF_DIR=$HFDIR DTYPE=bfloat16 WORKLOAD=$WC CONC="2" REPS=1 N_MULT=4 N_MIN=8 SERVING_CSV=$S/vllm.csv bash $CMP/serving/run_vllm.sh nograph_noprefix
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  if [ $fail = 0 ]; then st "STAGE_D_smoke_DONE (all components passed)"; else st "STAGE_D_smoke_FAILED"; return 1; fi
}

case "${1:-}" in
  smoke) phase_smoke;;
  ladder) phase_ladder;; sglang) phase_sglang;; vllm) phase_vllm;; openloop) phase_openloop;; c5) phase_c5;;
  all) phase_ladder; phase_sglang; phase_vllm; phase_openloop; phase_c5; phase_wrapup;;
  wrapup) phase_wrapup;;
  *) echo "usage: $0 smoke|ladder|sglang|vllm|openloop|c5|wrapup|all"; exit 2;;
esac
