#!/usr/bin/env bash
# Stage B (serving), decision record docs/sprints/week7-discussion.md D-7 "Serving". One phase per call so every phase can be pulled
# back before the next one starts (a credit-exhausted box stops and its disk is unreachable).
#
#   bash run_stage_b.sh smoke     EVERY component of the phases below at toy size (output in /tmp, never in results/final). Run it first:
#                                 three earlier launches died on a missing dependency / flag that a 1-minute smoke test would have caught
#   bash run_stage_b.sh prep      env_check + the random-init HF model (HF generate and SGLang read it)
#   bash run_stage_b.sh engine    engine: naive -> sdpa wrapper -> fa2 (prealloc KV + flash-attn decode + batched argmax)
#   bash run_stage_b.sh hf        HF generate: eager | sdpa | flash_attention_2, serial
#   bash run_stage_b.sh sglang    SGLang cb_only (flashinfer, triton) and default minus radix cache (flashinfer)
#   bash run_stage_b.sh attrib    cProfile of the engine, one decode step, C5 on the GPU, fa2 real-weights check
#   DRY_RUN=1 bash run_stage_b.sh all     print every command instead of running it (tested on the laptop)
#
# Rules: bf16 everywhere; mixed output lengths 32-96; requests = max(8 x C, 32); untimed warm-up max(2 x C, 8); 3 repetitions;
# random-init weights (speed only); nothing here spends money or touches the instance. Progress: $ST (last line STAGE_B_<phase>_DONE).
set -u
ROOT=${ROOT:-/workspace/llm-stack}; CMP=$ROOT/benchmarks/compare; SERV=$ROOT/serving
FINAL=$CMP/results/final; LOGS=$FINAL/logs/serving; RAW=$FINAL/serving_raw; CSV=$FINAL/serving.csv
W=$CMP/workloads/workload_mixed4096.json; W128=$CMP/workloads/workload_wiki128.json
PY=${PY:-/workspace/venv-fa/bin/python}; SGLV=${SGLV:-/workspace/venv-sglang}; HFDIR=${HFDIR:-/workspace/llm190m-hf-random}
CKPT=${CKPT:-/workspace/ckpt/model_only.pt}; CS=${CS:-"1 8 32 64"}; REPS=${REPS:-3}
export PYTHONPATH=$ROOT/training/cs336-basics:$ROOT/training
ST=${ST:-/workspace/STAGE_B_STATUS.txt}
RI="random-init weights; speed-only; RTX 3090 Ti (driver 550, SW power cap active); bf16"
# DRY_RUN prints on fd 3 (a copy of the original stderr): the callers redirect stdout AND stderr into log files, which would swallow the echo
exec 3>&2
x() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ $*" >&3; else "$@"; fi; }
st() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "# $*" >&3; else echo "$(date -Is) $*" >> "$ST"; fi; }
mkdir -p "$LOGS" "$RAW/engine" 2>/dev/null || true
[ "${DRY_RUN:-0}" = 1 ] && mkdir -p "$SERV" 2>/dev/null   # so that `( cd $SERV && ... )` reaches the echo
mf() { x env PYTHONPATH=$CMP:$PYTHONPATH $PY -c "import manifest; manifest.record('serving_$1', out_dir='$FINAL', argv=['run_stage_b.sh','$1'], extra=dict(csv='serving.csv', sglang='$SGLV'))"; }
nreq() { local n=$((8 * $1)); [ $n -lt 32 ] && n=32; echo $n; }
nwarm() { local n=$((2 * $1)); [ $n -lt 8 ] && n=8; echo $n; }

phase_prep() {
  st "prep"; mf prep
  x $PY $CMP/env_check.py --strict > $LOGS/env_check_stage_b.log 2>&1 || st "WARN env_check not PASS (see logs/serving/env_check_stage_b.log)"
  [ -d "$HFDIR" ] || x $PY $CMP/convert_hf/convert_190m_to_hf.py --random-init --out "$HFDIR" --tokenizer gpt2 > $LOGS/convert_random_init.log 2>&1
  st "STAGE_B_prep_DONE"
}

ingest_engine() { # arm name notes
  local arm=$1 name=$2 notes=$3 recs="" f
  for f in $RAW/engine/${name}_r*.json; do [ -f "$f" ] && grep -q '"status": "ok"' "$f" && recs="$recs $f"; done
  if [ "${DRY_RUN:-0}" = 1 ] || [ -n "$recs" ]; then
    x $PY $CMP/serving/closed_loop.py ingest --system engine_cb_${arm}_bfloat16 --recordings ${recs:-"<recordings>"} --out $CSV --notes "$notes" > $LOGS/ingest_$name.log 2>&1
  else st "no ok recording for $name (OOM / failed): not in the table"; fi
}

phase_engine() {
  st "engine"; mf engine
  local arm C name
  for arm in naive sdpa fa2; do
    case $arm in
      naive) D="arm=naive: HIS original attention fn, ragged KV (per-step torch.cat)";;
      sdpa)  D="arm=sdpa: wrapper, F.scaled_dot_product_attention in the same ragged path";;
      fa2)   D="arm=fa2: preallocated KV + flash_attn_with_kvcache + one batched argmax (three changes in one)";;
    esac
    for C in $CS; do
      name=${arm}_c${C}
      st "engine $name"
      ( cd $SERV && x timeout 1800 $PY benchmarks/record_demo.py --preset 190m --prompts-json $W --device cuda --attn $arm --dtype bfloat16 \
          --mode cb --concurrency $C --n-requests $(nreq $C) --reps $REPS --warmup-requests $(nwarm $C) --arm-name $name --out $RAW/engine/$name.json ) \
          > $LOGS/engine_$name.log 2>&1
      st "engine $name rc=$?"
      ingest_engine $arm $name "$D; mixed output lengths 32-96; n_requests=$(nreq $C); warm-up $(nwarm $C); $RI"
    done
  done
  st "STAGE_B_engine_DONE"
}

phase_hf() {
  st "hf"; mf hf
  local attn
  for attn in eager sdpa flash_attention_2; do
    st "hf $attn"
    x timeout 3000 $PY $CMP/serving/closed_loop.py run --backend hf --system hf_generate_${attn}_bfloat16 --model-dir $HFDIR --dtype bfloat16 \
        --attn $attn --workload $W --n 64 --concurrency 1 32 --reps $REPS --warmup 8 --out $CSV \
        --notes "generate() one request at a time (serial: throughput does not depend on C; TTFT includes the lock queue); attn=$attn; 64 requests; mixed output lengths; $RI" \
        > $LOGS/hf_$attn.log 2>&1
    st "hf $attn rc=$?"
  done
  st "STAGE_B_hf_DONE"
}

sglang_variant() { # variant attn [extra env...]
  local variant=$1 attn=$2; shift 2
  st "sglang $variant $attn"
  x timeout 3600 env PY=$SGLV/bin/python HF_DIR=$HFDIR ATTN=$attn DTYPE=bfloat16 WORKLOAD=$W CONC="$CS" REPS=$REPS N_MULT=8 N_MIN=32 SERVING_CSV=$CSV \
      NOTES_EXTRA="$RI; mixed output lengths 32-96; requests = max(8 x C, 32); untimed warm-up 2 x C" "$@" bash $CMP/serving/run_sglang.sh $variant > $LOGS/sglang_${variant}_${attn}.log 2>&1
  st "sglang $variant $attn rc=$?"
}

phase_sglang() {
  st "sglang"; mf sglang
  # cb_only: everything switchable is off (needs --mem-fraction-static, see run_sglang.sh). default: all on, minus the radix cache
  # (the workload repeats prompts, a prefix cache would flatter it) and with CUDA graphs captured up to C=64.
  sglang_variant cb_only flashinfer MEM_FRACTION=0.8
  sglang_variant cb_only triton MEM_FRACTION=0.8
  sglang_variant default_noradix flashinfer CUDA_GRAPH_MAX_BS=64
  st "STAGE_B_sglang_DONE"
}

phase_attrib() {
  st "attrib"; mf attrib
  local prof=$FINAL/serving_profiles; mkdir -p "$prof" 2>/dev/null || true
  # 1. cProfile of the engine's event loop + Python (a separate run: cProfile slows Python, so it never enters the throughput table)
  ( cd $SERV && x $PY benchmarks/record_demo.py --preset 190m --prompts-json $W --device cuda --attn naive --dtype bfloat16 --mode cb \
      --concurrency 32 --n-requests 256 --reps 1 --warmup-requests 64 --arm-name cprofile_naive_c32 --cprofile $prof/engine_naive_c32.prof \
      --out $prof/cprofile_naive_c32.json ) > $LOGS/cprofile_run.log 2>&1
  x $PY $CMP/serving/analyze_cprofile.py $prof/engine_naive_c32.prof > $prof/engine_naive_c32_cprofile.txt 2>&1
  # 2. one batched decode step (B=32): wall / GPU busy / launches / KV traffic, per arm
  local arm
  for arm in naive sdpa fa2; do
    ( cd $SERV && x $PY benchmarks/profile_decode_step.py --workload $W --batch 32 --attn $arm --dtype bfloat16 ) > $prof/decode_step_$arm.txt 2>&1
  done
  # 3. C5 on the GPU: strict fp32 must match HF token for token; bf16 only reports where the streams diverge
  x $PY $CMP/serving/c5_greedy_equal.py --checkpoint $CKPT --workload $W128 --n-prompts 32 --new-tokens 32 --dtype float32 --device cuda \
      --out $FINAL/serving_correctness_gpu.csv > $LOGS/c5_fp32_gpu.log 2>&1
  x $PY $CMP/serving/c5_greedy_equal.py --checkpoint $CKPT --workload $W128 --n-prompts 32 --new-tokens 32 --dtype bfloat16 --device cuda \
      --out $FINAL/serving_correctness_gpu.csv > $LOGS/c5_bf16_gpu.log 2>&1
  # 4. the fa2 path against real weights (the ladder's last rung must be shown correct, not assumed)
  ( cd $SERV && x $PY benchmarks/check_fa2_decode.py --workload $W128 --checkpoint $CKPT --dtype bfloat16 ) > $LOGS/check_fa2_real_weights_bf16.log 2>&1
  st "STAGE_B_attrib_DONE"
}

phase_smoke() {   # toy sizes; every line below must PASS before the real phases are started
  local S=${SMOKE:-/tmp/smoke_stage_b} fail=0 arm attn n=0
  rm -rf "$S"; mkdir -p "$S"
  chk() { local name=$1; shift
    if x "$@" > "$S/$name.log" 2>&1; then echo "SMOKE PASS $name"; else echo "SMOKE FAIL $name  (see $S/$name.log: $(tail -1 "$S/$name.log" | cut -c1-150))"; fail=1; fi; }
  for arm in naive sdpa fa2; do
    ( cd $SERV && chk engine_$arm $PY benchmarks/record_demo.py --preset 190m --prompts-json $W --device cuda --attn $arm --dtype bfloat16 --mode cb \
        --concurrency 4 --n-requests 8 --reps 2 --warmup-requests 4 --arm-name smoke_$arm --out $S/smoke_$arm.json )
    chk ingest_$arm $PY $CMP/serving/closed_loop.py ingest --system smoke_$arm --recordings $S/smoke_${arm}_r1.json $S/smoke_${arm}_r2.json --out $S/serving.csv
  done
  for attn in eager sdpa flash_attention_2; do
    chk hf_$attn $PY $CMP/serving/closed_loop.py run --backend hf --system smoke_hf_$attn --model-dir $HFDIR --dtype bfloat16 --attn $attn \
        --workload $W --n 6 --concurrency 1 2 --reps 1 --warmup 2 --out $S/serving.csv
  done
  ( cd $SERV && chk cprofile $PY benchmarks/record_demo.py --preset 190m --prompts-json $W --device cuda --attn naive --dtype bfloat16 \
      --mode cb --concurrency 4 --n-requests 8 --reps 1 --warmup-requests 4 --arm-name smoke_cprofile --cprofile $S/e.prof --out $S/cprofile.json )
  chk analyze_cprofile $PY $CMP/serving/analyze_cprofile.py $S/e.prof
  for arm in naive sdpa fa2; do
    ( cd $SERV && chk decode_step_$arm $PY benchmarks/profile_decode_step.py --workload $W --batch 8 --attn $arm --dtype bfloat16 --steps 3 )
  done
  chk c5_fp32 $PY $CMP/serving/c5_greedy_equal.py --checkpoint $CKPT --workload $W128 --n-prompts 3 --new-tokens 8 --dtype float32 --device cuda --out $S/c5.csv
  chk c5_bf16 $PY $CMP/serving/c5_greedy_equal.py --checkpoint $CKPT --workload $W128 --n-prompts 3 --new-tokens 8 --dtype bfloat16 --device cuda --out $S/c5.csv
  ( cd $SERV && chk check_fa2 $PY benchmarks/check_fa2_decode.py --workload $W128 --checkpoint $CKPT --dtype bfloat16 --rows 2 --steps 4 )
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  if [ $fail = 0 ]; then st "STAGE_B_smoke_DONE (all components passed)"; else st "STAGE_B_smoke_FAILED"; return 1; fi
}

case "${1:-}" in
  smoke) phase_smoke;;
  prep) phase_prep;; engine) phase_engine;; hf) phase_hf;; sglang) phase_sglang;; attrib) phase_attrib;;
  all) phase_prep; phase_engine; phase_hf; phase_sglang; phase_attrib;;
  *) echo "usage: $0 smoke|prep|engine|hf|sglang|attrib|all"; exit 2;;
esac
