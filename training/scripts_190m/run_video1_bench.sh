#!/usr/bin/env bash
# Video 1 data: per-epoch time of {naive attention, FlashAttention} x {1 GPU, naive DDP, overlapped DDP}
# on the 2x3090 box, 190M model shape, long context. Run on the vast instance AFTER the 190M run
# has finished and its eval is done (the GPUs must be idle -- the guard refuses otherwise).
#
#   bash scripts_190m/run_video1_bench.sh guard                  # GPUs idle? (also run by every stage)
#   bash scripts_190m/run_video1_bench.sh smoke                  # 2 epochs at ctx 1024, plumbing incl. NCCL (~3 min)
#   bash scripts_190m/run_video1_bench.sh calibrate              # longest ctx where NAIVE attention still fits
#   bash scripts_190m/run_video1_bench.sh full [CTX] [BATCH]     # the 6 configs, 12 timed epochs each
#   bash scripts_190m/run_video1_bench.sh ddp_flash              # DDP scaling with FlashAttention at ctx 1024 batch 8/GPU
#
# "epoch" = a fixed number of samples (S = 32 x batch): the 1-GPU run takes S/batch steps, a 2-GPU
# run S/(2*batch) steps, so every config's epoch time is for the SAME amount of data and the numbers
# are directly comparable. Results: artifacts/video1/<stage>/epoch_times__*.json, then
# summarize_epoch_times.py -> facts_v1.json. Pull artifacts/video1/ back before destroying the box.
#
# Why fwe_valid.npy as the "train" file: timing does not depend on token values, and the 800M-token
# train file would make DistributedSampler build a randperm of ~8e8 indices (6+ GB, slow) every epoch.
set -euo pipefail
STAGE=${1:?stage: guard smoke calibrate full}
D=data/pretrain190m
PY="uv run --no-sync python"
BENCH="$PY cs336_systems/experiments/benchmark_lm_matrix.py --config configs_190m/fwe.json \
  --train_path $D/fwe_valid.npy --val_path $D/fwe_valid.npy"
mkdir -p artifacts/video1 logs

guard() {
  if pgrep -f "run_pipeline.py" >/dev/null; then
    echo "ABORT: run_pipeline.py is still running (training not finished) -- timings would be garbage"; exit 1
  fi
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1500{n++} END{print n+0}')
  if [[ $busy -gt 0 ]]; then
    echo "ABORT: $busy GPU(s) have >1.5 GB in use:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 1
  fi
  echo "guard ok: no training process, GPUs idle"
}

status_of() {  # status_of <dir> -> prints "kernel wrapper status" per json
  $PY - "$1" <<'EOF'
import glob, json, sys
for p in sorted(glob.glob(sys.argv[1] + "/epoch_times__*.json")):
    r = json.load(open(p)); print(f"  {r['kernel']:<28}{r['ddp']:<28}{r['status']}")
EOF
}

case $STAGE in
guard) guard ;;
smoke)
  guard
  $BENCH --ctx 1024 --batch 1 --samples_per_epoch 8 --timed_epochs 2 --warmup_epochs 1 \
    --kernels scaled_dot_prod_attention flash_attention_triton \
    --wrappers "Local No DDP" "Bucketed Overlapping DDP" --out_dir artifacts/video1/smoke
  status_of artifacts/video1/smoke
  ;;
calibrate)
  guard
  # Naive attention stores several (heads x ctx x ctx) fp32 matrices per layer for backward, so it
  # OOMs first. Test the worst case (bucketed DDP adds bucket buffers on top of the 1-GPU memory).
  chosen=""
  for ctx in 3072 2560 2048 1536; do
    out=artifacts/video1/calib_$ctx
    echo "=== ctx $ctx batch 1: naive attention, 1 GPU + bucketed DDP ==="
    $BENCH --ctx $ctx --batch 1 --samples_per_epoch 8 --timed_epochs 1 --warmup_epochs 1 \
      --kernels scaled_dot_prod_attention --wrappers "Local No DDP" "Bucketed Overlapping DDP" \
      --out_dir $out || true
    status_of $out
    # grep -l exits 1 when nothing matches; under `set -e -o pipefail` that would kill the script
    # right after the first OOM candidate (it did), so the no-match case is handled explicitly.
    n_ok=$(grep -l '"status": "ok"' $out/epoch_times__*.json 2>/dev/null | wc -l) || n_ok=0
    if [[ $n_ok -eq 2 ]]; then chosen=$ctx; break; fi
  done
  [[ -n $chosen ]] || { echo "naive attention does not fit even at ctx 1536 batch 1"; exit 1; }
  echo "$chosen" > artifacts/video1/chosen_ctx.txt
  echo "CHOSEN ctx=$chosen batch=1 (all calib_* dirs kept: they are the OOM-boundary evidence)"
  ;;
full)
  guard
  CTX=${2:-$(cat artifacts/video1/chosen_ctx.txt)}; B=${3:-1}
  S=$((32 * B))
  out=artifacts/video1/full_ctx${CTX}_b${B}
  $BENCH --ctx $CTX --batch $B --samples_per_epoch $S --timed_epochs 12 --warmup_epochs 3 \
    --kernels scaled_dot_prod_attention flash_attention_triton \
    --wrappers "Local No DDP" "Naive DDP" "Bucketed Overlapping DDP" --out_dir $out 2>&1 | tee logs/video1_full.log
  $PY cs336_systems/experiments/summarize_epoch_times.py $out
  ;;
ddp_flash)
  # At batch 1/GPU the 190M model's step is so short that all-reducing 760 MB of gradients over PCIe
  # dominates it (smoke: a 2-GPU step took ~2x a 1-GPU step), so DDP cannot show a speed-up there.
  # DDP is therefore measured separately at a compute-heavy per-GPU batch. Naive attention OOMs at
  # this batch, so this stage is FlashAttention-only; it is NOT mixed into the FA-vs-naive comparison.
  guard
  out=artifacts/video1/ddp_flash_ctx1024_b8
  $BENCH --ctx 1024 --batch 8 --samples_per_epoch 256 --timed_epochs 12 --warmup_epochs 3 \
    --kernels flash_attention_triton \
    --wrappers "Local No DDP" "Naive DDP" "Bucketed Overlapping DDP" --out_dir $out 2>&1 | tee logs/video1_ddp_flash.log
  $PY cs336_systems/experiments/summarize_epoch_times.py $out
  ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
