#!/usr/bin/env bash
# T1 / T2 / T7 -- DDP comparison on the 2x3090 box. Reuses benchmark_lm_matrix.py (the existing
# harness) and only adds the missing row: PyTorch DDP under the *same* protocol as the video1
# "ddp_flash" stage (190M, ctx 1024, 8 samples/GPU/step, 256 samples/epoch, 3 warmup + 12 timed
# epochs), measured in ONE session together with the other wrappers so rows are comparable.
#
#   bash ddp/run_t1.sh topo      # record GPU interconnect (nvidia-smi topo -m) -- README needs it
#   bash ddp/run_t1.sh t1        # Local / Naive / Pytorch DDP / Bucketed, bucket 25 MB
#   bash ddp/run_t1.sh sweep     # T2: bucket 1/5/10/25/50/100 MB for Bucketed + Pytorch DDP
#   bash ddp/run_t1.sh nccl      # T7: nccl-tests all_reduce_perf ceiling (needs the binary)
#   bash ddp/run_t1.sh ingest    # CSVs -> benchmarks/compare/results/ddp.csv
#
# Run from anywhere; TRAIN_DIR defaults to ../../training relative to this file. GPUs must be idle
# (same guard as scripts_190m/run_video1_bench.sh). This script spends GPU time = money: it does
# not start or rent any instance, you run it where the 3090s already are.
set -euo pipefail
STAGE=${1:?stage: topo t1 sweep nccl ingest}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR=${TRAIN_DIR:-"$HERE/../../../training"}
OUT_ROOT=${OUT_ROOT:-"$TRAIN_DIR/artifacts/compare_ddp"}
RESULTS="$HERE/../results"
PY="uv run --no-sync python"
D=data/pretrain190m
CTX=1024; BATCH=8; SAMPLES=256; WARMUP=3; TIMED=12
mkdir -p "$OUT_ROOT" "$RESULTS"

guard() {
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1500{n++} END{print n+0}')
  if [[ $busy -gt 0 ]]; then
    echo "ABORT: $busy GPU(s) have >1.5 GB in use -- timings would be garbage"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 1
  fi
  echo "guard ok: GPUs idle"
}

bench() {  # bench <out_dir> <bucket_mb> <wrapper...>
  local out=$1 bucket=$2; shift 2
  (cd "$TRAIN_DIR" && $PY cs336_systems/experiments/benchmark_lm_matrix.py \
    --config configs_190m/fwe.json --train_path $D/fwe_valid.npy --val_path $D/fwe_valid.npy \
    --ctx $CTX --batch $BATCH --samples_per_epoch $SAMPLES \
    --timed_epochs $TIMED --warmup_epochs $WARMUP --bucket_size_mb "$bucket" \
    --kernels flash_attention_triton --wrappers "$@" --out_dir "$out")
}

case $STAGE in
topo)
  { date; nvidia-smi; echo; nvidia-smi topo -m; echo; nvidia-smi topo -p2p r; } 2>&1 | tee "$OUT_ROOT/topo.txt"
  ;;
t1)
  guard
  bench "$OUT_ROOT/t1_b25" 25 "Local No DDP" "Naive DDP" "Pytorch DDP" "Bucketed Overlapping DDP"
  ;;
sweep)
  guard
  for mb in 1 5 10 25 50 100; do
    bench "$OUT_ROOT/sweep_b${mb}" "$mb" "Pytorch DDP" "Bucketed Overlapping DDP"
  done
  ;;
nccl)
  # https://github.com/NVIDIA/nccl-tests ; build once: make MPI=0 CUDA_HOME=/usr/local/cuda
  if ! command -v all_reduce_perf >/dev/null; then
    echo "all_reduce_perf not on PATH. Build nccl-tests and put build/ on PATH, then rerun."; exit 1
  fi
  guard
  # 760 MB of fp32 gradients ~ the 190M model; -g 2 = two GPUs in one process
  all_reduce_perf -b 8M -e 1G -f 2 -g 2 2>&1 | tee "$OUT_ROOT/nccl_all_reduce_perf.txt"
  ;;
ingest)
  for d in "$OUT_ROOT"/t1_b25 "$OUT_ROOT"/sweep_b*; do
    [[ -f "$d/lm_matrix_results.csv" ]] || continue
    $PY "$HERE/ingest_lm_matrix.py" "$d/lm_matrix_results.csv" --ctx $CTX \
      --out "$RESULTS/ddp.csv" --source "$(basename "$d")"
  done
  ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
