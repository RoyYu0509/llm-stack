#!/usr/bin/env bash
# 190M rerun, one arm per 2x3090 instance. Run from llm-stack/training on the instance:
#
#   bash scripts_190m/run_arm.sh <fwe|c4> setup    # uv + deps (~5 min)
#   bash scripts_190m/run_arm.sh <fwe|c4> build    # token files (~20-40 min, needs HF_TOKEN)
#   bash scripts_190m/run_arm.sh <fwe|c4> smoke    # 300-step GPU smoke + both eval scripts (~10 min)
#   bash scripts_190m/run_arm.sh <fwe|c4> launch   # the real run, detached in tmux session "run190m"
#   bash scripts_190m/run_arm.sh <fwe|c4> status   # latest losses, speed, GPU
#   bash scripts_190m/run_arm.sh <fwe|c4> eval     # final evals -> artifacts/190m/
#
# Both arms build the SAME three validation files (fwe / c4 / wiki, same seeds), so every
# checkpoint is scored on identical tokens; `build` prints their md5 so you can compare
# the two instances. Decision table: llm/docs/sprints/iflytek-190m-run.md
set -euo pipefail
ARM=${1:?arm: fwe or c4}
STAGE=${2:?stage: setup build smoke launch status eval}
[[ $ARM == fwe || $ARM == c4 ]] || { echo "arm must be fwe or c4"; exit 1; }
D=data/pretrain190m
PY="uv run --no-sync python"
BUILD="$PY cs336_systems/data_harvest/build_pretrain_corpus.py --out-dir $D"
STREAMS="--stream fwe=$D/fwe_valid.npy --stream c4=$D/c4_valid.npy --stream wiki=$D/wiki_valid.npy"
mkdir -p logs artifacts/190m

case $STAGE in
setup)
  command -v uv >/dev/null || pip install -q uv
  command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)
  (cd .. && uv sync --all-packages)
  $PY -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), 'triton', triton.__version__)"
  df -h . | tail -1; free -g | head -2; nproc
  ;;
build)
  : "${HF_TOKEN:?export HF_TOKEN=... first (anonymous HF downloads get rate-limited)}"
  if [[ $ARM == fwe ]]; then
    $BUILD --dataset HuggingFaceFW/fineweb-edu --config sample-10BT --train-tokens 800000000 --val-tokens 5000000 --prefix fwe
    $BUILD --dataset allenai/c4 --config en --val-split validation --train-tokens 0 --val-tokens 5000000 --prefix c4
  else
    $BUILD --dataset allenai/c4 --config en --val-split validation --train-tokens 800000000 --val-tokens 5000000 --prefix c4
    $BUILD --dataset HuggingFaceFW/fineweb-edu --config sample-10BT --train-tokens 0 --val-tokens 5000000 --prefix fwe
  fi
  $BUILD --dataset wikimedia/wikipedia --config 20231101.en --train-tokens 0 --val-tokens 3000000 --prefix wiki
  md5sum $D/*_valid.npy | tee logs/valid_md5_$ARM.txt
  ls -la $D
  ;;
smoke)
  $PY - "$ARM" <<'EOF'
import json, sys
arm = sys.argv[1]
c = json.load(open(f"configs_190m/{arm}.json"))
c["_NOTE"] = "GPU smoke: same model/data/compile path; 300 steps, 3 checkpoints (so pruning runs), 3 evals."
c["training"].update(max_iters=300, warmup_iters=30, val_bat_num=10)
c["checkpointing"].update(checkpoint_dir=f"checkpoints_190m_smoke_{arm}", checkpoint_every_n_steps=100,
                          eval_every_n_steps=100, log_interval=10)
c["wandb"] = {}
json.dump(c, open(f"configs_190m/smoke_{arm}.json", "w"), indent=2)
EOF
  rm -rf checkpoints_190m_smoke_$ARM
  $PY cs336_systems/experiments/run_pipeline.py --config configs_190m/smoke_$ARM.json --skip_data 2>&1 | tee logs/smoke_$ARM.log
  CK=checkpoints_190m_smoke_$ARM/checkpoint_epoch_0001.pt
  $PY cs336_systems/experiments/eval_checkpoint.py --config configs_190m/smoke_$ARM.json --checkpoint $CK $STREAMS --batches 5
  $PY cs336_systems/experiments/eval_downstream.py --config configs_190m/smoke_$ARM.json --checkpoint $CK --limit 50
  echo "================ SMOKE CHECKLIST ================"
  echo "1) first / last train loss (first ~10.8 = ln 50257; last must be clearly lower):"
  grep "Rank 0 | step" logs/smoke_$ARM.log | grep "train loss" | sed -n '1p;$p'
  echo "2) val losses (3 evals + FINAL, all finite, decreasing):"
  grep "Rank 0" logs/smoke_$ARM.log | grep "Val loss"
  echo "3) checkpoints on disk (expect step_0200, step_0300, epoch_0001; step_0100 pruned):"
  ls checkpoints_190m_smoke_$ARM
  echo "4) dataset line (train samples ~781K for 800M tokens at stride 1024):"
  grep "data_stride" logs/smoke_$ARM.log
  echo "5) peak GPU memory right now / speed: run 'status' during launch instead"
  ;;
launch)
  [[ -f $D/${ARM}_train.npy ]] || { echo "missing $D/${ARM}_train.npy -- run build first"; exit 1; }
  tmux new-session -d -s run190m \
    "$PY cs336_systems/experiments/run_pipeline.py --config configs_190m/$ARM.json --skip_data 2>&1 | tee logs/$ARM.log"
  echo "launched: tmux attach -t run190m (detach: Ctrl-b d); log: logs/$ARM.log"
  ;;
status)
  grep "Rank 0 | step" logs/$ARM.log | grep "train loss" | tail -3 || true
  grep "Rank 0" logs/$ARM.log | grep "Val loss" | tail -3 || true
  tail -c 2000 logs/$ARM.log | tr '\r' '\n' | grep -a "Epoch" | tail -1 || true
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
  df -h . | tail -1
  ;;
eval)
  CK=checkpoints_190m_$ARM/checkpoint_epoch_0001.pt
  $PY cs336_systems/experiments/eval_checkpoint.py --config configs_190m/$ARM.json --checkpoint $CK $STREAMS \
      --batches 200 --json-out artifacts/190m/${ARM}_streams.json
  $PY cs336_systems/experiments/eval_downstream.py --config configs_190m/$ARM.json --checkpoint $CK \
      --json-out artifacts/190m/${ARM}_downstream.json
  $PY cs336_systems/experiments/sample_from_checkpoint.py --config configs_190m/$ARM.json --checkpoint $CK \
      --prompt "The history of the internet began in the" --prompt "Photosynthesis is the process by which" \
      --prompt "In recent years, researchers have found that" | tee artifacts/190m/${ARM}_samples.txt
  cp logs/$ARM.log artifacts/190m/
  ;;
*)
  echo "unknown stage $STAGE"; exit 1 ;;
esac
