#!/usr/bin/env bash
# Record the two arms of inference-demo video on the GPU box, with the freshly trained 190M checkpoint.
# Run on the vast instance from ~/llm-stack-m2 (a copy of the main tree incl. the uncommitted continuous-batching
# code; ~/llm-stack is the clean worktree the training ran from).
#
#   bash serving/benchmarks/run_demo_recording.sh gate     # does this tree compute the SAME model as the training tree?
#   bash serving/benchmarks/run_demo_recording.sh smoke    # tiny random model on cuda, both modes (device bugs show here)
#   bash serving/benchmarks/run_demo_recording.sh record   # 3 x (W1 seq, W2 cb), alternating -> demo_out/
#
# W1 = one request at a time (KV cache), W2 = continuous batching; both naive attention, one GPU.
set -euo pipefail
ROOT=$HOME/llm-stack-m2
TRAIN=$HOME/llm-stack/training
PY=$HOME/llm-stack/.venv/bin/python
export PYTHONPATH=$ROOT/training/cs336-basics:$ROOT/training:$ROOT/serving
CK=$TRAIN/checkpoints_190m_fwe/checkpoint_epoch_0001.pt
CFG=$TRAIN/configs_190m/fwe.json
PROMPTS=$TRAIN/data/pretrain190m/wiki_valid.npy
OUT=$ROOT/demo_out
mkdir -p $OUT
cd $ROOT/serving

guard() {
  if pgrep -f "benchmark_lm_matrix.py" >/dev/null || pgrep -f "run_video1_bench.sh" >/dev/null; then
    echo "ABORT: the video-1 benchmark is still running -- sharing the GPU would corrupt both"; exit 1
  fi
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1500{n++} END{print n+0}')
  [[ $busy -eq 0 ]] || { echo "ABORT: $busy GPU(s) busy"; exit 1; }
  echo "guard ok"
}

rec() {  # rec <mode> <name> [extra args]
  local mode=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=0 $PY benchmarks/record_demo.py --config $CFG --checkpoint $CK --prompts-npy $PROMPTS \
    --device cuda --attn naive --mode $mode --arm-name $name --out $OUT/demo_$name.json \
    --n-requests 16 --concurrency 8 --new-tokens 64 --prompt-len 256 900 "$@"
}

case ${1:?stage: gate smoke record} in
gate)
  guard
  # The checkpoint was trained and evaluated with the clean tree: fwe val loss 3.3125 over 200 batches.
  # Evaluating it through THIS tree (M2 changes to lm.py / attention / rope) must give the same number.
  CUDA_VISIBLE_DEVICES=0 $PY $ROOT/training/cs336_systems/experiments/eval_checkpoint.py --config $CFG --checkpoint $CK \
    --stream fwe=$TRAIN/data/pretrain190m/fwe_valid.npy --batches 200 --json-out $OUT/gate_fwe.json
  $PY - <<PYEOF
import json
a = json.load(open("$OUT/gate_fwe.json"))["streams"]["fwe"]["loss"]
b = json.load(open("$TRAIN/artifacts/190m/fwe_streams.json"))["streams"]["fwe"]["loss"]
print(f"M2 tree loss {a:.5f}  training tree loss {b:.5f}  diff {abs(a-b):.2e}")
assert abs(a - b) < 5e-3, "the staged tree does not compute the same model as the one that trained/evaluated the checkpoint"
print("GATE OK")
PYEOF
  ;;
smoke)
  guard
  for mode in seq cb; do
    CUDA_VISIBLE_DEVICES=0 $PY benchmarks/record_demo.py --preset small --synthetic --device cuda --attn naive --mode $mode \
      --arm-name smoke_$mode --out $OUT/smoke_$mode.json --n-requests 6 --concurrency 4 --new-tokens 12 --prompt-len 16 64
  done
  $PY benchmarks/record_demo.py compare $OUT/smoke_seq.json $OUT/smoke_cb.json
  ;;
record)
  guard
  for rep in 1 2 3; do   # alternate so drift in clocks / temperature hits both arms alike
    rec seq w1_r$rep
    rec cb  w2_r$rep
    $PY benchmarks/record_demo.py compare $OUT/demo_w1_r$rep.json $OUT/demo_w2_r$rep.json
  done
  ;;
*) echo "unknown stage"; exit 1 ;;
esac
