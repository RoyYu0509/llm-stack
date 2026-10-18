#!/usr/bin/env bash
# Concurrency sweep for the inference demo: one request at a time (seq) vs continuous batching (cb),
# naive attention, one GPU, the freshly trained 190M weights. Run on the vast box from ~/llm-stack.
#
#   bash serving/benchmarks/run_cb_sweep.sh gate      # weights-only checkpoint reproduces the known wiki val loss
#   bash serving/benchmarks/run_cb_sweep.sh smoke     # tiny model on cuda, both modes
#   bash serving/benchmarks/run_cb_sweep.sh sweep     # C in 8 16 32, n = 4C requests, 3 alternating repeats
#   bash serving/benchmarks/run_cb_sweep.sh load      # 开环压测:泊松到达、变长请求,扫到达率;goodput / 尾延迟 vs 负载(约 50 分钟)
#   bash serving/benchmarks/run_cb_sweep.sh hw        # 同上但开前向计时:MFU / MBU / 前向时间占比 / 显存(约 10 分钟)
#   bash serving/benchmarks/run_cb_sweep.sh analyze   # 不占 GPU:把 sweep 的闭环录像按完整指标(p95/p99/TPOT/goodput)重算
#
# Why n = 4C: with n close to C the run is mostly ramp-up and ramp-down (and the serial prefills are a big
# share of the batched run), which caps the measurable speed-up. cb_benchmark.py uses the same rule.
set -euo pipefail
ROOT=$HOME/llm-stack
PY=$ROOT/.venv/bin/python
export PYTHONPATH=$ROOT/training/cs336-basics:$ROOT/training:$ROOT/serving
CK=$ROOT/training/ckpt_190m_fwe/model_only.pt
CFG=$ROOT/training/configs_190m/fwe.json
PROMPTS=$ROOT/training/data/pretrain190m/wiki_valid.npy
OUT=$ROOT/sweep_out
mkdir -p $OUT
cd $ROOT/serving

guard() {
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1500{n++} END{print n+0}')
  [[ $busy -eq 0 ]] || { echo "ABORT: GPU busy"; exit 1; }
  echo "guard ok"
}

rec() {  # rec <mode> <name> <n> <C>
  $PY benchmarks/record_demo.py --config $CFG --checkpoint $CK --prompts-npy $PROMPTS --device cuda --attn naive \
    --mode $1 --arm-name $2 --out $OUT/demo_$2.json --n-requests $3 --concurrency $4 --new-tokens 64 --prompt-len 256 900
}

case ${1:?stage: gate smoke sweep load hw analyze} in
gate)
  guard
  # wiki_valid loss of the trained checkpoint, 200 batches, measured on the training box: 3.7402058935165403
  $PY $ROOT/training/cs336_systems/experiments/eval_checkpoint.py --config $CFG --checkpoint $CK \
    --stream wiki=$PROMPTS --batches 200 --json-out $OUT/gate_wiki.json
  $PY - <<PYEOF
import json
a = json.load(open("$OUT/gate_wiki.json"))["streams"]["wiki"]["loss"]
b = 3.7402058935165403
print(f"this box {a:.5f}  training box {b:.5f}  diff {abs(a-b):.2e}")
assert abs(a - b) < 5e-3, "weights-only checkpoint or this code tree does not reproduce the trained model"
print("GATE OK")
PYEOF
  ;;
smoke)
  guard
  for m in seq cb; do
    $PY benchmarks/record_demo.py --preset small --synthetic --device cuda --attn naive --mode $m --arm-name smoke_$m \
      --out $OUT/smoke_$m.json --n-requests 6 --concurrency 4 --new-tokens 12 --prompt-len 16 64
  done
  $PY benchmarks/record_demo.py compare $OUT/smoke_seq.json $OUT/smoke_cb.json
  ;;
sweep)
  guard
  for rep in 1 2 3; do            # rep-major and alternating, so clock / temperature drift hits both arms alike
    for C in 8 16 32; do
      n=$((4 * C))
      rec seq c${C}_seq_r$rep $n $C
      rec cb  c${C}_cb_r$rep  $n $C
      $PY benchmarks/record_demo.py compare $OUT/demo_c${C}_seq_r$rep.json $OUT/demo_c${C}_cb_r$rep.json
    done
  done
  echo SWEEP_DONE
  ;;
load)
  guard
  # 到达率 = 一次一条的服务速率 μ × {0.5 1 2 4 8 16},脚本自己先量 μ。每档 60 s 的到达 + 最多 60 s 排空。
  $PY benchmarks/serving_bench.py run --config $CFG --checkpoint $CK --prompts-npy $PROMPTS --device cuda \
    --workload chat --duration 60 --drain 60 --repeats 2 --vram-gb 24 --out-dir $OUT
  ;;
hw)
  guard
  $PY benchmarks/serving_bench.py run --config $CFG --checkpoint $CK --prompts-npy $PROMPTS --device cuda \
    --workload chat --rate-mult 1 4 16 --duration 30 --drain 30 --repeats 1 --profile --vram-gb 24 \
    --out-dir $OUT/hw
  ;;
analyze)
  $PY benchmarks/serving_bench.py analyze $OUT
  ;;
*) echo "unknown stage"; exit 1 ;;
esac
