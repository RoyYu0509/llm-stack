```
source .venv/bin/activate

uv run cs336_systems/Parallelization/FlashDDP/FlashDDP_runner.py \
  --TRAIN_PATH cs336-basics/data/tokenized/ts_train.npy \
  --VAL_PATH cs336-basics/data/tokenized/ts_valid.npy \
  --CONTEXT_LENGTH 256 \
  --EPOCHES 5 \
  --WARMUP_EPOCHS 2 \
  --TR_BAT_SIZE 16 \
  --VAL_BAT_SIZE 16 \
  --DTYPE "float32" \
  --ATTN_KERNEL "MyTriton"
  ```

git config --global user.email "royyu0509@gmail.com"
git config --global user.name "Yifan"