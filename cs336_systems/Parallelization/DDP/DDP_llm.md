```
uv run cs336_systems/Parallelization/DDP/DDP_runner.py \
  --TRAIN_PATH cs336-basics/data/tokenized/ts_train.npy \
  --VAL_PATH cs336-basics/data/tokenized/ts_val.npy \
  --CONTEXT_LENGTH 256 \
  --EPOCHES 3 \
  --TR_BAT_SIZE 4 \
  --VAL_BAT_SIZE 4 \
  --DTYPE "float32" \
  --ATTN_KERNEL "CompTorch"
  ```