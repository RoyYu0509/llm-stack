```
uv run python cs336_systems/Parallelization/DDP/DDP_llm.py \
  --TRAIN_PATH cs336-basics/data/tokenized/ts_train.npy \
  --CONTEXT_LENGTH 128 \
  --EPOCHES 3 \
  --TR_BAT_SIZE 16 \
  --DTYPE "float32" \
  --ATTN_KERNEL "MyTriton"
  ```