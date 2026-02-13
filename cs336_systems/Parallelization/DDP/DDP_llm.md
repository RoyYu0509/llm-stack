```
uv run python cs336_systems/Parallelization/DDP/DDP_llm.py \
  --TRAIN_PATH ./cs336-basics/data/tokenized/ts_train.npy \
  --CONTEXT_LENGTH 128 \
  --EPOCHS 3 \
  --BATCH_SIZE 64 \
  --DEVICE cuda \
  --DTYPE float32
  ```