```
source .venv/bin/activate

uv run cs336_systems/Parallelization/FlashDDP/FlashDDP_runner.py \
  --TRAIN_PATH data/tokenized/ts_train.npy \
  --VAL_PATH data/tokenized/ts_valid.npy \
  --CONTEXT_LENGTH 256 \
  --NUM_LAYERS 24\
  --D_MODEL 512 \
  --D_FF 3072 \
  --NUM_HEADS 16 \
  --EPOCHES 5 \
  --WARMUP_EPOCHS 2 \
  --TR_BAT_SIZE 4 \
  --VAL_BAT_SIZE 4 \
  --DTYPE "float32" \
  --ATTN_KERNEL "MyTriton" \
  --BUCKET_SIZE_MB 10
  ```

git config --global user.email "royyu0509@gmail.com"
git config --global user.name "Yifan"