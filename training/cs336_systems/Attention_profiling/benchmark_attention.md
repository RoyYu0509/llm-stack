# Benchmarking Attention Kernels
** HEAD_DIM /  NUM_HEADS must be power of 2**
```
source .venv/bin/activate

python cs336_systems/Attention_profiling/benchmark_attention.py \
  --DEVICE cuda \
  --DTYPE float32 \
  --BATCH_SIZE 16 \
  --NUM_HEADS 24 \
  --SEQ_LEN 512 \
  --HEAD_DIM 512 \
  --WARMUP_ITER 30 \
  --PROFILE_ITER 10 \
  --IS_CAUSAL \
  --SEED 0
```
