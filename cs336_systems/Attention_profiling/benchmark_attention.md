# Benchmarking Attention Kernels
** HEAD_DIM /  NUM_HEADS must be power of 2**
```
source .venv/bin/activate

python cs336_systems/Attention_profiling/benchmark_attention.py \
  --DEVICE cuda \
  --DTYPE float32 \
  --BATCH_SIZE 2 \
  --NUM_HEADS 256 \
  --SEQ_LEN 518 \
  --HEAD_DIM 756 \
  --WARMUP_ITER 20 \
  --PROFILE_ITER 10 \
  --IS_CAUSAL \
  --SEED 0
```
