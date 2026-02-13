```
source .venv/bin/activate

nsys profile -t cuda,nvtx -o ./profiling_attention_compile==True/my_profile_report --force-overwrite true python cs336_systems/benchmark_attention.py \
    --DTYPE "float32" \
    --PROFILE_FORWARD_MEMORY True \
    --PROFILE_BACKWARD_MEMORY True \
    --COMPILED


nsys profile -t cuda,nvtx -o ./profiling_attention_compile==False/my_profile_report --force-overwrite true python cs336_systems/benchmark_attention.py \
    --DTYPE "float32" \
    --PROFILE_FORWARD_MEMORY True \
    --PROFILE_BACKWARD_MEMORY True \
```


# Benchmarking Attention Kernels
** HEAD_DIM /  NUM_HEADS must be power of 2**
```
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
