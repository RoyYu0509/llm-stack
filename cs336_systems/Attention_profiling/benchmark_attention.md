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
```
python /Users/yifanyu/Desktop/llm_proj/SystemLM/cs336_systems/Attention_profiling/benchmark_attention.py \
  --DEVICE cpu \
  --DTYPE float32 \
  --BATCH_SIZE 8 \
  --NUM_HEADS 32 \
  --SEQ_LEN 512 \
  --HEAD_DIM 256 \
  --WARMUP_ITER 10 \
  --PROFILE_ITER 10 \
  --IS_CAUSAL \
  --SEED 0

```
