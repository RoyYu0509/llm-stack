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
# Baseline (default)
python cs336_systems/profiling/benchmark_attention.py

# Baseline Compiled
python cs336_systems/profiling/benchmark_attention.py --COMPILED

# My Triton implementation, compiled
python cs336_systems/profiling/benchmark_attention.py --MyTritonAttn 

# Vectorized Torch implementation
python cs336_systems/profiling/benchmark_attention.py --VecTorchAttn
```
