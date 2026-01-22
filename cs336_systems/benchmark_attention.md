```
nsys profile -t cuda,nvtx -o ./profiling_attention/my_profile_report --force-overwrite true python cs336_systems/benchmark_attention.py \
    --DTYPE "float32" \
    --PROFILE_FORWARD_MEMORY True\
    --PROFILE_BACKWARD_MEMORY True
```
