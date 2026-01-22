```
nsys profile -t cuda,nvtx -o my_profile_report --force-overwrite true python cs336_systems/benchmark_attention.py \
    --DTYPE "float32"
```
