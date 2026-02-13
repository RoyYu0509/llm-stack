# Profiling lm with different kernels (Memory + Runtime) + Nsight Profiling
```
source .venv/bin/activate

uv run nsys profile --wait primary -o cs336_systems/Attention_profiling/profiling_lm/result \
  python cs336_systems/Attention_profiling/profile_lm_kernels.py \
  --WARM_UP_ITER 10 \
  --PROFILE_ITER 5 \
  --TRAIN_PATH ./cs336-basics/data/tokenized/ts_train.npy \
  --DEVICE cuda \
  --DTYPE float32 \
  --ALL_KERNEL \
  --TORCH_MEM_PROF

```