# Profiling runtime
```
source .venv/bin/activate

uv run python cs336_systems/Attention_profiling/profile_lm_kernels.py \
  --WARM_UP_ITER 10 \
  --PROFILE_ITER 100 \
  --TRAIN_PATH ./cs336-basics/data/tokenized/ts_train.npy \
  --TR_BAT_SIZE 4 \
  --CONTEXT_LENGTH 256 \
  --VOCAB_SIZE 10000 \
  --DEVICE mps \ 
  --DTYPE float32
```

# If Profile with Nsight & Profile torch Memory
```
nsys profile -t cuda,nvtx -o ./cs336_systems/Attention_profiling/profiling_lm_uncompiled/my_profile_report --force-overwrite true \
python cs336_systems/Attention_profiling/profile_lm_kernels.py \
  --WARM_UP_ITER 10 \
  --PROFILE_ITER 10 \
  --TRAIN_PATH ./cs336-basics/data/tokenized/ts_train.npy \
  --TR_BAT_SIZE 4 \
  --CONTEXT_LENGTH 256 \
  --VOCAB_SIZE 10000 \
  --DEVICE cuda \
  --DTYPE bfloat16 \
  --ONLY_KERNEL "MyTriton"
```

# Torch.compile v.s. not compile
```
uv run python cs336_systems/Attention_profiling/profile_lm_kernels.py \
  --WARM_UP_ITER 30 \
  --PROFILE_ITER 10 \
  --TRAIN_PATH ./cs336-basics/data/tokenized/ts_train.npy \
  --TR_BAT_SIZE 4 \
  --CONTEXT_LENGTH 256 \
  --VOCAB_SIZE 10000 \
  --DEVICE cuda \
  --DTYPE bfloat16 \
  --COMPILED

uv run python cs336_systems/Attention_profiling/profile_lm_kernels.py \
  --WARM_UP_ITER 30 \
  --PROFILE_ITER 10 \
  --TRAIN_PATH ./cs336-basics/data/tokenized/ts_train.npy \
  --TR_BAT_SIZE 4 \
  --CONTEXT_LENGTH 256 \
  --VOCAB_SIZE 10000 \
  --DEVICE cuda \
  --DTYPE bfloat16
```