# Profiling runtime
```
uv run python profiling.py \
    --WARM_UP_ITER 10\
    --PROFILE_ITER 10\
    --TRAIN_PATH ./cs336-basics/data/tokenized/ts_train.npy \
    --VAL_PATH ./cs336-basics/data/tokenized/ts_valid.npy \
    --VOCAB_PATH ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --MERGES_PATH ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl \
    --TR_BAT_SIZE 4 \
    --CONTEXT_LENGTH 256 \
    --VOCAB_SIZE 10000 \
    --DEVICE "cuda" \
```


```
uv run nsys profile -o result python profiling.py
```

```
source .venv/bin/activate

nsys profile -t cuda,nvtx -o my_profile_report --force-overwrite true python profiling.py \
    --WARM_UP_ITER 10\
    --PROFILE_ITER 10\
    --TRAIN_PATH ./cs336-basics/data/tokenized/ts_train.npy \
    --VAL_PATH ./cs336-basics/data/tokenized/ts_valid.npy \
    --VOCAB_PATH ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --MERGES_PATH ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl \
    --TR_BAT_SIZE 4 \
    --CONTEXT_LENGTH 256 \
    --VOCAB_SIZE 10000 \
    --DEVICE "cuda" \
    --CAST_DTYPE "bfloat16" \
    --MEMORY_PROFILE True

```