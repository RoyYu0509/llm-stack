# 1. Train the tokenizer
uv run python ./cs336-basics/cs336_basics/build_tokenizer.py \
    --input ./cs336-basics/data/ts.txt \
    --vocab-size 10000 \
    --special-tokens "<|endoftext|>" \
    --num-processes 8 \
    --vocab-output ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-output ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl

# 2. Build the NumPy data from the raw text (train + valid)
```
uv run python ./cs336-basics/cs336_basics/build_dataset.py \
    --text-path  ./cs336-basics/data/ts.txt \
    --vocab-path ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl \
    --out ./cs336-basics/data/tokenized/ts_train.npy \
    --num-workers 10

uv run python ./cs336-basics/cs336_basics/build_dataset.py \
    --text-path ./cs336-basics/data/test_ts.txt \
    --vocab-path ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl \
    --out ./cs336-basics/data/tokenized/ts_valid.npy \
    --num-workers 10

uv run python ./cs336-basics/cs336_basics/build_dataset.py \
    --text-path ./cs336-basics/data/test_ts.txt \
    --vocab-path ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl \
    --out ./cs336-basics/data/tokenized/ts_test.npy \
    --num-workers 10
```

# 3. Train the LM using the NumPy Data
```
uv run python ./cs336-basics/cs336_basics/trainer.py \
    --TRAIN_PATH  ./cs336-basics/data/tokenized/ts_train.npy \
    --VAL_PATH  ./cs336-basics/data/tokenized/ts_valid.npy \
    --VOCAB_PATH ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --MERGES_PATH ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl \
    --TR_BAT_SIZE 32 \
    --VAL_BAT_SIZE 32 \
    --VAL_SAMP_SIZE 50\
    --CONTEXT_LENGTH 256 \
    --VOCAB_SIZE 10000 \
    --NUM_LAYERS 4 \
    --D_MODEL 512 \
    --NUM_HEADS 16 \
    --D_FF 1344 \
    --ROPE_THETA 10000 \
    --LR 6e-4\
    --WARMUP_ITERS 1500 \
    --MAX_ITERS 5000 \
    --EPOCHES 5000 \
    --WANDB_PROJECT "Train_Transformer_LM" \
    --DEVICE "cuda" \
    --COMPILE \
    --EVAL_INTERVAL 100 \
    --SAVE_INTERVAL 200
```

# 3b. Train with custom checkpoint cadence (checkpointing_every)
```
# Save every 500 steps instead of SAVE_INTERVAL=200:
uv run python ./cs336-basics/cs336_basics/trainer.py \
    --TRAIN_PATH  ./cs336-basics/data/tokenized/ts_train.npy \
    --VAL_PATH  ./cs336-basics/data/tokenized/ts_valid.npy \
    --VOCAB_PATH ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --MERGES_PATH ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl \
    --EPOCHES 5000 \
    --WANDB_PROJECT "Train_Transformer_LM" \
    --DEVICE "cuda" \
    --COMPILE \
    --CHECKPOINTING_EVERY 500

# Disable periodic checkpoints, only save at the final iteration:
# --CHECKPOINTING_EVERY 0
```


# 4. Text Generation
```
uv run python ./cs336-basics/cs336_basics/text_gen.py \
    --model-checkpoint ./cs336-basics/artifacts/iter_4999-loss_10.660388946533203.pt \
    --input-text "Once, there were" \
    --max-new-tokens 500 \
    --temperature 0.75 \
    --top-p 0.9 \
    --device "cuda" \
    --dtype "float32" \
    --vocab-path ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl \
    --vocab-size 10000 \
    --context-length 256 \
    --num-layers 4 \
    --d-model 512 \
    --num-heads 16 \
    --d-ff 1344 \
    --rope-theta 10000
```

# 5. End-to-End Pipeline (download + tokenize + train, all in one)
```
# Default single-GPU (uses default_pipeline_config.json)
uv run python cs336_systems/experiments/run_pipeline.py \
    --config cs336_systems/experiments/default_pipeline_config.json

# With FlashAttention Triton + FlashDDP
uv run python cs336_systems/experiments/run_pipeline.py \
    --config cs336_systems/experiments/default_pipeline_config.json \
    --attention_kernel flash_attention_triton \
    --ddp_wrapper flashddp

# Skip data stages, override epochs
uv run python cs336_systems/experiments/run_pipeline.py \
    --config cs336_systems/experiments/default_pipeline_config.json \
    --skip_data \
    --override training.epochs=1000
```

# 6. Benchmark: LM Training Matrix (kernel × DDP)
```
uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
    --train_path cs336-basics/data/tokenized/ts_train.npy \
    --val_path   cs336-basics/data/tokenized/ts_valid.npy \
    --epochs 3 --tr_batch_size 8
# → artifacts/lm_matrix_results.csv, lm_matrix_time.png, lm_matrix_memory.png, lm_matrix_report.md
```

# 7. Benchmark: Attention Forward Sweep (Small → XXL)
```
uv run python cs336_systems/experiments/benchmark_attention_sweep.py
# → artifacts/attention_sweep_results.csv, attention_sweep_forward.png, attention_sweep_heatmap.png

# Custom rectangular tiers:
uv run python cs336_systems/experiments/benchmark_attention_sweep.py \
    --custom_tiers 128:64 256:128 512:256 1024:512
```
