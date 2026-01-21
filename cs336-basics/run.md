# Train the tokenizer
uv run python ./cs336-basics/cs336_basics/build_tokenizer.py \
    --input ./cs336-basics/data/ts.txt \
    --vocab-size 10000 \
    --special-tokens "<|endoftext|>" \
    --num-processes 8 \
    --vocab-output ./cs336-basics/cs336_basics/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-output ./cs336-basics/cs336_basics/bpe_tokenizer/merges_seq.pkl

# Build the NumPy data from the raw text (train + valid)
```
uv run python ./cs336-basics/cs336_basics/build_dataset.py \
    --size 5000000 \
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

# Train the LM using the NumPy Data
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
    --DEVICE "mps" \
    --COMPILE \
    --EVAL_INTERVAL 100 \
    --SAVE_INTERVAL 200 
```


# Text Generation
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
