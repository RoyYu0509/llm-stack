# Train the tokenizer
uv run python src/build_tokenizer.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 \
    --special-tokens "<|endoftext|>" \
    --num-processes 8 \
    --vocab-output src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-output src/bpe_tokenizer/merges_seq.pkl

# Build the NumPy data from the raw text (train + valid)
```
uv run python src/build_dataset.py \
    --size 5000000 \
    --text-path data/TinyStoriesV2-GPT4-train.txt \
    --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path src/bpe_tokenizer/merges_seq.pkl \
    --out data/tokenized/TS_train_tokens.npy \
    --num-workers 10

uv run python src/build_dataset.py \
    --text-path data/TinyStoriesV2-GPT4-valid.txt \
    --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path src/bpe_tokenizer/merges_seq.pkl \
    --out data/tokenized/TS_val_tokens.npy \
    --num-workers 10

uv run python src/build_dataset.py \
    --text-path data/owt_train.txt \
    --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path src/bpe_tokenizer/merges_seq.pkl \
    --out data/tokenized/OWT_train_tokens.npy \
    --num-workers 10

uv run python src/build_dataset.py \
    --text-path data/owt_valid.txt \
    --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path src/bpe_tokenizer/merges_seq.pkl \
    --out data/tokenized/OWT_train_tokens.npy \
    --num-workers 10
```

# Train the LM using the NumPy Data
```
uv run python src/trainer.py \
    --TRAIN_PATH data/tokenized/TS_train_tokens.npy \
    --VAL_PATH data/tokenized/TS_val_tokens.npy \
    --VOCAB_PATH src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --MERGES_PATH src/bpe_tokenizer/merges_seq.pkl \
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
    --SAVE_INTERVAL 200 \  
```


# Text Generation
```
uv run python src/text_gen.py \
    --model-checkpoint checkpoints/lr-0.0006-beta1-0.9-beta2-0.999/iter_4400-loss_10.617208480834961.pt \
    --input-text "Once, there were" \
    --max-new-tokens 500 \
    --temperature 0.75 \
    --top-p 0.9 \
    --device "mps" \
    --dtype "float32" \
    --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path src/bpe_tokenizer/merges_seq.pkl \
    --vocab-size 10000 \
    --context-length 256 \
    --num-layers 4 \
    --d-model 512 \
    --num-heads 16 \
    --d-ff 1344 \
    --rope-theta 10000
```
