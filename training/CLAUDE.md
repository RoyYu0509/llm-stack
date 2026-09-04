# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Transformer Language Model systems engineering project focused on two major optimizations:
1. **FlashAttention** — custom Triton GPU kernels for memory-efficient attention
2. **Distributed Training (DDP)** — bucketed overlapping gradient synchronization

The project depends on `cs336-basics` (a sibling package in `cs336-basics/`) for the base Transformer LM, tokenizer, and training loop.

## Commands

**Setup:**
```bash
uv sync
source .venv/bin/activate
```

**Testing:**
```bash
uv run pytest                                    # all tests
uv run pytest tests/test_attention.py            # single test file
uv run pytest tests/test_ddp.py -v               # verbose
uv run pytest -v ./tests --junitxml=test_results.xml
```

**End-to-end pipeline:**
```bash
uv run python cs336_systems/experiments/run_pipeline.py \
  --config cs336_systems/experiments/default_pipeline_config.json \
  --attention_kernel flash_attention_triton \
  --ddp_wrapper flashddp \
  --skip_data   # skip tokenization if data already prepared
```

**Benchmarking:**
```bash
uv run python cs336_systems/experiments/benchmark_attention_sweep.py
uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
  --config cs336_systems/experiments/default_pipeline_config.json \
  --train_path data/tokenized/ts_train.npy \
  --val_path data/tokenized/ts_valid.npy \
  --timed_epochs 3 \
  --kernels flash_attention_triton \
  --wrappers "Local No DDP" "Naive DDP" "Pytorch DDP" "Bucketed Overlapping DDP"
```

**Useful env vars for debugging:**
```bash
DEBUG_DDP=1 uv run python ...
TRITON_PRINT_AUTOTUNING=1 uv run python ...
CS336_DISABLE_TF32=1 uv run python ...   # TF32 is on by default on Ampere; this turns it
                                          # off so the speedup can be measured, not assumed
```

**Resuming vs starting from pretrained weights** — two different config keys, and the
distinction matters:
- `resume_from`: continue an interrupted run (weights + optimizer moments + `global_step`,
  so the LR schedule picks up where it stopped). Requires a checkpoint written after the
  schema fix; older ones store a single ambiguous `iter` field and are rejected explicitly.
- `init_from`: start a **new** run from pretrained weights only — fresh optimizer state and
  a fresh LR schedule. This is what fine-tuning wants, since the pretraining run ended with
  its cosine decayed to the minimum.

## Architecture

### `cs336-basics/` — Base LM package
- `lm.py`: `TransformerLM` — pre-norm Transformer with RoPE, SwiGLU FFN, causal masking
- `transfromer/`: individual components (attention, FFN, RMSNorm, RoPE, embedding)
- `trainer.py` / `lm_trainer.py`: training loop with custom AdamW and LR scheduling
- `build_dataset.py`: parallel tokenization to NumPy `.npy` arrays
- `bpe_tokenizer/`: BPE tokenizer training

### `cs336_systems/` — Systems optimizations
- **`FlashAttention/`**: three attention implementations selectable at runtime
  - `flash_attention_torch_naive.py` — O(N²) memory baseline
  - `flash_attention_torch_vectorized.py` — vectorized PyTorch
  - `flash_attention_triton.py` — autotuned Triton kernel (6.54× faster at seq_len=8192; only one that handles seq_len=16384)

- **`Parallelization/DDP/`**: PyTorch DDP baseline
  - `naiveDDP.py`: per-parameter all-reduce
  - `DDP_runner.py`: training wrapper

- **`Parallelization/FlashDDP/`**: custom bucketed overlapping DDP
  - `FlashDDP.py`: async DDP base **and** `DDPOverlapBucketed` -- the class that is actually
    wired up and produced the benchmarked numbers (85.9% scaling efficiency, +9.4% over naive,
    on par with PyTorch official DDP). Source: `artifacts/lm_matrix_table_flash_attention_triton.png`
  - `BucketedOverlapDDP.py`: a separate bucketed/overlapped implementation that is **not** wired
    into the benchmark path -- do not attribute the numbers above to this file
  - `FlashDDP_runner.py`: training wrapper

- **`data_harvest/`**: corpus construction for the real-data pretraining run
  - `build_mixed_corpus.py`: streams C4 / Wikipedia / Alpaca from HuggingFace into one
    70/20/10 mixed corpus (`data/mixed_corpus/`) plus a `manifest.json` recording the
    ratios actually achieved and Alpaca's oversample factor
  - `build_sft_corpus.py`: Alpaca-only corpus for instruction fine-tuning, plus an
    `alpaca_replay` variant mixing in 25% pretraining data to measure forgetting
  - `build_domain_valid.py`: three separate validation streams (web / wiki / instruction),
    so per-domain effects of the mixture can be measured rather than assumed

- **`experiments/`**: orchestration and benchmarking
  - `run_pipeline.py`: full pipeline (download → tokenize → train) with JSON config + CLI overrides
  - `benchmark_attention_sweep.py`: sweeps seq_len 128→16384, outputs to `artifacts/`
  - `benchmark_lm_matrix.py`: kernel × DDP strategy grid, outputs to `artifacts/`
  - `eval_checkpoint.py`: score one checkpoint against any number of `.npy` token streams.
    Uses a fixed seed so every checkpoint sees **identical batches** — the comparison is
    paired, which is what makes 0.05-scale differences between runs detectable
  - `sample_from_checkpoint.py`: qualitative generation. Use this rather than
    `cs336_basics/text_gen.py`, which rebuilds a custom BPE tokenizer this project no
    longer trains (see `real_pretrain_config.json`'s `_NOTE_tokenizer_swap`)
  - `bench_serving.py`: TTFT / decode latency / throughput through `llm-serving`'s real
    `LMEngine` path; `--modes recompute cache` measures the KV cache's benefit directly
  - `kv_cache_gate.py`: correctness gate for the inference KV cache — cached decode must
    be token-identical to recompute before any cached number is reported
  - `make_night_report.py`: collapses eval JSONs, per-arm logs and benchmark artifacts
    into a single `artifacts/NIGHT_REPORT.md`
  - `default_pipeline_config.json`: 12-layer, d_model=768, 12-head model; batch_size=8, lr=6e-4
  - `real_pretrain_config.json`: the real 0.19B pretraining run. Its `_NOTE_*` fields record
    the failures that shaped it (FFN init bug, OOM, tokenizer swap, W&B artifact hang) —
    read them before changing model size, batch size, or LR

**Inference KV cache**: `TransformerLM.forward_with_cache(x, past_kv)` (plus
`forward_with_cache` on `PreNormTransformer` / `MultiHeadsAttention`) is an **additive**
inference-only path — `forward` is untouched and remains the training path. Two invariants
it must preserve: decode passes `is_causal=False` (the causal mask is built as
`tril(seq_q, seq_k)` and would expose only key 0 to a single query), and RoPE is applied at
absolute positions *before* keys enter the cache. `tests/test_kv_cache.py` enforces both.

### Data flow
```
Raw text → BPE tokenizer → .npy arrays → DataLoader
  → TransformerLM (pluggable attention kernel)
  → Loss → Backward (optionally overlapped DDP all-reduce)
  → AdamW update
```

### Tests
- `tests/adapters.py`: bridge between test harness and implementations
- `tests/conftest.py`: `NumpySnapshot` utility for array snapshot testing
- Snapshot files stored in `tests/_snapshots/`; use `--snapshot-exact` for strict matching

## Key Design Patterns

- **Pluggable kernels**: attention implementation is selected by string name at runtime (passed via `--attention_kernel`)
- **Pluggable DDP**: wrapper strategy is selected by string name (`--ddp_wrapper`)
- **Config override pattern**: `default_pipeline_config.json` provides defaults; CLI `--override key=value` patches specific fields
- **Artifact outputs**: all benchmark results (CSV, PNG, markdown) go to `artifacts/`
