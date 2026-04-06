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
```

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
  - `FlashDDP.py`: async DDP base
  - `BucketedOverlapDDP.py`: groups params into size-bounded buckets, overlaps all-reduce with backward pass (85.9% scaling efficiency, 9.4% throughput gain)
  - `FlashDDP_runner.py`: training wrapper

- **`experiments/`**: orchestration and benchmarking
  - `run_pipeline.py`: full pipeline (download → tokenize → train) with JSON config + CLI overrides
  - `benchmark_attention_sweep.py`: sweeps seq_len 128→16384, outputs to `artifacts/`
  - `benchmark_lm_matrix.py`: kernel × DDP strategy grid, outputs to `artifacts/`
  - `default_pipeline_config.json`: 12-layer, d_model=768, 12-head model; batch_size=8, lr=6e-4

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
