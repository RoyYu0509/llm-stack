# CS336 Spring 2025 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.




CS336_Basics — core LM + training primitives
- BPE/tokenizer training pipeline
- Built a Transformer LM module (TransformerLM) that wires together embedding, multiple transformer blocks, final norm, and LM head projection.
- Implemented a pre-norm Transformer block
- Implemented attention stack pieces
- Implemented core training

CS336_Systems — FlashAttention2 + bucketed/overlapped DDP
- FlashAttention-2 (PyTorch) implementation exposed via an autograd Function adapter path.
- FlashAttention-2 (Triton) implementation (kernel code + autotune configs present).
- Vectorized Torch attention implementation
- Naïve DDP (all-reduce per-parameter gradients).
- Naïve Bucketed DDP (Bucketed all parameters and send them at once).
- Bucketed + overlapped DDP container class: you have a DDPOverlapBucketed implementation that buckets params, registers per-param grad hooks, and async all-reduces buckets.

------------

## Experiments & Benchmarks
```bash
source .venv/bin/activate
```

### Pipeline Script

A single end-to-end pipeline script that downloads TinyStories, trains a BPE tokenizer, tokenizes the dataset, and trains a Transformer LM — all configurable from one JSON config file:

```bash
# Default single-GPU run
uv run python cs336_systems/experiments/run_pipeline.py \
    --config cs336_systems/experiments/default_pipeline_config.json

# Override attention kernel & DDP wrapper from CLI
uv run python cs336_systems/experiments/run_pipeline.py \
    --config cs336_systems/experiments/default_pipeline_config.json \
    --attention_kernel flash_attention_triton \
    --ddp_wrapper flashddp

# Skip data prep if .npy files already exist
uv run python cs336_systems/experiments/run_pipeline.py \
    --config cs336_systems/experiments/default_pipeline_config.json \
    --skip_data
```

**Supported attention kernels:** `scaled_dot_prod_attention`, `vectorized_torch`, `flash_attention_triton`

**Supported DDP wrappers:** `none` (single-GPU), `naive` (all-reduce per-param), `flashddp` (bucketed + overlapped)

### Checkpoint Cadence: `checkpointing_interval`

A new first-class hyperparameter that controls checkpoint save frequency:

| Flag                | Behavior |
|---------------------|----------|
| `> 0`              | Saves every N steps |
| `<= 0`             | Disables periodic checkpoints; only saves at the final iteration |
| Final iteration     | Always saved regardless of cadence |

Available in both `trainer.py` (CLI) and `lm_trainer.py` (Python API), and in the pipeline config under `checkpointing.checkpointing_interval`.

### LM Training Benchmark Matrix

Benchmarks short training runs across **3 attention kernels × 3 DDP wrappers** (single-GPU + naive + FlashDDP):

```bash
uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
    --train_path data/tokenized/ts_train.npy \
    --val_path   data/tokenized/ts_valid.npy \
    --epochs 5 --tr_batch_size 8 --context_length 256 \
    --vocab_size 10_000 \
    --d_model 512 --d_ff 4096 --num_layers 24 --num_heads 16 \
    --kernels scaled_dot_prod_attention vectorized_torch flash_attention_triton \
    --wrappers none naive flashddp torch_ddp    
```

Quick single-kernel test (e.g., just FlashAttention Triton):

```bash
uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
    --train_path data/tokenized/ts_train.npy \
    --val_path   data/tokenized/ts_valid.npy \
    --epochs 3 --tr_batch_size 16 --context_length 256 \
    --kernels flash_attention_triton
```

Only DDP wrappers (skip single-GPU baseline):

```bash
uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
    --train_path data/tokenized/ts_train.npy \
    --val_path   data/tokenized/ts_valid.npy \
    --epochs 3 --tr_batch_size 16 --context_length 256 \
    --wrappers naive flashddp torch_ddp
```

Outputs: `artifacts/lm_matrix_results.csv`, `artifacts/lm_matrix_time.png`, `artifacts/lm_matrix_memory.png`, `artifacts/lm_matrix_throughput.png`, `artifacts/lm_matrix_report.md`

### Attention Forward Benchmark Sweep

Pure forward-only attention benchmarks from Small (128) to XXL (2048), including rectangular (Q_N ≠ K_N) patterns:

```bash
uv run python cs336_systems/experiments/benchmark_attention_sweep.py

# Custom tiers (must be powers of 2)
uv run python cs336_systems/experiments/benchmark_attention_sweep.py \
    --custom_tiers 128:64 256:128 512:256 1024:512 \
    --warmup 10 --iters 50
```

Outputs: `artifacts/attention_sweep_results.csv`, `artifacts/attention_sweep_forward.png`, `artifacts/attention_sweep_heatmap.png`, `artifacts/attention_sweep_scaling.png`, `artifacts/attention_sweep_report.md`

### Visualization Summary

1. **Training loss curve** — WandB dashboard showing training/validation loss over iterations
2. **Attention kernel forward time** — Grouped bar chart comparing all kernels across sequence-length tiers
3. **LM training throughput** — Tokens/sec across kernel × DDP combinations
4. **DDP speedup** — Wall time per epoch: single-GPU → naive DDP → FlashDDP (bucketed + overlapped)


