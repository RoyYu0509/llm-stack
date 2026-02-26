# Transformer Language Model Systems Project

Full GPT-style decoder-only language model and systems stack built from scratch with Python and PyTorch (project period: Nov 2025 - Feb 2026, NUS).

## Project Focus

This repository centers on building and optimizing a full Transformer LM pipeline:

- End-to-end LM implementation and training stack
- Long-context attention optimization with Triton FlashAttention
- Distributed multi-GPU training from first principles
- Reproducible experiment pipelines and benchmark reporting

## What I Implemented

| Area | Implementation |
|---|---|
| Tokenization | Parallel BPE tokenizer training + dataset tokenization to `.npy` |
| Model | Decoder-only `TransformerLM` with RoPE, pre-norm residual blocks, RMSNorm, SwiGLU FFN |
| Optimization | Custom AdamW optimizer, LR scheduling, grad clipping |
| Inference | Nucleus (`top-p`) sampling text generation |
| Attention Kernels | `scaled_dot_prod_attention`, `vectorized_torch`, Triton `flash_attention_triton` (autotuned) with reference kernel validation |
| Distributed Training | Naive per-parameter DDP, Torch DDP baseline, custom bucketed-overlapped `DDPOverlapBucketed` (`flashddp`) |
| Profiling | NVIDIA Nsight Systems + PyTorch profiler workflow for kernel/runtime bottleneck analysis |
| Experiment Infra | Unified JSON/CLI pipeline, checkpointing, optional model compilation with `torch.compile`, W&B logging |

## Key Results

### 1. Attention Forward Performance (Long Context)

Representative tier: `Q=K=8192, head_dim=64, heads=12, batch=1`

| Kernel | Avg Forward Time (ms) | Relative to Triton Flash |
|---|---:|---:|
| `scaled_dot_prod_attention` | 116.65 | 6.40x slower |
| `vectorized_torch` | 97.83 | 5.36x slower |
| `vectorized_torch_compiled` | 28.45 | 1.56x slower |
| `flash_attention_triton` | **18.24** | 1.00x |

Longer-context behavior (`Q=K=16384`, same tier family):

- `flash_attention_triton`: **72.32 ms**
- `vectorized_torch`: OOM in this sweep
- `scaled_dot_prod_attention`: OOM in this sweep

This matches the long-context objective in the resume entry: major forward-pass latency reduction and practical 16K-context execution where naive/vectorized baselines fail in the same benchmark family.

### 2. Multi-GPU Training Throughput (3x4 Matrix)

From `artifacts/lm_matrix_results.csv` using only `flash_attention_triton`:

| DDP Method\Attention Kernel (`flash_attention_triton`) | tok/s | Scaling Efficiency | memory Overhead |
|---|---:|---:|---:|
| `none` (1 GPU) | 2791.4 | N/A (no scaling) | 0 MB (0.0%) |
| `naive` (2 GPU) | 3868.5 | 69.3% | +0.0 MB (0.0%) |
| `flashddp` (2 GPU) | **4565.6** | **81.8%** | +719.1 MB (9.6%) |
| `torch_ddp` (2 GPU) | 4566.0 | 81.8% | +720.7 MB (9.6%) |

`Scaling Efficiency` for 2-GPU rows is measured against ideal 2x throughput over the same-kernel `none` baseline. `memory Overhead` is per-GPU peak memory increase vs the same-kernel `none` baseline.

Summary:

- Up to **81.8% (~82%)** scaling efficiency on 2 GPUs
- **+18.0%** throughput (`flashddp` vs `naive`) for `flash_attention_triton`
- **<10%** per-GPU memory overhead for overlapped bucketing

## Figures

### Attention Sweep

![Attention Forward Benchmark](artifacts/attention_sweep_forward.png)
![Attention Heatmap](artifacts/attention_sweep_heatmap.png)

### LM Matrix Benchmarks

![LM Throughput](artifacts/lm_matrix_throughput.png)
![LM Time per Epoch](artifacts/lm_matrix_time.png)
![LM Peak Memory](artifacts/lm_matrix_memory.png)

## Reproduce Main Experiments

### Environment

```bash
uv sync
source .venv/bin/activate
export WANDB_API_KEY=...
```

### End-to-End Pipeline

```bash
# Default single-GPU run
uv run python cs336_systems/experiments/run_pipeline.py \
  --config cs336_systems/experiments/default_pipeline_config.json

# Override kernel + DDP wrapper
uv run python cs336_systems/experiments/run_pipeline.py \
  --config cs336_systems/experiments/default_pipeline_config.json \
  --attention_kernel flash_attention_triton \
  --ddp_wrapper flashddp \
  --skip_data
```

Supported attention kernels:

- `scaled_dot_prod_attention`
- `vectorized_torch`
- `flash_attention_triton`

Supported wrappers:

- `none`
- `naive`
- `flashddp`

### LM Kernel x DDP Matrix

```bash
uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
  --train_path data/tokenized/ts_train.npy \
  --val_path data/tokenized/ts_valid.npy \
  --epochs 5 --tr_batch_size 8 --context_length 256 \
  --vocab_size 10000 \
  --d_model 512 --d_ff 4096 --num_layers 24 --num_heads 16 \
  --kernels scaled_dot_prod_attention vectorized_torch flash_attention_triton \
  --wrappers none naive flashddp torch_ddp
```

Outputs:

- `artifacts/lm_matrix_results.csv`
- `artifacts/lm_matrix_report.md`
- `artifacts/lm_matrix_time.png`
- `artifacts/lm_matrix_memory.png`
- `artifacts/lm_matrix_throughput.png`

### Attention Forward Sweep

```bash
uv run python cs336_systems/experiments/benchmark_attention_sweep.py
```

Outputs:

- `artifacts/attention_sweep_results.csv`
- `artifacts/attention_sweep_report.md`
- `artifacts/attention_sweep_forward.png`
- `artifacts/attention_sweep_heatmap.png`
- `artifacts/attention_sweep_scaling.png`

## Repository Pointers

- `cs336-basics/cs336_basics`: core LM, tokenizer, optimizer, trainer
- `cs336_systems/FlashAttention`: Triton FlashAttention kernel + tuning scripts
- `cs336_systems/Parallelization/FlashDDP`: custom bucketed-overlapped DDP wrapper
- `cs336_systems/experiments`: pipeline + benchmark entrypoints
- `artifacts`: generated reports and figures

## References (Coursework Origin)

This project was developed on top of CS336 coursework scaffolding; assignment-specific details are archived here for reference only.

- CS336 systems handout: [`cs336_spring2025_assignment2_systems.pdf`](./cs336_spring2025_assignment2_systems.pdf)
- Assignment packaging script: `./test_and_make_submission.sh`
- Original assignment directory split: `cs336-basics` (assignment-1 baseline module) and `cs336_systems` (systems/optimization implementation)
