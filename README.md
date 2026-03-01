# Transformer Systems Engineering Project

I implemented the model and training pipeline from scratch and then added systems optimizations that measurably improved speed and scalability, using **PyTorch** and **Triton GPU programming**.

- **End-to-end LM stack** [`cs336_basics`](./cs336_basics)
  - Byte-level BPE tokenizer (parallel training)
  - Parallel dataset tokenization to NumPy arrays
  - Pre-norm Transformer LM
  - Custom AdamW + LR schedule + gradient clipping
  - Evaluation, WanDB logging & checkpointing, generation sampling.
  
- **Attention kernels** [`cs336_systems/FlashAttention`](./cs336_systems/FlashAttention)
  - Baseline scaled dot-product attention
  - Vectorized PyTorch attention baseline
  - Autotuned **Triton FlashAttention** integrated into the model interface

- **Distributed training** [`cs336_systems/Parallelization`](cs336_systems/Parallelization)
  - Naive per-parameter synchronization baseline
  - PyTorch DDP baseline
  - **Custom bucketed, overlapped DDP** wrapper (communication/computation overlap)

- **Experiment + profiling workflow** [`cs336_systems/experiments`](./cs336_systems/experiments) & [`cs336_systems/Attention_profiling`](./cs336_systems/Attention_profiling)
  - Benchmark drivers that emit **CSV + Markdown summaries**
  - Plots for timing, throughput, convergence, and memory behavior
  - Artifacts saved under [`artifacts/`](./artifacts/) for review and sharing

This repo shows both algorithm-level and systems-level engineering: I did not just train a model, I also optimized how efficiently the machine runs it.


## Achievements Summary

| Summary | Achievements | Implication |
|---|---|---|
| Flash Attention speed | At **sequence length 8,192**, Triton FlashAttention is **6.54×** faster than PyTorch vectorized attention (**27.20 ms** vs **177.82 ms**) | Faster attention directly reduces response/training latency |
| Flash Attention long-context feasibility | At **sequence length 16,384**, **only** Triton FlashAttention completes (**116.60 ms**); other kernels hit OOM | Longer context improves the model’s ability to handle long prompts |
| Multi-GPU training pipeline | On 2× GPUs, **Bucketed + Overlapped gradient DDP training** , reaching **85.9% scaling efficiency** with only **+3.3%** peak per-GPU memory overhead, and is **+9.4%** faster than ***Naive DDP*** (**25,336.7 → 27,719.1 tok/s**) | More tokens/sec without a big memory penalty → better utilization and scalability |

#### Environment Specifics

Benchmarks were run on a rented SSH VM with **2× RTX 3090**, dual **Xeon E5-2680 v4**, **193 GB RAM**.


## Dicussion of Benchmark Highlights
---
### 1) FlashAttention Performance

In the attention sweep (**head_dim=64, heads=12, batch=4**), Triton FlashAttention is fastest at every measured sequence length, **2~7×** faster than Compiled/Standard Attention kernels, and is the only kernel that completed **sequence length 16,384** in this benchmark family, while alternatives hit OOM at that length. 

The following plot shows the forward pass time for the different attention kernels across sequence lengths configs, where the x-axis is the sequence length (Q=K) and the y-axis is the forward pass time in milliseconds. We see that FlashAttention is significantly faster than the other kernels, and is the only one that can handle the longest sequence length without running out of memory.

![Attention Sweep Forward Time][fig-attn-sweep]

FlashAttention improves memory efficiency by fusing the attention computation and reducing intermediate activations to gain efficiency, and compute on smaller tiles to fit longer contexts in memory. 

This optimization keeps the system responsive and able to run where baselines fail.

---
### 2) Multi-GPU Scaling with DDP

DDP trains one model replica per GPU and synchronizes gradients each step. Compared to Naive DDP implementation, **Bucketed + Overlapped DDP** improved throughput from **25,336.7** to **27,719.1 tok/s** (**+9.4%**). It achieves **85.9%** scaling efficiency and only **+498.8 MB** (**+3%**) peak per-GPU memory overhead versus local 1-GPU.

The following plot shows the training throughput (tokens/sec) for the FlashAttention kernel across different DDP strategies. The x-axis is the DDP strategy and the y-axis is the training throughput in tokens/sec. We see that the Bucketed + Overlapped DDP strategy achieves the on par throughput with PyTorch DDP, and is significantly faster than the Naive DDP strategy.

![FlashAttention DDP Throughput][fig-ddp-throughput]  
![FlashAttention DDP Summary Table][fig-ddp-table]

Practical interpretation: this is a systems efficiency gain, not just a benchmark trick; more useful work is completed per second without a large memory penalty.

---

## Reproduce Key Results

```bash
uv sync
source .venv/bin/activate
```

```bash
# End-to-end pipeline runner (configurable via JSON)
# DDP wrapper options: none | naive | flashddp
uv run python cs336_systems/experiments/run_pipeline.py \
  --config cs336_systems/experiments/default_pipeline_config.json \
  --attention_kernel flash_attention_triton \
  --ddp_wrapper flashddp \
  --skip_data
```

```bash
# Attention forward benchmark sweep (sequence length scaling + latency)
uv run python cs336_systems/experiments/benchmark_attention_sweep.py
```

```bash
# LM benchmark matrix (kernel × DDP strategy) with throughput/memory artifacts
uv run python cs336_systems/experiments/benchmark_lm_matrix.py \
  --train_path data/tokenized/ts_train.npy \
  --val_path   data/tokenized/ts_valid.npy \
  --kernels scaled_dot_prod_attention vectorized_torch flash_attention_triton \
  --wrappers "Local No DDP" "Naive DDP" "Bucketed Overlapping DDP" "Pytorch DDP"
```

## Coursework Attribution

This project extends **Stanford CS336** assignment scaffolding with my own systems implementation and benchmarking workflow. Source handouts:
- [CS336 Assignment 1 (Basics) handout](./cs336-basics/cs336_spring2025_assignment1_basics.pdf)
- [CS336 Assignment 2 (Systems) handout](./cs336_spring2025_assignment2_systems.pdf)

---

<!-- Figure references (keeps the main text clean while preserving exact paths) -->
[fig-attn-sweep]: artifacts/attention_sweep_forward.png
[fig-ddp-throughput]: artifacts/flash_attention_triton_lm_matrix_throughput.png
[fig-ddp-table]: artifacts/lm_matrix_table_flash_attention_triton.png
