# LM Training Benchmark Matrix

## Design

- **Local batch size** is the **same** across single-GPU and DDP.
- **Steps per epoch** is identical for all configurations.
- DDP therefore processes `world_size ×` more samples per epoch in roughly the same wall-clock time.
- Memory is reported **per-GPU** (worst-case across ranks).

## Results

| Kernel | DDP | GPUs | Epochs | Steps/ep | Global BS | Local BS | Samples/ep | Wall (s) | s/ep | tok/s | Peak GPU MB |
|--------|-----|------|--------|----------|-----------|----------|------------|----------|------|-------|-------------|
| flash_attention_triton | Bucketed Overlapping DDP | 2 | 2 | 51 | 16 | 8 | 816 | 174.271 | 87.136 | 9589.5 | 20735.7 |

## Charts

![Peak Per-GPU Memory](lm_matrix_memory.png)

### flash_attention_triton

![Time per Epoch](flash_attention_triton_lm_matrix_time.png)

![Throughput](flash_attention_triton_lm_matrix_throughput.png)

![Samples per Epoch](flash_attention_triton_lm_matrix_samples.png)

![Loss Convergence](lm_matrix_convergence.png)
