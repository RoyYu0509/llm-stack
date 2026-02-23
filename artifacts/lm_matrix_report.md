# LM Training Benchmark Matrix

## Design

- **Local batch size** is the **same** across single-GPU and DDP.
- **Steps per epoch** is identical for all configurations.
- DDP therefore processes `world_size ×` more samples per epoch in roughly the same wall-clock time.
- Memory is reported **per-GPU** (worst-case across ranks).

## Results

| Kernel | DDP | GPUs | Epochs | Steps/ep | Global BS | Local BS | Samples/ep | Wall (s) | s/ep | tok/s | Peak GPU MB |
|--------|-----|------|--------|----------|-----------|----------|------------|----------|------|-------|-------------|
| scaled_dot_prod_attention | none | 1 | 5 | 51 | 4 | 4 | 204 | 185.437 | 37.087 | 2816.3 | 7674.8 |
| scaled_dot_prod_attention | naive | 2 | 5 | 51 | 8 | 4 | 408 | 283.945 | 56.789 | 3678.5 | 7674.8 |
| scaled_dot_prod_attention | flashddp | 2 | 5 | 51 | 8 | 4 | 408 | 230.764 | 46.153 | 4526.2 | 8394.3 |
| scaled_dot_prod_attention | torch_ddp | 2 | 5 | 51 | 8 | 4 | 408 | 230.959 | 46.192 | 4522.4 | 8396.4 |
| vectorized_torch | none | 1 | 5 | 51 | 4 | 4 | 204 | 180.458 | 36.092 | 2894.0 | 7572.7 |
| vectorized_torch | naive | 2 | 5 | 51 | 8 | 4 | 408 | 265.247 | 53.049 | 3937.8 | 7572.7 |
| vectorized_torch | flashddp | 2 | 5 | 51 | 8 | 4 | 408 | 226.695 | 45.339 | 4607.4 | 8291.8 |
| vectorized_torch | torch_ddp | 2 | 5 | 51 | 8 | 4 | 408 | 230.335 | 46.067 | 4534.6 | 8294.0 |
| flash_attention_triton | none | 1 | 5 | 51 | 4 | 4 | 204 | 187.088 | 37.418 | 2791.4 | 7476.9 |
| flash_attention_triton | naive | 2 | 5 | 51 | 8 | 4 | 408 | 269.993 | 53.999 | 3868.5 | 7476.9 |
| flash_attention_triton | flashddp | 2 | 5 | 51 | 8 | 4 | 408 | 228.77 | 45.754 | 4565.6 | 8196.0 |
| flash_attention_triton | torch_ddp | 2 | 5 | 51 | 8 | 4 | 408 | 228.753 | 45.751 | 4566.0 | 8197.6 |

## Charts

![Time per Epoch](lm_matrix_time.png)

![Peak Per-GPU Memory](lm_matrix_memory.png)

![Throughput](lm_matrix_throughput.png)

![Samples per Epoch](lm_matrix_samples.png)

![Loss Convergence](lm_matrix_convergence.png)
