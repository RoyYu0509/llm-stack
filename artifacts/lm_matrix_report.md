# LM Training Benchmark Matrix

## Design

- **Local batch size** is the **same** across single-GPU and DDP.
- **Steps per epoch** is identical for all configurations.
- DDP therefore processes `world_size ×` more samples per epoch in roughly the same wall-clock time.
- Memory is reported **per-GPU** (worst-case across ranks).

## Results

| Kernel | DDP | GPUs | Epochs | Steps/ep | Global BS | Local BS | Samples/ep | Wall (s) | s/ep | tok/s | Peak GPU MB |
|--------|-----|------|--------|----------|-----------|----------|------------|----------|------|-------|-------------|
| vectorized_torch | Local No DDP | 1 | 5 | 51 | 8 | 8 | 408 | 24.649 | 4.93 | 5296.8 | 652.6 |
| vectorized_torch | Naive DDP | 2 | 5 | 51 | 16 | 8 | 816 | 66.134 | 13.227 | 3948.3 | 653.1 |
| vectorized_torch | Bucketed Overlapping DDP | 2 | 5 | 51 | 16 | 8 | 816 | 61.238 | 12.248 | 4264.0 | 777.5 |
| vectorized_torch | Pytorch DDP | 2 | 5 | 51 | 16 | 8 | 816 | 60.994 | 12.199 | 4281.1 | 776.0 |

## Charts

![Time per Epoch](lm_matrix_time.png)

![Peak Per-GPU Memory](lm_matrix_memory.png)

![Throughput](lm_matrix_throughput.png)

![Samples per Epoch](lm_matrix_samples.png)

![Loss Convergence](lm_matrix_convergence.png)
