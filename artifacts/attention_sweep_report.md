# Attention Forward Benchmark Sweep

| Kernel | Tier | Q_N | K_N | head_dim | heads | batch | avg_ms | std_ms | min_ms |
|--------|------|-----|-----|----------|-------|-------|--------|--------|--------|
| scaled_dot_prod_attention | XS-seq2048-hd64-12h-8b | 2048 | 2048 | 64 | 12 | 4 | 14.0893 | 0.151 | 13.9472 |
| scaled_dot_prod_attention | XS-seq4096-hd64-12h-8b | 4096 | 4096 | 64 | 12 | 4 | 56.0007 | 0.1561 | 55.6567 |
| scaled_dot_prod_attention | XS-seq8192-hd64-12h-8b | 8192 | 8192 | 64 | 12 | 4 | 0 | 0 | 0 |
| scaled_dot_prod_attention | XS-seq16384-hd64-12h-8b | 16384 | 16384 | 64 | 12 | 4 | 0 | 0 | 0 |
| vectorized_torch | XS-seq2048-hd64-12h-8b | 2048 | 2048 | 64 | 12 | 4 | 11.1239 | 0.0389 | 11.0656 |
| vectorized_torch | XS-seq4096-hd64-12h-8b | 4096 | 4096 | 64 | 12 | 4 | 43.7941 | 0.0404 | 43.7203 |
| vectorized_torch | XS-seq8192-hd64-12h-8b | 8192 | 8192 | 64 | 12 | 4 | 177.8193 | 0.4083 | 177.0101 |
| vectorized_torch | XS-seq16384-hd64-12h-8b | 16384 | 16384 | 64 | 12 | 4 | 0 | 0 | 0 |
| vectorized_torch_compiled | XS-seq2048-hd64-12h-8b | 2048 | 2048 | 64 | 12 | 4 | 2.9847 | 0.0341 | 2.9082 |
| vectorized_torch_compiled | XS-seq4096-hd64-12h-8b | 4096 | 4096 | 64 | 12 | 4 | 13.0045 | 0.2115 | 12.7568 |
| vectorized_torch_compiled | XS-seq8192-hd64-12h-8b | 8192 | 8192 | 64 | 12 | 4 | 52.9499 | 0.1856 | 52.4883 |
| vectorized_torch_compiled | XS-seq16384-hd64-12h-8b | 16384 | 16384 | 64 | 12 | 4 | 0 | 0 | 0 |
| flash_attention_triton | XS-seq2048-hd64-12h-8b | 2048 | 2048 | 64 | 12 | 4 | 1.8115 | 0.015 | 1.8021 |
| flash_attention_triton | XS-seq4096-hd64-12h-8b | 4096 | 4096 | 64 | 12 | 4 | 6.8102 | 0.0069 | 6.8021 |
| flash_attention_triton | XS-seq8192-hd64-12h-8b | 8192 | 8192 | 64 | 12 | 4 | 27.199 | 0.0099 | 27.1851 |
| flash_attention_triton | XS-seq16384-hd64-12h-8b | 16384 | 16384 | 64 | 12 | 4 | 116.6027 | 1.4708 | 113.7471 |

![Forward Time](attention_sweep_forward.png)

![Heatmap](attention_sweep_heatmap.png)

![Scaling Curve](attention_sweep_scaling.png)
