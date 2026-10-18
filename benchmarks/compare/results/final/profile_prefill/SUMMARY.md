# Attention prefill profile -- grouped by attention kernel type
bf16, causal, 12 heads x hd64, B*S = 32768, forward only; torch.profiler over 20 calls after warm-up.

| attention kernel type | lengths ok | mean baseline ms | mean GPU kernel ms/call | mean launches/call | mean GPU-busy frac | geo-mean latency vs sdpa_flash |
|---|---|---|---|---|---|---|
| attention | 6/8 | 29.640 | 29.008 | 21.3 | 0.99 | 19.97x |
| torch_compile_attention | 8/8 | 22.885 | 22.808 | 3.0 | 1.00 | 5.66x |
| sdpa_flash | 8/8 | 3.178 | 3.158 | 1.0 | 0.99 | 1.00x |
| flash_attn_pkg | 8/8 | 3.497 | 3.533 | 1.0 | 0.99 | 1.14x |
| triton_fa_mine | 8/8 | 9.785 | 9.600 | 3.0 | 0.99 | 2.62x |

Nsight Systems cells captured: 38

Means over lengths mix very different sizes; use them to compare kernel *types*, and read the per-length CSV (`attention_profile.csv`) for any single claim.