| C | rung | tok/s | vs prev | vs rung 0 | TTFT p50 | TTFT p99 | steady TTFT p50 / p99 | TPOT p50 | ITL p99 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 fa2 (stage-B arm, re-run) | 58 |  |  | 22.6 | 24.8 | 23 / 25 | 17.1 | 18.2 |
| 1 | 1 + last-position logits | 58 | 1.00x | 1.00x | 22.0 | 23.5 | 22 / 24 | 17.1 | 18.1 |
| 1 | 2 + batched prefill, 1 group/round | 57 | 0.99x | 0.99x | 22.2 | 23.8 | 22 / 24 | 17.3 | 18.4 |
| 1 | 3 + torch.compile decode | 163 | 2.84x | 2.82x | 21.9 | 23.3 | 22 / 23 | 5.9 | 6.3 |
| 8 | 0 fa2 (stage-B arm, re-run) | 348 |  |  | 88.1 | 187.5 | 67 / 135 | 20.8 | 67.0 |
| 8 | 1 + last-position logits | 354 | 1.02x | 1.02x | 84.6 | 177.5 | 64 / 128 | 20.4 | 64.4 |
| 8 | 2 + batched prefill, 1 group/round | 372 | 1.05x | 1.07x | 48.7 | 63.3 | 45 / 63 | 19.9 | 45.5 |
| 8 | 3 + torch.compile decode | 901 | 2.42x | 2.59x | 35.8 | 58.8 | 32 / 50 | 8.0 | 32.3 |
| 32 | 0 fa2 (stage-B arm, re-run) | 1003 |  |  | 222.8 | 730.8 | 222 / 339 | 26.3 | 224.5 |
| 32 | 1 + last-position logits | 998 | 1.00x | 1.00x | 219.5 | 724.0 | 219 / 333 | 26.6 | 220.6 |
| 32 | 2 + batched prefill, 1 group/round | 1272 | 1.27x | 1.27x | 62.9 | 274.8 | 54 / 104 | 22.7 | 76.5 |
| 32 | 3 + torch.compile decode | 2500 | 1.97x | 2.49x | 51.9 | 259.6 | 43 / 93 | 11.4 | 65.1 |
| 64 | 0 fa2 (stage-B arm, re-run) | 1428 |  |  | 419.6 | 1518.6 | 408 / 661 | 36.0 | 435.9 |
| 64 | 1 + last-position logits | 1515 | 1.06x | 1.06x | 384.4 | 1411.4 | 381 / 612 | 34.1 | 403.1 |
| 64 | 2 + batched prefill, 1 group/round | 2093 | 1.38x | 1.47x | 65.0 | 557.3 | 60 / 104 | 28.7 | 85.8 |
| 64 | 3 + torch.compile decode | 3752 | 1.79x | 2.63x | 52.4 | 520.4 | 48 / 91 | 16.0 | 73.5 |

Drift check (stage-B naive arm, C=32, re-run in this session): 714 tok/s, TPOT p50 38.1 ms (stage B on the other machine: 828 tok/s, 33.2 ms).
