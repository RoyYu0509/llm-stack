# Overnight run report

## 1. Pretraining

| step | val loss |
|---|---|
| 6000 | 4.4579 |
| 12000 | 4.2015 |
| 18000 | 4.0937 |
| 24000 | 4.0288 |
| 30000 | 4.0057 |

In-training validation loss: **4.0057** at step 30000.

> **Two caveats on this number — do not quote it as "validation loss on the mixed corpus".**
> 1. The training loop's val sampler has `shuffle=False` and `_run_eval` takes the *first* `val_bat_num` batches, while `valid.txt` is written domain-ordered (web → wiki → instruction). So every eval scored roughly the first ~1,800 tokens of the **web** section — the same documents each time. That makes the curve a low-variance *paired* progress signal (the decrease is real), but not a mixed-corpus number.
> 2. `mixed_valid` itself has a contaminated instruction region: `build_mixed_corpus.build_instruction` cycles the held-out Alpaca pool 1.91× to hit the 10% ratio, and repeated text is artificially easy.
>
> The clean figures are the per-domain streams in section 3 (web / wiki / instruction, none of them oversampled).

## 2. Instruction fine-tuning arms

| arm | evals | best val | final val | note |
|---|---|---|---|---|
| lr1e4_plain | 9 | 2.4542 | 2.7179 | overfit? |
| lr1e4_replay | 9 | 2.4571 | 2.6310 | overfit? |
| lr1e5_plain | 9 | 2.4390 | 2.4631 | overfit? |
| lr1e5_replay | 9 | 2.4472 | 2.4566 | overfit? |
| lr3e5_plain | 9 | 2.4430 | 2.5308 | overfit? |
| lr3e5_replay | 9 | 2.4475 | 2.5097 | overfit? |

_'overfit?' means the final eval is worse than the best one — expected on a 4.31M-token corpus seen ~3.4 times._

## 3. Per-domain loss (paired: identical batches, fixed seed)

| model | general | web | wiki | instruction |
|---|---|---|---|---|
| pretrained | 3.5866 | 3.8469 | 3.5518 | 2.7345 |
| sft_lr1e4_plain | 3.9988 | 4.2319 | 3.9224 | 2.6589 |
| sft_lr1e4_replay | 3.7698 | 3.9914 | 3.7542 | 2.5685 |
| sft_lr1e5_plain | 3.6572 | 3.8859 | 3.5954 | 2.4150 |
| sft_lr1e5_replay | 3.6104 | 3.8375 | 3.5656 | 2.4083 |
| sft_lr3e5_plain | 3.7375 | 3.9637 | 3.6695 | 2.4793 |
| sft_lr3e5_replay | 3.6424 | 3.8650 | 3.6044 | 2.4531 |

### Change vs the pretrained model

_Negative = better. This is the forgetting trade-off: instruction should improve, general/web/wiki should degrade, and the replay arms should degrade less._

| model | general | web | wiki | instruction |
|---|---|---|---|---|
| sft_lr1e4_plain | +0.4122 | +0.3850 | +0.3706 | -0.0756 |
| sft_lr1e4_replay | +0.1832 | +0.1445 | +0.2024 | -0.1661 |
| sft_lr1e5_plain | +0.0706 | +0.0390 | +0.0437 | -0.3195 |
| sft_lr1e5_replay | +0.0238 | -0.0093 | +0.0138 | -0.3263 |
| sft_lr3e5_plain | +0.1509 | +0.1168 | +0.1177 | -0.2553 |
| sft_lr3e5_replay | +0.0557 | +0.0181 | +0.0526 | -0.2814 |

## 4. Serving

### recompute only

**recompute**

| prompt tokens | TTFT ms | decode p50 ms | tok/s |
|---|---|---|---|
| 16 | 60.3 | 60.32 | 16.6 |
| 64 | 61.1 | 61.74 | 16.1 |
| 128 | 57.8 | 61.08 | 16.5 |
| 256 | 64.0 | 62.47 | 16.0 |
| 512 | 64.7 | 63.08 | 16.1 |

### recompute vs KV cache

**recompute**

| prompt tokens | TTFT ms | decode p50 ms | tok/s |
|---|---|---|---|
| 16 | 61.3 | 63.03 | 15.9 |
| 64 | 65.7 | 61.25 | 16.4 |
| 128 | 61.8 | 62.79 | 15.7 |
| 256 | 65.3 | 64.08 | 15.6 |
| 512 | 64.5 | 64.82 | 15.6 |

**cache**

| prompt tokens | TTFT ms | decode p50 ms | tok/s |
|---|---|---|---|
| 16 | 62.3 | 52.45 | 19.2 |
| 64 | 63.0 | 48.06 | 21.0 |
| 128 | 59.3 | 52.78 | 18.9 |
| 256 | 61.2 | 51.80 | 19.8 |
| 512 | 61.5 | 52.12 | 19.2 |

**KV cache speedup by prompt length**

| prompt tokens | speedup |
|---|---|
| 16 | 1.20x |
| 64 | 1.28x |
| 128 | 1.20x |
| 256 | 1.27x |
| 512 | 1.23x |

_Speedup is roughly flat at ~1.24x and does **not** grow with prompt length. At this model size per-step cost is dominated by the fixed forward pass over the weights, so the prefix-dependent attention work a cache eliminates is only a small share of it. A KV cache pays off at longer contexts and larger batch, not here._

## 5. Systems benchmarks

Artifacts written:

- `artifacts/bench_bucket_100mb/lm_matrix_loss_curves.csv`
- `artifacts/bench_bucket_100mb/lm_matrix_report.md`
- `artifacts/bench_bucket_100mb/lm_matrix_results.csv`
- `artifacts/bench_bucket_10mb/lm_matrix_loss_curves.csv`
- `artifacts/bench_bucket_10mb/lm_matrix_report.md`
- `artifacts/bench_bucket_10mb/lm_matrix_results.csv`
- `artifacts/bench_bucket_1mb/lm_matrix_loss_curves.csv`
- `artifacts/bench_bucket_1mb/lm_matrix_report.md`
- `artifacts/bench_bucket_1mb/lm_matrix_results.csv`
- `artifacts/bench_bucket_25mb/lm_matrix_loss_curves.csv`
- `artifacts/bench_bucket_25mb/lm_matrix_report.md`
- `artifacts/bench_bucket_25mb/lm_matrix_results.csv`
- `artifacts/bench_bucket_50mb/lm_matrix_loss_curves.csv`
- `artifacts/bench_bucket_ 50mb/lm_matrix_report.md`
- `artifacts/bench_bucket_50mb/lm_matrix_results.csv`
- `artifacts/bench_matrix/lm_matrix_loss_curves.csv`
- `artifacts/bench_matrix/lm_matrix_report.md`
- `artifacts/bench_matrix/lm_matrix_results.csv`

