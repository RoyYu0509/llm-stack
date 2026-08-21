# v0-lm-backend
**Author**: [Yifan Yu] | **Date**: [2026-04-09] | **Status**: Approved
> **Model:** GPT-2 124M (FP32)
> 
> 
> **Hardware:** Single NVIDIA RTX 3060/3070 GPU (12 GB VRAM)
> 
> **Target:** 4 concurrent requests, ≥10 tokens/s per request, low TTFT
> 

---

## 1. Engineering Goal

构建一个 **text-input CLI inference platform**，多个用户通过 terminal 同时交互 (类似 Ollama).

1. **Fast prefill** — 尽量降低 Time To First Token (TTFT)。
2. **Decode throughput** — 4 路并发下每条 request ≥10 tok/s。
3. **Multi-user serving** — requests 要被 batch 和 schedule，不能 serialize。

---

## 2. Constraints

| # | Constraint |
| --- | --- |
| C1 | Decode 是 **auto-regressive** 的：每个 iteration 每条 sequence 只能生成一个 token，无法 parallelize 单条 request 内的 token generation。 |
| C2 | TTFT (Prefill) 要尽量低，因为人都是不耐心的。 |
| C3 | Batch decode requests 能提高 throughput，但 batch 太大会导致 per-request latency 升高（compute 被 share）。需要找到合适的 batch size。 |
| C4 | 每条 batched request 需要独立的 KV Cache，必须 manage 每条 request 的 allocation/deallocation。 |

---

## 3. System Architecture

```
                         ┌─────────────────────────────────────────┐
                         │           Core LM-Engine                │
                         │                                         │
┌──────────────┐         │  ┌─────────────────────────────────┐    │         ┌────────────────┐
│   Request    │         │  │         Scheduler               │    │         │    Output      │
│   Receiver   │         │  │   (prefill-prioritized, 2:1)    │    │         │    Module      │
│              │         │  └────────┬────────────┬───────────┘    │         │                │
│  - TCP/stdin │         │           │            │                │         │  - Detokenize  │
│  - Tokenize  ├────────▶│           ▼            ▼                ├────────▶│  - Stream to   │
│  - Enqueue   │         │  ┌────────────┐ ┌─────────────┐         │         │    each user   │
└──────────────┘         │  │  Prefill   │ │   Decode    │         │         └────────────────┘
                         │  │  Queue     │ │   Queue     │         │
                         │  └─────┬──────┘ └──────┬──────┘         │
                         │        │               │                │
                         │        └───────┬───────┘                │
                         │                ▼                        │
                         │  ┌─────────────────────────────────┐    │
                         │  │     Continuous Batcher          │    │
                         │  │  (assemble batch per iteration) │    │
                         │  └────────────────┬────────────────┘    │
                         │                   ▼                     │
                         │  ┌─────────────────────────────────┐    │
                         │  │   GPT-2 Forward Pass (GPU)      │    │
                         │  │                                 │    │
                         │  │   ┌─────────────────────────┐   │    │
                         │  │   │  KV Cache Manager       │   │    │
                         │  │   │  (static pre-alloc)     │   │    │
                         │  │   └─────────────────────────┘   │    │
                         │  └─────────────────────────────────┘    │
                         └─────────────────────────────────────────┘
```

**Data flow:** User input → Receiver (tokenize + enqueue) → Scheduler (选 prefill 还是 decode) → Batcher (组装 GPU batch) → Forward pass (读写 KV Cache) → Output module (detokenize + stream back)。

---

## 4. Key Design Decisions

### 4.1 KV Cache Management — Static Pre-allocation

**决定：** 为每个 request slot 预分配固定大小的 KV Cache buffer，按满 context window (1024 tokens) 分配。

**Resource budget：**

| Item | Formula | Size |
| --- | --- | --- |
| 单条 request 的 KV Cache | `2 × 12 layers × 12 heads × 1024 seq × 64 dim × 4 bytes` | **75.5 MB** |
| Model weights | `124M params × 4 bytes` | **496 MB** |
| 可用于 KV Cache 的 VRAM | `12 GB − 0.5 GB (model) − ~0.5 GB (overhead)` | **~11 GB** |
| 最大并发 slots | `11 GB / 0.075 GB` | **~146 slots** |

**4 条 concurrent requests 只需 ~302 MB KV Cache，完全没有 memory pressure。**

**Tradeoff：** 浪费 60–80% 的已分配内存（大部分 sequence 远短于 1024）。4 slots 下完全可以接受；超过 ~32 concurrent requests 时需要迁移到 PagedAttention。

**Future path：** Scale 到 高并发时迁移到 PagedAttention。

### 4.2 Batching Strategy — Continuous Batching

**决定：** 以 **request level** 进行 load in/off，不是 batch level。

一条 request 生成完毕（EOS 或 max length）后立即 evict，马上从 waiting queue admit 新 request。不会出现快 request 等慢 request 的情况。

**为什么不用 Static Batching：** 4 条 request 的 sequence length 分别是 50、100、200、500 tokens。Static batching 下前三条必须等第四条生成完 500 tokens 才能返回。对第一个用户来说，50 tokens 五秒就能拿到结果，现在要等 50 秒。不可接受。

### 4.3 Request Scheduling — Flexible Prefill-Prioritized

**决定：** 维护两个 logical queue：**prefill queue** 和 **decode queue**。每个 iteration 只跑 prefill batch 或 decode batch（V0 不混合）。

**Scheduling rule：**

```
each iteration:
    if prefill_queue is non-empty:
        run 2 rounds prefill, then 1 round decode  (2:1 ratio)
    else:
        run decode
```

**Rationale：**

- Prefill 优先降低 TTFT — 新 request 能被快速处理。
- 交替跑 decode 防止 in-flight requests 被 starve。
- 2:1 是 starting heuristic，后续根据 profiling 调整。

**Future path (V1)：** Ragged batching — 在单个 GPU batch 中混合 prefill 和 decode tokens，提高 utilization。需要 padding/masking logic。

---

## 5. Alternatives Considered

| Component | Rejected Option | Reason |
| --- | --- | --- |
| KV Cache | PagedAttention | V0 实现复杂度太高；4 slots 没有 memory pressure |
| Batching | Static Batching | Head-of-line blocking — 快 request 要等慢 request |
| Scheduling | Batch-level FIFO | 同样的 head-of-line blocking；无法 interleave prefill/decode |
| Scheduling | 先做完所有 prefill 再 decode | Prefill requests 持续到达时会 starve decode queue |

---

## 6. Open Questions

| # | Question | Current Thinking |
| --- | --- | --- |
| Q1 | 怎么手动 manage GPU memory？用 external allocator 还是 internally track available VRAM？ | 写一个 `KVCacheManager` class track slot allocation。用 `torch.empty()` pre-allocate tensors；现阶段不需要 raw CUDA malloc。 |
| Q2 | 怎么 parallelize async work（接收 request、跑 GPU forward、streaming output）？ | Python `asyncio` 负责 I/O (receive/send)。GPU forward 在 main loop synchronously 跑或放到 dedicated thread。 |
| Q3 | 什么是 GPU 的一个 "iteration"？怎么决定每个 iteration process 什么？ | 一个 iteration = 一次 forward pass。Scheduler 决定跑哪个 queue、包含哪些 requests。Batch size 受 VRAM budget 约束。 |
| Q4 | 模块化到什么程度？ | 分成 `Receiver`、`Scheduler`、`KVCacheManager`、`Batcher`、`ModelRunner`、`OutputStreamer`。通过 queue 通信。Interface 尽量窄，方便独立替换。 |

---

## 7. Milestones

| Week | Deliverable | Acceptance Criteria |
| --- | --- | --- |
| 1 | **Request Receiver Module** | 能同时接受多条用户连接。Tokenize 每条 request。Enqueue 到 LM-Engine。 |
| 1 | **LM-Engine: Request Intake** | 接收 Receiver 发出的 tokenized requests，存到 decode queue。 |
| 2–3 | **Static KV Cache Manager** | Pre-allocate slots。超出 VRAM 的 requests 被 halt 并 queue 住。Request 完成后 free slot。 |
| 3–4 | **Continuous Batching** | Prefill 阶段的 continuous batching：拼接短 requests，pad 到 batch 内最长。每个 iteration evict 已完成的 request。 |
| 4 | **Prefill-Prioritized Scheduler** | 实现 2:1 prefill:decode interleaving rule。新到达的 request 正确路由到 prefill queue。 |
| 5 | **Output Module** | Detokenize 生成的 tokens。Stream partial output 回对应的 user connection。 |
| 6 | **End-to-end Testing & Profiling** | 完整 pipeline 在 4 concurrent requests 下跑通。用 profiling data 找出 5 个 bottleneck，每个提出 optimization plan。 |
| 7 | **KV Cache Optimization** | 实现最优先的 optimization（e.g., dynamic allocation, memory pooling）。量化对比 week 5 baseline 的 memory reduction。 |
| 8 | **Scheduling & Batching Optimization** | 实现 ragged batching 或改进 scheduling。目标：相对 week 1–4 naive baseline **50% speedup, 60% memory reduction**。 |

