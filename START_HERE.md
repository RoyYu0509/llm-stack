# START_HERE

一页索引，只回答两件事：**模块在哪个文件**、**哪份文档现在有效、哪份已经过期**。
别的都不在这份文件的职责里——设计动机看 `docs/`，进度看 GitHub issue。

---

## 1. 过期文档（已知打架，别照着建）

| 文档 | 问题 | 现在以谁为准 |
|---|---|---|
| `serving/docs/designs/DESIGN.md` | 头部写 `Status: Approved`，但**静态预分配 KV cache** 和 **2:1 prefill 调度**已被 Week 3 re-scope 否掉 | `serving/docs/progress/week3-discussion.md` |
| `serving/docs/progress/future-tasks.md` | 要求实现 `Scheduler` + `Batcher` 两层独立 class | week3-discussion.md 明确否掉 pluggable scheduler 这个方向 |
| `serving/BACKLOG.md` | P0「inference 要用 try/finally」 | 已实现，见 `serving/inferenceLM/engine/inference_engine.py:47-58` |
| `training/CLAUDE.md`、`training/README.md` | 把 **85.9% scaling efficiency** 记在 `BucketedOverlapDDP.py` 名下 | 实际接线跑出这个数的是 `FlashDDP.py` 里的 `DDPOverlapBucketed` 类，`BucketedOverlapDDP.py` 没接线 |
| `serving/ROADMAP.md` §3（Stage-homogeneous batching） | 把"HF `attention_mask` 混不了 prefill/decode"当死约束，"两次顺序 microbatch"标成已 LOCKED | `docs/sprints/week5-discussion.md`（流程仓）D-2——约束本身能绕开（serving 用的是自己写的 CS336 GPT2，不是 HF `model()`），整个设计换成完整 RadixAttention |

**这张表是本仓库"哪份文档过期了"的唯一权威**——`serving/ROADMAP.md` 自己不再重复维护这份判断，只指回这里。

## 2. ⚠️ 设计 vs 实现进度（容易踩的坑）

**continuous batching 已经实现（2026-10），但还不是目标设计。** 现在的状态：

| 已实现 | 在哪里 |
|---|---|
| 逐轮推进：每轮 admit → prefill → **所有**在飞请求各 decode 一步 → 让出 | `serving/inferenceLM/engine/inference_engine.py::_keep_get_request_and_inference` |
| 变长批量 decode：左补零的 KV 矩形 + 每行位置 + key mask，一次前向 | `serving/inferenceLM/engine/ragged_batch.py`（`RaggedDecodeBatch`）、`TransformerLM.forward_with_cache(positions=, attn_mask=)` |
| 优化路径（**默认关闭**）：预分配 KV、FA-2 decode、合批 prefill、`torch.compile` | `serving/inferenceLM/engine/prealloc_batch.py`、`training/cs336-basics/.../kv_cache_attention.py`、`InferenceEngine.prefill_batch_tokens` |
| 对标 SGLang / vLLM / flash-attn 的 benchmark 与结果 | `benchmarks/compare/`（结果在 `results/final/`，入口见根目录 README） |

**还没做的（别把"有 CB"当成"有这些"）**：block-based（paged）KV cache 分配器（LMS-8, #12）、不 padding 的 ragged attention（LMS-9, #13）、radix tree 前缀缓存、
**准入控制**（在飞请求超过 KV 槽位数时 `_batch.add()` 抛错没人接，主循环 task 静默死掉）、CUDA graph decode。

**目标设计换过一次，别照旧的读**：`week3-discussion.md` 锁定的是「每轮两次 `model()` 调用，
prefill microbatch 和 decode microbatch 顺序执行」（stage-homogeneous batching）。这个设计
2026-08-23 被 `docs/sprints/week5-discussion.md`（流程仓）D-2 取代——约束的根源（HF
`attention_mask` 只认矩形 padding mask）本身能绕开，因为 serving 用的是自己写的 CS336 GPT2
架构，不是 HF `model()`。现在的目标是完整 RadixAttention：block-based KV cache 分配器
（paged）+ ragged attention forward（按 block table gather，不再 padding）+ radix tree
前缀缓存（引用计数/copy-on-write）+ LRU 淘汰。**读代码时别把 stage-homogeneous batching 当成还要去建的目标——
那个设计已经作废；也别把现在的左补零 ragged 矩形当成最终的 KV 布局——它是 LMS-7/10 的中间态。**

## 3. 模块地图

### training/（原 LLM-system-project）

| 模块 | 文件 |
|---|---|
| TransformerLM 组件（RoPE / RMSNorm / SwiGLU / attention） | `training/cs336-basics/cs336_basics/transfromer/*.py`（目录名本身是历史 typo，不是 transformer） |
| 整体 LM / 训练循环 | `training/cs336-basics/cs336_basics/lm.py`、`lm_trainer.py`、`trainer.py` |
| BPE tokenizer（Python 版 + C++ 版） | `training/cs336-basics/cs336_basics/bpe_tokenizer/`、`bpe_tokenizer_cpp/` |
| FlashAttention（naive / vectorized / triton 三版都在） | `training/cs336_systems/FlashAttention/flash_attention_triton.py` 是 Triton kernel |
| Bucketed + Overlapped DDP（85.9% 那个数的来源） | `training/cs336_systems/Parallelization/FlashDDP/FlashDDP.py` → `class DDPOverlapBucketed` |
| benchmark 产出 | `training/artifacts/` |

### serving/（原 llm-serving）

| 模块 | 文件 |
|---|---|
| 推理主循环入口（逐轮调度） | `serving/inferenceLM/engine/inference_engine.py` → `InferenceEngine._keep_get_request_and_inference`（见上面第 2 节的限定） |
| 批量 decode 的 KV 数据结构 | `serving/inferenceLM/engine/ragged_batch.py`· `prealloc_batch.py`（默认关闭） |
| serving 测量工具 | `serving/benchmarks/`（指标定义只在 `serving_metrics.py`） |
| 跨系统对标（SGLang / vLLM / flash-attn / torch DDP） | `benchmarks/compare/`，入口 `README.md` · `RUNBOOK.md` · `results/final/` |
| 单请求 prefill/decode 逻辑 | `serving/inferenceLM/engine/lm_engine.py` → `LMEngine.prefill` / `.decode` / `.inference` |
| 请求接收 / tokenize | `serving/inferenceLM/request_receiver/` |
| 请求数据结构 / 状态 | `serving/inferenceLM/data/request.py`、`request_status.py`、`tokenized_data.py` |
| 架构决定记录 | `serving/docs/decisions/adr000-request.md`、`adr001-inference.md` |

## 4. 有效设计文档（按可信度排）

1. `docs/sprints/week5-discussion.md`（流程仓）—— **Phase 1③ batching 设计的当前 SSOT**（D-2：
   完整 RadixAttention，取代 stage-homogeneous batching；D-1 定了目标岗位和主线不变；
   D-3 定了 `update_weights` 接口要覆盖 pretraining）
2. `serving/docs/progress/week3-discussion.md` —— continuous batching 其余部分仍然有效的 SSOT
   （engine 生命周期状态机、request/iteration lifecycle 拆分——这两块和 D-2 正交，不受影响）
3. `serving/docs/decisions/adr*.md` —— 已拍板的架构决定
4. 顶层 `README.md`（`llm-internship/`，流程仓）的「一个月的形状」表 —— Week 5-7 的范围
