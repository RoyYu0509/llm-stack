# ROADMAP

> 两部分：**Part I 技术路线**（这个仓库接下来建什么）和 **Part II 学习路线**（怎么把已经建好的东西真正记住）。
> 最后更新 2026-07-29。

配套讲义：[`docs/handout/assignment-a.tex`](docs/handout/assignment-a.tex) —《Inference Serving Engine：一个请求在系统里的一生》。下文提到的 `Checkpoint (slug)` 都是那份讲义里的自测题块。

---

## Part I — 技术路线

### 现状定位

Week 1、Week 2 完成并各有 retro。Week 3 **有设计、无代码**：`docs/progress/week3-discussion.md` 锁定了关键决策，但 `git log` 停在 `Week3 start`，engine 至今仍是「取一条请求 → 跑完它的完整 prefill→decode → 再取下一条」。

**读文档前先看这个**：`docs/designs/DESIGN.md` 的 Status 是 Approved，但它写于 Week 3 re-scope 之前，它的**静态预分配 KV cache** 和 **2:1 prefill 优先调度**都已被 `week3-discussion.md` 取代。照 DESIGN.md 建会建错。

`BACKLOG.md` 也已过期：P0 里的 `[BUG] Inference 中要用 try / finally` **实际上已经实现了**（`inference_engine.py` 的消费循环里有 `finally: task_done()`），应该移到 Done。

### 建设顺序

排序依据是**依赖关系**，不是价值大小。

#### 1. Engine 生命周期状态机（`INIT → RUNNING → DRAINING → STOPPED`）

这是唯一的真正阻塞项，也是 `BACKLOG.md` 里仅剩的有效 P0。

现在 `kill()` 只是把 `self.open` 置 False，而循环正卡在 `await pending_queue.get()` 上——一个空队列上的 killed engine 会永远挂着。所以每个测试都用 `task.cancel()` 而不是 `kill()`，靠 pytest 的 event loop 清理兜底。

**为什么它排第一**：continuous batching 的循环不是「取一条、做完」，而是一个持续维护 batch 成员的循环。没有 DRAINING 状态，你无法回答「正在 decode 到一半的请求，shutdown 时怎么办」——而这个问题在有 scheduler 之后每次都要回答。

第一步：给 `InferenceEngine` 加显式状态字段和 `async shutdown(drain=True)`，把测试里的 `task.cancel()` 逐个换掉。换不掉的那个就是设计还没想清楚的地方。

#### 2. 拆开 request lifecycle 与 iteration lifecycle

`week3-discussion.md` 对现状的诊断原话是：一次 iteration 等于一条请求的完整生命周期。**这是根因；「GPU 利用率低」只是症状**——作者自己在 §4 的自我修正表里记了这一条。

具体动作：`_keep_get_request_and_inference()` 从「取一条跑到底」改成「每轮迭代推进所有在飞请求各一步」。需要一个 in-flight 集合来放 `(request_id, latest_token, past_key_values)`，也就是 `future-tasks.md` 里那个 `DecodeStorage`。

⚠️ `future-tasks.md` 是 DESIGN.md 时代的产物，它分了 Scheduler + Batcher 两层，而 `week3-discussion.md` 明确说不做 pluggable scheduler。**动手前先决定哪个是现行意图**，这两份文档目前不一致。

#### 3. Stage-homogeneous batching

已 LOCKED，理由是后端约束而非偏好：HF `model()` 的 `attention_mask` 是 `[B, S]` 的 padding mask，不是 per-pair 的 causal mask，所以一次 forward 里混不了 prefill 和 decode。

每轮迭代两次 `model()` 调用：先一个 prefill microbatch，再一个 decode microbatch。**这两次是顺序执行，不是 GPU 并行**——写代码时容易顺手把它想成并行。

prefill-first 也已 LOCKED：TTFT 更好，decode 的 ITL 更差。

#### 4. Admission policy（**仍然 OPEN**）

DESIGN.md 的 2:1 比例在「两个阶段都在同一轮迭代里」这个新模型下已经没有原来的含义了，而替代方案没定。

这是整条路线上唯一一个还没有答案的设计问题。建议在有了第 2、3 步的可运行循环之后再定——先拿到真实的 TTFT/ITL 曲线，再决定策略，而不是反过来。

#### 5. 真正的 KV cache manager

目前 KV cache 是每请求一份、请求结束就扔，没有跨请求复用也没有预分配。DESIGN.md 那套「75.5 MB/slot × 146 slots」的预算表算的是**容量规划**，那部分推理仍然有效，被取代的是「静态预分配」这个策略选择。

姊妹项目已经写了一个真正的增量 KV cache（在 `TransformerLM` 上），可以先去读那个实现再设计这里的 manager：见 [`../LLM-system-project/docs/handout/assignment-b.tex`](../LLM-system-project/docs/handout/assignment-b.tex) 第 4 章。

#### 6. `output/` 层与 `benchmarks/`

两个目录都是空的。

`output/` 是 detokenize 与流式回传。现在没有 response 通道——调用方只能轮询 `request_store[request_id]`，这在有了 streaming 之后必须改。

`benchmarks/` 是空的，而唯一一个真实的 serving benchmark 在姊妹仓库里（`cs336_systems/experiments/bench_serving.py`，它穿过这里的 `LMEngine` 真实路径）。**在这个仓库里重写一个不如把那个搬过来**，否则会出现两份互相漂移的测量代码。

#### 7. 收尾项（随时可做，互不阻塞）

- `LICENSE` 是 0 字节
- `README.md` 还是脚手架，写着「[Project Name — TO BE DECIDED]」
- `ruff` 和 `mypy` 声明为 dev 依赖但没有任何配置、也没有 CI
- `pyproject.toml` 里 `pytest-asyncio` 放在运行时依赖而不是 dev 依赖
- `[tool.setuptools.packages.find]` 没有 include 过滤，把 `tests`、`docs` 也当成包了
- `RequestStatus.PREFILLING` / `DECODING` 是死值。第 2 步做完之后它们要么被真正写入、要么删掉——**两个都不动是现在这个状态**

---

## Part II — 学习路线

依据是 [`../interview_prep/RECALL_LOG.md`](../interview_prep/RECALL_LOG.md) 里对本仓库的那次回忆测试，结果是 🟡（答了个大概）。下面按**错得最结构性的排在最前**。

### 🟡 请求生命周期 —— 三处记错，一处完全没提到

| 当时的回答 | 实际 | 去读 |
|---|---|---|
| 以为 tokenize 发生在 `InferenceEngine` 从队列取出之后 | Receiver 在**放进队列之前**就 tokenize 好了 | 讲义 §1；`Checkpoint (two_data_classes)` |
| 以为 `InferenceEngine` 和 `LMEngine` 之间还有一个 prefill queue | 整个系统只有一个 `pending_queue`；`InferenceEngine → LMEngine` 是直接协程调用 | `Checkpoint (shared_mutable_coupling)` |
| **完全没提到 `request_store`** | 没有 response queue，调用方要轮询 `request_store[request_id]`。这份共享 dict 和 `pending_queue` 一起构成了系统里**仅有的**耦合 | 讲义 §1.4；`Checkpoint (shared_mutable_coupling)` |
| decode 的具体机制、错误处理是提醒后现学的 | 只传最新 token + `past_key_values`；`try/except/finally` + `task_done()` 防止队列卡死 | `Checkpoint (prefill_decode_shape)`、`Checkpoint (task_done_finally)` |

这四条指向同一个薄弱点：**记住了数据流的形状，没记住状态放在哪里**。先把 §1.4 那张「两个共享可变对象」的图默画一遍，再往下走。

### 从未被问到、但一被问就会卡的

这些在 RECALL_LOG 里没有记录，因为压根没被问到。按「答不出来最难看」排序：

1. **`await asyncio.sleep(0)` 到底让出了什么** — `Checkpoint (sleep_zero_semantics)`、`Checkpoint (weak_async_alternative)`。这是全仓库最容易讲成「async 让 CPU 和 GPU 并行了」的地方，而那句话是错的。跑一遍仓库根目录的 `draft.py` 比读十遍代码有用。
2. **为什么不用 `model.generate()`** — `Checkpoint (why_not_generate)`。这是 ADR-001 的第一条决策，也是整个项目能存在的理由。
3. **KV cache 为什么合法** — `Checkpoint (kv_cache_legality)`。正确答案是 causal mask，不是「过去的 token 不会变」。后者听起来对，但它解释不了为什么双向模型不能这么做。
4. **`run()` 为什么不是 `async def`** — `Checkpoint (run_returns_task)`。
5. **continuous batching 的定义** — `Checkpoint (continuous_batching_definition)`。答案里必须出现「step 粒度」和「batch membership」，「同时跑很多请求」不算答对。

### 一个容易被忽略的整体判断

`Checkpoint (ssot_contradicts_itself)` 考的不是某个机制，而是「当两份文档冲突时你怎么办」。这个仓库里 `DESIGN.md` 和 `week3-discussion.md` 在 batching 上冲突，而 `week3-discussion.md` **自己内部**也有一处不一致。能识别出这个，比记住任何一个具体设计都重要。

### 建议的读法

第一遍顺读讲义，Checkpoint 全部跳过。第二遍只做 Checkpoint，答不上来再翻回对应小节。想要空白自测版：

```bash
cd docs/handout && latexmk -xelatex -jobname=assignment-a-quiz -pretex='\def\HANDOUTQUIZ{}' -usepretex assignment-a.tex
```
