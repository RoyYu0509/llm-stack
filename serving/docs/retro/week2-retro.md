# Week 2 Retro — Inference Engine

> 目标：这份 retro 不是为了记录“我写了哪些 code”，而是为了以后回头时，能快速 reconstruct：**这周做了什么、为什么这样做、是怎么完成的、学到了什么、下周要注意什么**。
> 
> 
> 这个版本保留三大 sections：**Engineering / Knowledge / 这周的问题**。表达上刻意用中英文混杂，因为以后不管是自己复盘还是跟 interviewer 讨论 systems design，都更贴近真实思考方式。
> 

# 1. Engineering

## 1.0 This week，我在 engineering 上真正完成了什么

---

这周的核心不是“写完 inference backend”，而是把 整个 **system 的 boundary** 搭清楚：

1. Request 如何进入 system
2. inference loop 该暴露哪些 control surface
3. async loop 应该怎样和 request intake coexist
4. 哪些 abstraction 现在该有，哪些现在不该有
5. compute layer 和 orchestration layer 的边界应该切在哪里

从结果上看，这周真正落下来的不是 feature 数量，而是 5 个 design decisions。它们会直接决定 Week 3/4 做 continuous batching、paged KV cache、scheduler 时，代码是不是还能继续长下去。

## 1.1 用 `model(..., past_key_values=...)`，不用 `generate()`

---

### Decision

这周明确选了 **manual inference loop**，也就是直接调用 `model()` forward，并且显式传 `past_key_values`，而不是把整个 profill 和 decode 的过程交给 `model.generate()`。

### Why this decision matters

`generate()` 对普通 application 很方便，但对 inference backend 太高层了。它把整个 prefill → decode loop 封装成 black box，只给最终 output ids。这样做的问题是：后面真正重要的 systems hooks 都拿不到。

- Week 3/4 需要直接 control 的东西包括：
    - per-step `logits`
    - updated `past_key_values`
    - prefill / decode 的 boundary
    - step-level scheduling point
    - 未来做 speculative decoding 或 paged KV cache 时对中间状态的 access

### What I learned

这周最重要的一个 engineering lesson 是：

> High-level API 的代价，不是“少一点灵活性”，而是你会失去整个 backend architecture 的 control point。
> 

`generate()` 不是不能用，而是它适合“我要生成文本”；不适合“我要实现一个 inference runtime”。

### What to watch next week

下周如果开始做 batching / scheduler，不要再退回去想“能不能还是先用 generate 快速做”。这会直接把后续 extension 路堵死。

## 1.2 在 decode loop 里加 `await asyncio.sleep(0)`，目的是 cooperative scheduling

---

### Decision

decode loop 每一步结束后，要显式 `await asyncio.sleep(0)`。

### Why this decision matters

一开始最容易犯的错，是把这件事理解成“async 可以让 GPU/CPU 并行”或者“这样 inference 会更快”。这两个理解都不对。

这里的目标只有一个：**在 Python event loop 层面，让其他 coroutine 有机会运行**。

PyTorch operation 都是 Eager Evaluation + Async GPU Computes, 所以在 Decode Loop 中, 绝大部分的 Tensor Computation 都是会直接 return 一个 连着 GPU 的 Tensor handel, 然后 CPU 继续运行下面的代码, 直到遇到 Tensor Sync Point. 比如:

```
tensor_1.item() # get tensor value
ext_token = argmax(logits) # 需要用 tensor value 做 computation
```

一旦走到这个点，CPU 就会等待 GPU result ready。此时如果整个 inference coroutine 一直不 yield，event loop 会被长期占住，`RequestReceiver` 这种 coroutine 就没有机会把新 request 放进 `pending_queue`。

### How I completed it

这周我把 decode loop 的 mental model 改成：

```
while not finished:
    run_one_decode_step()
    await asyncio.sleep(0) # 单纯让CPU推进一下别的 process (不做 CPU GPU overlap)
    materialize_next_token()   # sync point
```

关键点不是 `sleep(0)` 这个 API 本身，而是它表达的 scheduling contract：

- inference coroutine 不独占 event loop
- request intake 可以在 step boundary 插进来
- Week 2 的 async behavior 建立在 cooperative scheduling 上，不是 preemptive scheduling 上

### What I learned

这周一个非常关键的 distinction 是：

- **GPU async**：kernel launch 和 CPU control flow 的关系
- **Python asyncio**：coroutine 之间如何 cooperative yield

这两层都叫 async，但是 completely different problem。

### What to watch next week

下周如果讨论 “overlap”, “parallelism”, “throughput optimization”，必须先问清楚是哪一层：

- Python event loop fairness?
- CPU/GPU overlap?
- CUDA stream concurrency?

不先分层，后面的 reasoning 会乱。

## ⭐️ 1.3 `run()` 负责 spawn background task，然后立刻 return

---

### Decision

`InferenceEngine.run()` 的职责不是“执行 engine loop 到结束”，而是 **把 engine loop 开启并注册进 event loop，然后将这个loop作为`Task`**返回.

### Why this decision matters

engine loop 本质上是 long-lived / near-infinite loop。只要 engine 还 open，它就应该持续：

- 看 queue 里有没有新 request
- 拉 request
- 跑 inference
- 更新状态

#### 重点: `run()` 不能里面写成`await engine_loop()`, 因为这个loop永远不return, 这样写caller 就再也拿不到 control，我们就彻底卡死在这里.

所以, 一定要把 engine_loop() 作为一个background task来注册上. 这样 CPU 继续往下跑, loop 作为一个 coro task 在未来CPU 切换的时候有可能被提上, 而不是切换到 loop 中然后被钉死在loop里.

所以 `run()` 的正确语义不是 runner，而是 launcher。

### How I completed it

我把它抽象成下面这个结构：

```
function run():
    assert engine_is_open
    # ⚠️注意: engine_loop 中一定要有await 或者 sleep 让 Event loop 可以切出来继续推进别的 Task
    task = create_background_task(engine_loop()) 
    return task
```

反例是：

```
async function run():
    await engine_loop() # 钉死进 engine_loop() 中出不来 + 没法 explicitly shutdown
```

### What I learned

这里学到的不是一个 API trick，而是一个 interface design lesson：

> 一个 function 的 return behavior，必须和它在系统里的角色一致。
> 

如果它是 launcher，就应该 non-blocking return。
如果它是 terminal execution，就应该 block 到完成。
不能名字叫 `run()`，语义却混在中间。

### What to watch next week

如果下周 engine loop 里再引入 scheduler / running batch，`run()` 的 contract 还是不要变。稳定的 launcher interface 很重要。

## 1.4 Design 要做到 本周scope 的 Minimalism

---

### Decision

这周不暴露还没有真实实现支撑的 interface，例如：

- 不提前暴露 `do_sample`
- 不保留没有实际职责的 `waiting_queue`

### Why this decision matters

这是这周最值得记住的 design discipline 之一。

很多时候最容易自我说服的理由是：

- “反正以后会用到”
- “先把 field 留着，之后再实现”
- “先把 queue 放着，scheduler 来了自然会用”

问题在于，这种做法会让 interface 先于实现存在。结果就是 public API 开始对 user 撒谎。

比如 `do_sample=True` 暴露出来了，但 backend 其实只支持 greedy decoding，那最后只会有两种结果：

- silently ignore
- accept request first, then fail later

这两种都不好。前者是 silent contract violation，后者是 delayed failure。

### How I completed it

这周我给自己定了一个更硬的判断标准：

```
if feature has no real execution path this week:
    do not expose it

if abstraction solves no current problem:
    remove it
```

也就是说，**抽象不是因为“看起来以后需要”才存在，而是因为“现在已经有一个具体问题需要它”才存在**。

### What I learned

public interface 是 promise，不是 wishlist。

这点以后跟 interviewer 聊 design 时也很重要，因为这类决定体现的是 engineering judgment：你是否会为了“想象中的 extensibility”牺牲当前 system 的 truthfulness。

### What to watch next week

如果下周真的开始做 sampling / scheduler，再把对应 interface 加回来。顺序必须是：

1. execution path 先存在
2. state transition 先成立
3. public interface 再暴露

不要反过来。

## 1.5 `LMEngine` 只做 pure compute；`InferenceEngine` 负责 orchestration (Layer 的职责要分清楚)

---

### Decision

这周把 layering 切得更清楚了：

- `LMEngine`：只管 model execution
- `InferenceEngine`：只管 request lifecycle / queue / store / state transition

### Why this decision matters

一开始很容易把“inference 做完后要更新 status”这件事，顺手塞进 `LMEngine`。但一旦这么做，compute layer 就开始知道 request lifecycle，coupling 会快速扩散。

正确的分层应该是：

**`LMEngine` knows**
- tokenized input
- forward pass
- generated token ids

**`LMEngine` does not know**
- request id
- request status
- request store
- orchestration policy

而这些长生命周期状态，应该属于 `InferenceEngine`。

### How I completed it

我把责任拆成下面这个 flow：

```
request_id -> fetch request payload
payload -> tokenize / prepare input
generated_tokens = LMEngine.infer(payload)
InferenceEngine writes result back to request_store
InferenceEngine updates status
```

这样 `LMEngine` 就保持成一个 pure compute unit，而 `InferenceEngine` 是 orchestration layer。

### What I learned

这周对 separation of concerns 的理解更具体了：

它不是为了“代码看起来干净”，而是为了：

- unit test 更容易写
- storage implementation 未来可以替换
- orchestration policy 可以单独演化
- compute module 可以在别的 execution context 复用

### What to watch next week

下周如果加入 batch-level state，不要把 batch metadata 塞进 `LMEngine`。batch orchestration 还是应该在 engine loop 外层处理。

## 1.6 Engineering 收获总结

---

如果用一句话概括，这周真正完成的是：

> 我没有只是在“写 inference code”，而是在确定 **inference runtime 的 control boundary 和 layering**。
> 

这些决定短期看起来只是 API 选择、queue 设计、async 写法；但长期看，它们决定后面能不能继续做：

- continuous batching
- paged KV cache
- scheduler-aware engine loop
- speculative decoding hooks

### Engineering 下周注意事项

1. 不要为了快而破坏 control surface。
2. 不要把 Python asyncio 和 GPU async 混成一层。
3. 不要提前暴露 interface。
4. 不要让 `LMEngine` 重新吸收 orchestration state。
5. 新 abstraction 一律先问：**它解决的是本周哪个真实 execution problem？**

# 2. Knowledge

## 2.0 This week，我真正搞懂了哪些 concept

---

这周的 knowledge gain 不是“多记了几个术语”，而是把几个以后会反复用到的 systems concepts 连接起来了：

- KV cache 为什么成立
- prefill / decode 到底差在哪里
- Python asyncio 和 GPU async 到底是不是一回事
- continuous batching 的结构本质到底是什么

这些 concept 如果只停留在定义层面，之后一做 implementation 就会混乱。这周比较大的进步是：它们开始能和具体 backend design 一一对应起来。

## 2.1 KV cache 成立的根本原因，是causal mask，不是“past token 没变”

---

### What I understand now

以前最容易说出一句模糊的话：

> KV cache works because previous tokens do not change.
> 

这句话不能说错，但不够。真正要能讲清楚，必须追问一句：**为什么 previous tokens 不需要重算？**

### 答案是 causal mask

在 decoder-only transformer 里，token `i` 只能 attend 到 `0..i`。所以当一个新 token `n+1` 到来时，之前 token 的 attention context 并不会因为这个 future token 而改变. 
***(注意KV Cache有很多层, 不是你单纯简单想到的那一层, 详细看下面这个链接解释)***

[解释 KV-Caching 的作用以及背后支持的理论](https://www.notion.so/KV-Caching-342f4a72836b80e58cfef7838e7a51bc?pvs=21)

因此：

- earlier token 的 `K_i / V_i` 仍然有效
- 不需要为历史 token 重新算 K/V
- 所以 cache 可以复用

### Pseudo reasoning

```
for token i:
    visible context = tokens[0..i]

when token n+1 arrives:
    visible context for token i (i <= n) does not change
    therefore past K/V remain valid
```

### Why this matters

这不是 paper definition，而是面试和系统设计里会直接被追问的 reasoning chain。

如果只会说“because past token is fixed”，会显得停留在表面。更完整的说法应该是：

> Because causal masking prevents earlier tokens from depending on future tokens, previously computed K/V remain reusable when new tokens are appended.
> 

### What to watch next week

以后讲 KV compression / paged KV cache 时，要始终从 **causal dependency structure** 出发，而不是只从 implementation trick 出发。

---

## 2.2 Prefill 和 decode 的关键区别是 input shape，不只是 phase 名字不同

### What I understand now

以前很容易把两者理解成：

- prefill = 读 prompt
- decode = 生成 token

这没错，但对 systems reasoning 不够有力。更重要的区别是 **shape**。

- prefill: `input_ids` shape 是 `[batch, seq_len]`
- decode: `input_ids` shape 是 `[batch, 1]`

这件事一旦看清楚，很多 systems consequence 就自然出来了。

### Pseudo code

```
# Prefill
outputs = model(full_prompt, past_kv = none)
kv_cache = outputs.past_kv

# Decode
repeat:
    outputs = model(last_token, past_kv = kv_cache)
    kv_cache = outputs.past_kv
```

### Why this matters

真正重要的是下面这几个 implications：

- decode step 和 decode step 更容易 batch 在一起
- prefill 和 decode 的 shape 不同，所以 mixed-stage batching 不自然
- continuous batching 不能只想“多个 request 一起跑”，还要想它们现在处于哪个 stage
- chunked prefill 之所以是一个真实优化点，就是因为它试图把 prefill 变得更像 scheduler 可管理的单位

### What I learned

这周对 “shape is a systems concern” 这句话有了更具体的理解。shape 不只是 tensor bookkeeping，它会直接决定：

- batching strategy
- scheduler complexity
- kernel behavior
- latency / throughput trade-off

### What to watch next week

下周如果画 scheduler 设计图，一定要把 request stage 画出来。不能只画 queue 和 batch。

## 2.3 GPU async 和 Python asyncio 是两层 completely different 的 async

---

### What I understand now

这周一个很重要的纠偏是：

“async” 这个词在系统里太 overloaded 了。

### Layer 1: GPU async (CPU continue GPU work)

当 CPU launch GPU kernel 时，Python control flow 往往可以先继续往下走；真正的 waiting 会发生在显式 sync point。

### Layer 2: Python asyncio (Concurrency)

这是单线程 event loop 上的 cooperative scheduling。coroutine 能不能继续推进，不取决于 GPU，而取决于有没有 `await` 把 control 还给 event loop。

### Pseudo model

```
# GPU layer
launch_gpu_work()
cpu_continue_until_sync_point()
wait_when_gpu_materialized_result()

# Python coroutine layer
do_some_work()
yield_control()
other_coroutine_runs()
```

### Why this matters

如果不把这两层分开，就很容易得出错误结论，比如：

- “用了 asyncio 所以 GPU/CPU 就 overlap 了”
- “在 loop 里 yield 一下就能减少 GPU wait”
- “async inference 本身就意味着 compute parallelism”

这些都不成立。

### What I learned

以后每次看到 async，都要先问：

> async at which layer?
> 

这是比 API 细节更重要的思维习惯。

### What to watch next week

如果开始碰更深的 performance optimization，记得把这两层拆开分析。Python coroutine fairness 和 CUDA-level overlap 是两类完全不同的优化问题。

## 2.4 Continuous batching 的本质，是单个 engine loop 在 decode step 之间吸纳新 request

---

### What I understand now

这周之前，continuous batching 很容易被想成一种“多开几个 async task 并发跑 inference”的结构。但这不是本质。

真正的 continuous batching 更接近：

- system 里有一个长期存在的 engine loop
- 它维护一个 running batch
- 每个 decode step 结束后，它会检查 queue
- 新 request 可以在 step boundary 被吸进当前运行中的 batch

### Pseudo structure

```
running_batch = []

while engine_is_open:
    admit_new_requests_into(running_batch)

    if running_batch is not empty:
        decode_one_step_for_all(running_batch)
        remove_finished_requests(running_batch)

    yield_to_event_loop()
```

### Why this matters

这个理解很关键，因为它直接改变“scheduler 应该放在哪里”的答案。

scheduler 不是一个独立于 engine loop 的外置幻想层；它应该嵌在 engine loop 的 step-by-step admission / execution policy 里。

### What I learned

这周最大的收获之一是：

> continuous batching 不是“并发地跑很多 request”，而是“在 step granularity 上动态维护 batch membership”。
> 

这个表述以后无论是写设计文档还是跟 interviewer 解释，都会清楚很多。

### What to watch next week

下周如果开始实现 continuous batching，先保证 loop structure 正确，再考虑 fancy policy。结构错了，后面 policy 再漂亮也会很别扭。

## 2.5 Knowledge 收获总结

---

这周真正补起来的是从 concept 到 system design 的桥：

- causal mask → why KV cache is valid
- shape difference → why batching is hard
- async layering → why scheduling reasoning must be split
- engine loop structure → what continuous batching really is

这部分以后面试很有用，因为 interviewer 往往不是问你“定义是什么”，而是问你：

- 为什么这样设计？
- alternative 是什么？
- 这个 choice 对后面的 scheduler / cache / throughput 有什么影响？

如果这些 reasoning chain 能讲顺，technical depth 会明显更扎实。

#### Knowledge 下周注意事项

1. 所有 concept 都要能往 implementation 映射。
2. 讲原理时不要停留在 slogan，要讲 dependency chain。
3. 讲 batching 时必须带 shape 和 stage。
4. 讲 async 时必须先分 layer。
5. 讲 continuous batching 时先讲 loop structure，再讲 policy。

# 🥵 3. 这周的问题

---

> 这一节记录的不是“我这周犯了什么小错”，而是 **我在 engineering judgment 上反复暴露出来的 failure pattern**。这些 pattern 如果不主动纠正，后面做 systems project 会一直重复出现。
> 

---

## 问题 1：加不必要的中间层

---

### Failure pattern

我很容易因为“以后可能会用到”而提前加 abstraction，例如先放一个 `Scheduler`、先多放一层 queue、先把 interface 设计得很 general。

### 为什么这是问题

这种做法表面上像是在为 extensibility 做准备，实际上经常是在制造：

- zero-coverage abstraction
- extra state to maintain
- 更高的 mental overhead
- 未来 refactor 时更多无意义包袱

本质问题不是“多写了几行 code”，而是 **我让 design 脱离了 current execution problem**。

### Correct rule

每加一个 abstraction 之前，先问：

> 它解决的是本周哪个真实 test case / execution problem？
> 

答不上来，就不该存在。

### Concrete example

如果 Week 2 只是 sequential single-request inference，那 `Scheduler` 就没有存在理由。最好的处理方式不是“先留一个空壳”，而是写一句：

> Scheduler deferred to Week 3 because Week 2 has no scheduling decision.
> 

### 下周 guardrail

下周再想加 abstraction 时，先写一句 one-line justification。没有 justification，不开工。

## 问题 2：跳过 design gate，先写再解释

---

### Failure pattern

有些时候我会先按自己当下的理解把 code 写出来，等被指出 layering 或 interface 不对，再回头解释为什么这么做。

### 为什么这是问题

这种流程训练出来的不是 reasoning，而是 rationalization。

- **事前解释**：会逼我先把设计想清楚
- **事后解释**：通常只是在替已经写出来的 code 找理由

对 systems work 来说，后者很危险，因为 layering mistake 一旦写进去，后面会扩散。

### Correct rule

动手前先过 design gate：

1. 我准备改什么
2. 为什么要这么改
3. 这个改动落在哪一层
4. 它会不会破坏已有 contract

如果这四件事说不清，就先别写。

### Concrete example

比如想把 `request_store` 注进 `LMEngine`，正确动作不是直接改 `__init__`，而是先问：

> status update 属于 compute layer 还是 orchestration layer?
> 

一旦这个问题问清楚，多半就不会写错。

### 下周 guardrail

任何涉及 layering / API / state ownership 的改动，先用 3–5 句话写 design intent，再开始实现。

## 问题 3：ADR 和代码不一致

---

### Failure pattern

设计文档里写的是一套状态机，代码里实现的是另一套；或者 ADR 里写了 abstraction，但代码已经偏离了，自己却没有及时同步。

### 为什么这是问题

这会导致两个后果：

- 文档失去约束力，最后变成历史遗迹
- 自己对 system 的 mental model 被分裂成两套

对面试来说也很危险，因为 interviewer 一追问“你当时为什么这样定义状态”，你可能会把设计稿和最终实现讲混。

### Correct rule

code merge 前，ADR 和 implementation 必须二选一对齐：

- 要么代码改回 ADR
- 要么 ADR 更新成当前实现

不能两个都不动。

### Concrete example

如果 ADR 写 `PREFILLING → DECODING → DONE`，但 Week 2 实际实现只是 `PROCESSING → DONE`，那就要明确写出来：

> Week 2 collapses prefill/decode into `PROCESSING`; fine-grained states deferred to Week 3 because no scheduler exists yet.
> 

### 下周 guardrail

每次提交前加一个固定检查：

```
git diff main -- adr/
```

如果相关 ADR 没看过，不提交。

## 问题 4：Push back 时只给结论，不给 reasoning

---

### Failure pattern

当我不同意一个 design decision 时，容易先说“我觉得应该怎样”，但没有把 underlying reasoning 展开。

### 为什么这是问题

只给结论，别人没法判断：

- 你到底是基于什么 assumption
- 你考虑了哪些 trade-off
- 你的 proposal 会怎么执行
- 你想优化的 objective 是什么

结果 push back 会变成 opinion clash，而不是 design discussion。

### Correct rule

push back 用 PREP 结构：

1. **Problem**：我不同意什么
2. **Reason**：我为什么不同意
3. **Execution**：如果按我的方案做，具体怎么落地
4. **Payoff**：这样做带来的收益是什么

### Concrete example

不要只说：

> 我觉得 `do_sample` 应该先留在 `RequestData`。
> 

要说：

> 我认为把 `do_sample` 暴露在 interface 里，虽然当前未实现，但可以降低 Week 3 改 interface 的成本。具体做法是先保留 field，并在 execution path 里显式 reject unsupported requests。收益是 contract 更稳定。
> 

然后再认真面对 counter-argument：这是不是在暴露 fake capability？

### 下周 guardrail

每次想说“我觉得”，强制补一句“because …”。如果补不出来，说明 reasoning 还没成型。

## 问题 5：提交没跑过的 test

---

### Failure pattern

改完 code 之后，会有一种“我觉得这次应该过”的冲动，然后想直接 commit / push。

### 为什么这是问题

systems code 一旦引入 async、queue、状态机，很多 bug 都不是肉眼扫一遍能发现的。没有 test evidence 的“我觉得没问题”，基本没有信息量。

### Correct rule

commit 前必须看到 test output，而不是脑内假设它会通过。

### Concrete example

标准流程应该是：

1. 改完代码
2. 跑 `pytest`
3. 如果有 fail，先修
4. 全绿后再 commit

如果某个 test 属于 Week 3 scope，那就：

- 暂时 skip
- 或者删掉，等 scope 到了再补

不能带着 failing test 进入主线。

### 下周 guardrail

把“`git commit` 前先 `pytest`”变成硬规则，不再靠自觉。

## 问题 6：Scope creep in commits

---

### Failure pattern

在做一个小 task 的时候，顺手把另一个 future task 的 scaffold 也改进去，导致一个 commit 里混了多个 intent。

### 为什么这是问题

这样会直接破坏：

- reviewability
- reversibility
- commit message 的真实性
- 自己回头看历史时的可读性

更现实一点说，面试官如果看 commit history，混乱的 commit 也会暴露你当时的 execution discipline 不够稳。

### Correct rule

一个 commit 只做一件事；commit message 描述的是 **actually changed what**，不是 **I intended to do what**。

### Concrete example

如果这次只是做 Task 2 cleanup，但 diff 里已经混进了 `LMEngine` 的 scaffold，那就应该：

- unstage 不相关内容
- 用 `git add -p` 只留当前 task 的改动
- 未来 task 单独一个 commit

### 下周 guardrail

每次 commit 前强制看：

```
git diff --staged
```

如果一个 commit 讲不清楚“它只完成了哪一件事”，就先别 commit。

---

## 本周问题总结

如果把这周暴露出来的问题抽象成一句话，就是：

> 我容易在 design 还没 fully justified 的时候，提前把东西写进 code。
> 

具体表现为：

- abstraction 先行
- design gate 缺失
- 文档不同步
- reasoning 不展开
- test discipline 不够硬
- commit scope 控制不稳

这些都不是语法问题，也不是 API 熟练度问题，而是 **engineering execution discipline** 的问题。

这反而是好事，因为它说明现在最该修的不是“多学几个框架”，而是把系统设计和实现之间的工作流变硬。

---

## 下周最该盯住的 6 条规则

1. **No abstraction without a concrete execution problem.**
2. **No code before a clear design intent.**
3. **No ADR drift before merge.**
4. **No push back without explicit reasoning.**
5. **No commit without passing tests.**
6. **No mixed-intent commit.**

如果下周能把这 6 条执行稳，系统复杂度一上来，你的 code quality 和 design clarity 才不会一起崩。
