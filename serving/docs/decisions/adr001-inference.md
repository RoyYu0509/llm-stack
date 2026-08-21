# ADR-001: Inference Engine 的设计

**Date**: [04-15-2026] | **Status**: Proposed

## Context
1. Inference 用 HuggingFace 中的 model() Forward Pass 来做.
2. 整个 Inference Flow 的 Structure

## Decision
### Inference Engine 在接收到 request 之后怎么跑 inference?
**✨ Final Decision**
使用 HuggingFace 的 `model(input, kv_cache)` 来进行 inference. 好处是: 我们可以手动控制 input 是什么, 然后 output 是什么.

```
outputs = model(
    input_ids=input_ids,          # shape: [batch_size, seq_len]
    past_key_values=past_kv,      # 上一步存的 KV cache，第一次是 None
    use_cache=True,               # 让它返回 KV cache
)

logits = outputs.logits           # shape: [batch_size, seq_len, vocab_size]
new_kv = outputs.past_key_values  # 传给下一步用
```

**Alternatives**
> 1. 用 HuggingFace 的 model.generate() 来做 inference.
**问题:** 它不暴露中间状态（KV cache、per-step logits），导致你无法在 Week 3/4 实现 continuous batching 和 paged KV cache。这才是 rejection reason.
**好处**: 代码量少、不需要手动管理 KV cache、不需要自己写 stopping logic、HuggingFace 已经帮你处理了 edge case.

### Inference 是 Sync 还是 Async 的?
**✨ Final Decision**
当前实现是 cooperative scheduling (Weak Async)：asyncio.sleep(0) 让出 event loop 给其他 task（比如 RequestReceiver 的 put）。不是真正的 async GPU execution——CPU 在 tensor.item() 时仍会 block 等 GPU。真正的 GPU/CPU 并行需要 CUDA stream 或 event-based sync，defer 到 Week 3/4。

**Alternatives**
> 1. 设计成 sync function, 每次 fetch 一个 request, 跑完 inference loop, 再 fetch 下一个 request.
**问题:** CPU 会有大量空转时间, 因为每次submit完之后, CPU 都要停下来, 等GPU完成computation之后, 再去干别的事情, 比如 fetch request, send response 等等.


### Inference Engine 里需要几个 queue?
**✨ Final Decision**
Inference Engine 只保留一个 queue, pending queue. Request Receiver fetch request 的时候, 直接放到 pending queue 里. Inference Engine 从 pending queue 里拿 request 来跑 inference loop. 这样设计的好处是简单, 不需要在不同的 queue 之间同步 request 的状态.

**Alternatives**
> 1. 设计两个 queue, 一个是 pending queue, 一个是 waiting queue. 额外的 waiting_queue 用来存放等待跑 inference loop 的 request.
**问题:** 完全不需要这额外的 一层, 直接用 pending queue 简简单单.


### Inference Data Flow 的设计
**✨ Final Decision**
1. 把 prefill 和 decode 包装到一个 inference() 中, 跑完所有 decoding 后再出来. 再次期间把 request 的 status 更新成 PROCESSING.
2. KV Cache 只保留现在正在 running 的 request 的 KV Cache. 跑完一个 request 的时候, 就丢掉它的 KV Cache.
3. 生成 new_token 的时候, 用 greedy sampling, 也就是选概率最高的 token 作为 new_token, 因为这样是 deterministic的, 方便对比我们自己控制的 inference 和 model.generate() 是一致的.
4. Generating Stopping Criteria: 达到最大长度 或 生成<EOS> token.

**Alternatives:**
> 1. Prefill 和 Decode 的流程完全分开, prefill 完了之后, 把 request 放到 waiting queue 里, 等 scheduler 调度的时候再从 waiting queue 里拿出来继续 decode.
1. Prefill Step: 
   a. Input: TokenizedData Class
   b. 调用 model(input_ids, past_key_values=None) -> new_token_logits, new_kv
   c. 用 greedy sampling 从 new_token_logits 生成 new_token.
   d. 更新 request 的 status 从 WAITING → PREFILLING. (因为我们是从 Inference Engine 的 waiting_queue 里拿到 request 的, 所以最开始的 status 是 WAITING)
   e. Output: new_token 和 new_kv

2. Decoding Step:
   a. Input: 上一步的 new_token 和 new_kv
   b. 更新 request 的 status 从 PREFILLING → DECODING. 
   c. WHILE: 调用 model(new_token, past_key_values=new_kv) -> logits, new_kv, 用 greedy sampling 从 new_token_logits 生成 new_token.
   d. 当达到最大长度 或 生成<EOS>时: Break Decoding WHILE loop.
   e. 更新 request 的 status 从 DECODING → DONE. 
   f. Output: new_token_sequence

3,4,5. 同上

**好处:** 这个设计的好处是清晰, prefill 和 decode 的流程完全分开, 而且可以准确的更新 request 的 status -> PREFILLING / DECODING.
**问题:** 不选这个是因为 week2 的主要任务是跑通 inference 的 end-to-end data flow. 我们会在 week3/4 的时候, 当需要 schedule 和 batching 的时候我们会重新实现.


### InferenceEngine.run() 需要 return asyncio.Task 吗?
**✨ Final Decision**
InferenceEngine.run() return asyncio.Task(_keep_get_request_and_inference()). 这样create_task spawn task 然后立刻 return。

**Alternatives:**
> 1. InferenceEngine.run() 直接 await _keep_get_request_and_inference()，不 return task.
**问题:** 如果这样设计的话, run() function 就一直不会 return, 因为 _keep_get_request_and_inference() 是一个 while True 的 loop, 只有当 engine 被 kill 的时候才会 break. 这样就导致我们在 await run() 的时候, Event loop 就被 run() 给占住了, 我们甚至无法 kill run() 内部的 _keep_get_request_and_inference 中的 while loop, 因为我们根本没有机会去调用 kill() function (必须在下一行).

## Consequences
1. 现在的 Inference Engine 可以处理 单个 request 的 prefill -> decoding 的完整流程, KV Cache 用完就丢, 不需要存储.
