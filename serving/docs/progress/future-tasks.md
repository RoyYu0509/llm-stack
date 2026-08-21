
### Task 1: Implement Inference Data Class
- **What:** 创建一个专门用于管理 正在跑 inference 的 那些 request data, 包括 request id, 当前的token_sequence, KV cache id, status. 考虑重新设计TokenizedData class, 让它更适合 inference 阶段的需求。
- **Why:** 我们必须要有一个 data structure 可以让我们在 Engine 内部去管理跑那些 data 的 inference. 这个 data structure 需要包含 inference 阶段需要的所有信息（比如 KV cache id, request id），并且要方便我们在 prefill 和 decode 阶段更新系统信息。
- **Estimated Effort:** 2 hr


### Task 2: Implement Storage Class
- **What:** 实现一个 DecodeStorage class, 专门用来存储正在 prefill 和 decode 的 request 的 token sequence 和 KV cache. 这个 class 需要提供接口来更新 token sequence 和 KV cache, 以及查询当前的状态。
- **Why:** 因为我们需要一个 storage 可以让 scheduler 从里面拿 所有处于 prefill 和 decode 阶段的 Inference Data, 然后让 LM Compute 可以把 generate 更新后的 Inference Data 存回去。这个 storage 是 scheduler 和 LM Compute 之间的桥梁。
- **Estimated Effort:** 2 hr


### Task 3: Implement Scheduler Skeleton
- **What:** 实现一个 Scheduler class 的 skeleton, 可以 schedule 下一步往 LM Compute 中送哪个 Inference data. 先实现一个非常简单的 scheduling 策略（比如 FIFO），后面再慢慢优化。
- **Why:** Scheduler 是整个系统的核心，负责规划 每一步 compute 那些 data. 现在先搭好框架, 确认 data flow 是通的.
- **Estimated Effort:** 3 hr


### Task 4: Implement Batcher
- **What:** 这个 batcher 处于 scheduler 之后, 负责把 scheduler 选中的那些 Inference data 组装成一个 Batch. 
- **Why:** 因为 scheduler 选出来的这些 inference data 长短不一样, 直接送到 LM Compute 是不行的. 需要一个 batcher 来做 padding 和 batch assembly.
- **Estimated Effort:** 1 hr
