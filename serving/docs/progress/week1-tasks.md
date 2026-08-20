# Week 1 Goal：Request Receiver + Engine Request Intake — 从用户输入到 engine queue 的完整 pipeline 跑通。

## Tasks
### Task 1: Request Data Model

What：定义 Request 和 Response 的 data class。一个 Request 至少包含 request_id、user_id、prompt_text、timestamp、status（pending / prefilling / decoding / completed）。
Why：这是所有 module 之间的 contract。先定好 data model，后面 module 之间的 interface 才不会反复改。
Acceptance Criteria：data class 定义完成，有 type hints，有 unit test 验证 serialization/deserialization。
Estimated Effort：2 小时

### Task 2: Tokenizer Wrapper

What：封装一个 Tokenizer class，wrap HuggingFace 的 GPT-2 tokenizer。接口是 encode(text) -> list[int] 和 decode(token_ids) -> str。
Why：把 tokenizer 的具体实现隔离开。如果以后换 tokenizer（比如你自己的 BPE），只改这一个 class。
Acceptance Criteria：能正确 encode/decode GPT-2 的 vocabulary。Edge case test：空字符串、超长输入（超过 1024 tokens）、special characters。
Estimated Effort：2 小时

### Task 3: Request Receiver

What：实现一个 RequestReceiver class，能异步接收多个用户的 request。V0 不需要 HTTP server — 用 asyncio.Queue 模拟。提供 submit_request(prompt_text, user_id) -> request_id 接口。内部流程：创建 Request 对象 → tokenize → 放入 engine input queue。
Why：这是系统的 entry point。先把 input pipeline 跑通，后面 engine 才有数据可以处理。
Acceptance Criteria：并发 submit 10 个 request，每个都被正确 tokenize 并进入 queue，顺序与 submit 顺序一致。无 race condition。
Estimated Effort：4-5 小时

### Task 4: Engine Skeleton + Request Intake

What：实现 Engine class 的 skeleton。这周只实现 request intake 部分 — 从 input queue 读取 request，放入 scheduler 的 waiting queue。Engine 的 main loop 结构搭出来，但 compute 部分留空（Week 2-4 实现）。
Why：先把 engine 的骨架搭好，验证 Request Receiver 和 Engine 之间的 data flow 是通的。
Acceptance Criteria：启动 Engine，通过 Request Receiver submit 5 个 request，验证这 5 个 request 都出现在 scheduler 的 waiting queue 里，request 状态从 pending 变为 waiting。
Estimated Effort：4-5 小时

### Task 5: Integration Test

What：一个 end-to-end test，模拟多个用户同时 submit request，验证 Request Receiver → Tokenizer → Engine Input Queue 的完整 flow。
Why：验证 Week 1 的所有组件能正确协作。
Acceptance Criteria：3 个模拟用户，每个发送 3 个 request（共 9 个），全部正确进入 engine 的 waiting queue，无丢失、无重复、无乱序。
Estimated Effort：2-3 小时


## Week 1 Summary
***Week 1 总计：约 15-17 小时。按每天 5 小时，3-4 天完成***

### Week 1 Risk/Blocker： 

如果你没写过 Python asyncio，Task 3 会是 blocker。不需要系统性学 — 理解 async/await、asyncio.Queue、asyncio.gather 这三个概念就够用了。

### Deliverables
上面 5 个 task 全部完成并有 passing tests
一份简短的 weekly retro（放在 docs/progress/week1-retro.md）：什么比预期顺利 / 什么比预期难 / 下周要改进什么
