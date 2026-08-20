# Project Backlog

> Last updated: 2026-04-16 | Owner: Yifan Yu

---

## Status Legend

- 🔴 **P0** — This week, blocking
- 🟠 **P1** — Next, high value
- 🟡 **P2** — Later, nice to have
- 🧊 **Icebox** — Someday/maybe, not committed

Item types: `[FEAT]` feature · `[BUG]` bug · `[TECH]` tech debt · `[SPIKE]` investigation · `[CHORE]` maintenance · `[DOC]` documentation

---

## 🔴 P0 — In Progress / This Week
  
### [CHORE] Engine lifecycle management
- **Why**: 目前 `keep_fetching_requests` 这个 task 在测试结束时不会 graceful shutdown，依赖 pytest event loop cleanup 做兜底。需要 engine 提供明确的 shutdown 机制（cancel pending tasks + drain queue）来确保资源正确释放，避免潜在的 memory leak 或 dangling tasks。比如: (init / start / shutdown 三个 state 的 state machine).
- **Acceptance criteria**: Engine 提供 `shutdown()` 方法，能够取消所有 pending tasks 并清空 queue。测试结束时调用 `shutdown()`，确保没有未完成的 tasks 或未处理的 requests。
- **Estimate**: M
- **Owner**: @yifan
- **Trigger**: Week 3 做 continuous batching 时，scheduler 和 fetcher 的 lifecycle 需要统一管理，这时需要这个 shutdown 机制来确保不同组件能够正确协同工作

### [BUG] Inference 中要用 try / finally 
- **Why**: 在 inference loop 里，fetch request 之后的 processing 可能会抛异常（比如 GPU out of memory）。如果没有 try/finally 来保证 `pending_queue.task_done()` 一定被调用，那么这个 request 就会一直卡在 pending queue 里，导致后续的 requests 无法被处理，整个 engine 卡死。
- **Acceptance criteria**: 尝试不同的 Exception 方法, 保证 inference loop 里无论成功还是失败, 都能正确调用 `pending_queue.task_done()`, 确保 pending queue 最后会 empty. 
- **Estimate**: S
- **Owner**: @yifan
- **Trigger**: Week 3 做 continuous batching 时，单个 batch 里任何一个 request 的异常都不应该影响整个 batch 的处理，所以需要这个 try/finally 来保证异常被正确捕获和处理。

---

## 🟠 P1 — Next Up

---

## 🟡 P2 — Later

### [CHORE] 清理 do_sample 参数
- **Why**: Week 2 prefill/decode 签名里有 do_sample 参数但只支持 do_sample=False，do_sample=True 直接 NotImplementedError。参数本身在 Week 2 不起作用，但为未来 sampling 策略预留接口。
- **Acceptance criteria**: 实现至少一种 sampling（top-k 或 top-p），do_sample=True 不再 raise
- **Estimate**: S
- **Owner**: @yifan
- **Trigger**: Week 3 或有实际需求时

### [TECH] Test 里的 RuntimeError match 字符串不美观
- **Why**: 现在是直接 match 整个 error message, 但是这个 message 里有一些 dynamic content (比如 max_length 的值), 导致测试不够 robust. 我们应该只 match error type 和 message 的 static part.
- **Acceptance criteria**: RuntimeError 的测试改成 match error type + static part of message
- **Estimate**: S
- **Owner**: @yifan
- **Trigger**: 不急，等有空了再改

---

## 🧊 Icebox


---

## ✅ Done (last 2 weeks)

Move completed items here for visibility. Archive monthly.


---

## 📝 Grooming Notes

> Updated during weekly review. Capture decisions, deferrals, and re-prioritization rationale.

