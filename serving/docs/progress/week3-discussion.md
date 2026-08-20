# Week 3 Design Discussion Log

**Date**: 2026-04-21
**Participants**: Yifan (intern), Manager (Claude)
**Context**: Pre-Week-3 scope review and continuous batching design meeting

---

## 1. Project Direction Change

### Trigger
Advisor feedback: project should serve real lab needs, not be a vLLM clone. Deep research report reviewed (from separate chat) recommending pivot to "research-serving runtime."

### Decision
**Project goal changed from "private vLLM clone" to "lab-grade research-serving runtime for local CoT models."**

Rationale:
- vLLM's full feature surface (PagedAttention, CUDA/HIP graphs, chunked prefill, multi-hardware, OpenAI-compatible server, etc.) is maintained by a large team over years. One intern in 8 weeks can only reach ~20% coverage across every layer.
- Research-serving runtime narrows scope to: continuous batching, lifecycle management, data collection — done **correctly and cleanly**.
- Portfolio framing: "designed and delivered a usable runtime for actual lab workloads" > "attempted to replicate vLLM but only finished the first 20%."

### What was adopted from the deep research report
- Direction: research-serving runtime (yes)
- Phased roadmap structure (yes, as backlog reference)
- Stage-homogeneous microbatch approach (yes)
- Lifecycle state machine design (yes)

### What was NOT adopted
- Week 3 timeline from report — over-scoped (8 tasks / ~20 hours including pluggable scheduler, hooks, traces, metrics)
- Pluggable scheduler interface — premature abstraction, no second policy exists yet
- TraceEvent / hooks / async trace writer — Phase B, not Week 3
- Rich schema redesign (RequestRecord 12 fields, BatchItem 8 fields) — incremental evolution preferred
- Metrics taxonomy — comes after the runtime works

---

## 2. Continuous Batching Design Decisions

### 2.1 Backend Constraint (CRITICAL)

**HuggingFace `model()` cannot mix prefill and decode in one forward pass.**

- HF attention_mask is `[batch_size, seq_len]` — a padding mask only
- Packing multiple sequences with `<BOS>` separators and per-pair causal masks requires custom attention kernels (FlashAttention with block tables, xformers)
- GPT-2's HF implementation does not support ragged/packed sequence input

**Consequence: stage-homogeneous batching only.**
- Prefill batch: `input_ids: [B_p, S_max]`, `past_key_values=None`
- Decode batch: `input_ids: [B_d, 1]`, `past_key_values=cached_kv`
- These are TWO separate `model()` calls per iteration, not one

### 2.2 Iteration Structure (LOCKED)

Each engine iteration:
1. **Prefill microbatch forward** — all pending new requests that are admitted this iteration
2. **Decode microbatch forward** — all in-flight requests that are currently decoding

Sequential execution, not GPU-parallel. Two `model()` calls.

### 2.3 Prefill-First Order (LOCKED)

**Decision: Prefill before decode in each iteration.**

- Optimizes TTFT (time-to-first-token): new requests get their first token in the same iteration they arrive
- Trade-off: decode ITL (inter-token latency) is worse — in-flight decode requests wait for prefill forward to complete before getting their next token
- Acceptable for current lab workload (no streaming, short prompts)

### 2.4 Scheduling Rule (OPEN — must be resolved in task breakdown)

The original 2:1 ratio was designed for either/or model (2 prefill-only iterations, 1 decode-only iteration). Under both-in-one-iteration model, this ratio no longer directly applies.

**Open question:** What is the admission policy per iteration? Options:
- Admit all pending prefill requests every iteration
- Cap at N prefill requests per iteration
- Some other rule

This must be addressed in the task breakdown.

### 2.5 Root Cause of Current Design Limitation

Current loop: one iteration = one request's full lifecycle (fetch → prefill → decode loop → done → next request). This couples request lifecycle to iteration lifecycle, preventing any work interleaving.

New loop: one iteration = one step for ALL in-flight requests + admission of new requests. Request lifecycle spans many iterations.

---

## 3. Week 3 Scope

### Goals
1. Engine handles 4+ in-flight requests simultaneously
2. New requests admitted between decode iterations
3. 2:1 prefill-prioritized scheduling (hardcoded in loop, no abstraction)
4. All requests → status DONE with correct generated_tokens
5. Lifecycle state machine: `INIT → RUNNING → DRAINING → STOPPED`
6. `shutdown(drain=True)` → reject new requests, drain in-flight, clean exit

### Explicit Non-Goals
- Pluggable scheduler interface
- Hooks / trace events / async writer
- Metrics collection
- Token-budget scheduling
- Chunked prefill
- Schema redesign (RequestRecord, BatchItem, TraceEvent from report)
- Multi-hardware support

### Backlog (deferred)
- P2: Clean up `do_sample` parameter (trigger: implement sampling)
- P2: RuntimeError match strings in tests
- Future: task_done() exception safety
- Future: Pluggable scheduler interface (trigger: need second policy)
- Future: Hooks/traces/metrics (trigger: runtime is stable and correct)
- Future: Token-budget scheduling, chunked prefill
- Future: Paged KV cache (trigger: move beyond GPT-2 124M)

---

## 4. Key Corrections Made During Discussion

| What Yifan said | What was wrong | Correct understanding |
|---|---|---|
| "每一轮判断是 decode 还是 prefill" | Implies either/or per iteration | Both can happen in one iteration (two forward passes) |
| Pack prefill+decode with `<BOS>` separator and mask | Requires custom attention kernel | HF model() only supports `[B, S]` padding mask, not per-pair causal mask |
| "GPU utilization 低" as root cause | That's the symptom | Root cause: request lifecycle coupled to iteration lifecycle |
| 2:1 ratio still applies directly | Under both-in-one-iteration model, original ratio semantics change | Need new admission policy definition |

---

## 5. Next Steps

1. **Yifan writes task breakdown** for Week 3 goal (sequenced tasks with What/Why/AC/Effort)
2. Task breakdown must include admission policy definition as a task
3. Manager reviews task breakdown before any code is written
4. After task breakdown approved, Yifan writes design proposal / ADR update
5. Retro debt: add "Delivery Status" section to Week 2 retro (async, not blocking)
