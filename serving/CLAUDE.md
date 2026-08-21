# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`llm-serving` (package name `ai-internship-project` in `pyproject.toml`) is a from-scratch LLM inference serving system, built incrementally as a portfolio project. Originally scoped as a "private vLLM clone," it was **re-scoped in Week 3** (see `docs/progress/week3-discussion.md`) to a narrower "research-serving runtime for local CoT models": continuous batching, request lifecycle management, and data collection, done correctly rather than attempting broad vLLM feature parity.

The current code implements a working end-to-end pipeline — request → tokenize → prefill/decode → generated tokens — for **one request at a time**. Continuous batching, the scheduler, and a real KV cache manager (described in `docs/designs/DESIGN.md`) are Week 3+ work and not yet implemented; today `InferenceEngine` runs a request's full prefill→decode loop to completion before dequeuing the next one. Check `BACKLOG.md` and `docs/progress/` for the current state before assuming a design doc is already built.

## Commands

**Setup:**
```bash
uv sync
source .venv/bin/activate
```

**Testing:**
```bash
uv run pytest                                     # all tests
uv run pytest tests/inference_engine_test.py -v   # single test file
uv run pytest tests/lm_engine_test.py::test_name  # single test
```
Tests exercise real HuggingFace models (`gpt2` via `GPT2LMHeadModel`/`GPT2Tokenizer`, loaded once per file with module-scoped fixtures) rather than mocking the model — first run downloads weights. Async engine tests share one pattern: start the engine with `task = engine.run()` (returns an `asyncio.Task` without awaiting it), `await pending_queue.join()`, then `task.cancel()`.

**Lint / typecheck** (declared as dev deps; no CI or config file wires these up yet):
```bash
uv run ruff check .
uv run mypy inferenceLM
```

## Architecture

Request lifecycle: `RequestReceiver` → shared `asyncio.Queue` (`pending_queue`) → `InferenceEngine` → `LMEngine`. Two objects are passed by reference into both `RequestReceiver` and `InferenceEngine` at construction time and act as the shared state between them: `pending_queue: asyncio.Queue[TokenizedData]` and `request_store: Dict[str, RequestData]`.

- **`inferenceLM/data/`**
  - `request.py` — `RequestData`, a pydantic `BaseModel` holding full per-request lifecycle state (`user_id`, `request_id`, `status`, `prompt_text`, `generated_tokens`, ...); serialized via pydantic JSON.
  - `request_status.py` — `RequestStatus` enum. Only `PENDING → PROCESSING → DONE/FAILED` are used today; `PREFILLING`/`DECODING` are reserved for the split-phase scheduler that doesn't exist yet.
  - `tokenized_data.py` — `TokenizedData` (`request_id` + `tokens`), deliberately kept separate from `RequestData` so the receiver layer and engine layer don't share a bloated type (rationale in `docs/decisions/adr000-request.md`).

- **`inferenceLM/request_receiver/`**
  - `request_receiver.py` — `RequestReceiver.submit_request()` builds a `RequestData`, stores it in `request_store` keyed by `request_id`, tokenizes the prompt, and pushes a `TokenizedData` onto `pending_queue`.
  - `tokenizer.py` — thin wrapper around `transformers.AutoTokenizer`.

- **`inferenceLM/engine/`**
  - `inference_engine.py` — `InferenceEngine._keep_get_request_and_inference()` is the consume loop: pulls from `pending_queue`, calls `LMEngine.inference()`, writes `generated_tokens`/`status` back onto the matching `request_store` entry, and always calls `pending_queue.task_done()` in a `finally` block so a failed request doesn't wedge the queue. `run()` returns the `asyncio.Task` immediately instead of awaiting it — awaiting would block the event loop forever since the loop only exits via `kill()` (rationale in `docs/decisions/adr001-inference.md`).
  - `lm_engine.py` — `LMEngine` calls the HuggingFace model directly (`model(input_ids, past_key_values=...)`) rather than `model.generate()`, to keep the KV cache and per-step logits controllable for future batching (ADR-001). `prefill()` runs the first forward pass; `decode()` repeats with `past_key_values`; `inference()` drives prefill→decode with greedy sampling (`do_sample=True` raises `NotImplementedError` — tracked in `BACKLOG.md`) until `stopping_criteria()` (max length or EOS) fires. KV cache is per-request and dropped when the request finishes; there is no cross-request cache reuse or pre-allocation yet.

- **`inferenceLM/output/`** — empty; reserved for the detokenize/stream-back module in `docs/designs/DESIGN.md`.

### Where the design is heading

- `docs/designs/DESIGN.md` — the original v0 design (Scheduler, Continuous Batcher, static KV-cache pre-allocation sizing, milestone plan). Approved 2026-04-09, before the Week 3 re-scope — treat its scheduling/batching sections as directional, not current spec.
- `docs/progress/week3-discussion.md` — supersedes DESIGN.md on batching/scheduling specifics: HF's `model()` can't mix prefill and decode tokens in one forward pass (no packed/ragged attention support), so batching must be **stage-homogeneous** (one prefill microbatch forward + one decode microbatch forward per iteration, prefill-first for TTFT). This is the most current source of truth for engine-loop design; read it before touching scheduling/batching code.
- `docs/decisions/adr000-request.md`, `adr001-inference.md` — why `RequestData`/`TokenizedData` are split, why inference drives the model manually instead of via `.generate()`, why `run()` returns a bare `Task`.
- `BACKLOG.md` — current priorities and known gaps (e.g., engine has no graceful `shutdown()`/lifecycle state machine yet).

## Git Conventions
- Commit format: `type(scope): description`
- Types: feat, fix, refactor, test, docs, chore, perf
- One logical change per commit. Never mix refactor + feature.
- Branch naming: `feature/short-desc`, `fix/short-desc`, `docs/short-desc`
- All changes go through PR to main. Never push directly to main.

## Code Standards
- Python 3.12+, C++20+ (C++ is planned for performance-critical components per the tech stack; none exists in the repo yet)
- Max function length: 50 lines. Extract if longer.
- All public APIs must have documentation comments.
- No commented-out code in commits.
- No TODO without linked issue number.

## Forbidden
- Do not modify files outside `inferenceLM/` and `tests/` without explicit discussion.
- Do not install new dependencies without documenting rationale.
- Do not auto-generate large amounts of boilerplate — prefer understanding over speed.

## Development Workflow
- CPU/MPS functional testing during development.
- Benchmarking & production target Nvidia GPU (CUDA).
