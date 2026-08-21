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

### #001 [CHORE] Short title here
- **Why**: One sentence on the value or problem this solves.
- **Acceptance criteria**:
  - [ ] Criterion 1 (testable, specific)
  - [ ] Criterion 2
- **Estimate**: S / M / L (or hours)
- **Owner**: @yifan
- **Notes**: Link to design doc / related issue / blocker

### #002 [BUG] Short title here
- **Repro**: Steps or command to trigger the bug
- **Expected**: What should happen
- **Actual**: What happens instead
- **Acceptance criteria**:
  - [ ] Bug no longer reproduces
  - [ ] Regression test added
- **Estimate**: S
- **Owner**: @yifan

---

## 🟠 P1 — Next Up

### #003 [FEAT] Short title here
- **Why**:
- **Acceptance criteria**:
  - [ ]
- **Estimate**:
- **Depends on**: #001

### #004 [TECH] Short title here
- **Why**:
- **Acceptance criteria**:
  - [ ]
- **Estimate**:

---

## 🟡 P2 — Later

### #005 [SPIKE] Short title here
- **Question to answer**: What specifically are we trying to learn?
- **Output**: Design doc / benchmark / ADR
- **Time-box**: 1 day max
- **Estimate**: S

### #006 [CHORE] Short title here
- **Why**:
- **Estimate**:

---

## 🧊 Icebox

- **#007 [FEAT]** Multi-GPU inference support — wait until single-GPU stable
- **#008 [FEAT]** Prometheus metrics endpoint — only if we deploy beyond local
- **#009 [DOC]** Architecture overview blog post — post-MVP

---

## ✅ Done (last 2 weeks)

Move completed items here for visibility. Archive monthly.

- **#000 [FEAT]** Initial `pyproject.toml` setup with `uv` — 2026-04-10
- **#-01 [TECH]** Migrated test fixtures to `parametrize` — 2026-04-12

---

## 📝 Grooming Notes

> Updated during weekly review. Capture decisions, deferrals, and re-prioritization rationale.

- **2026-04-16**: Deferred #007 (multi-GPU) to icebox — single-GPU correctness still unstable, multi-GPU adds debugging surface area without clear ROI yet.
- **2026-04-09**: Split original "build inference server" into #001 / #003 / #004 — too coarse to estimate.
