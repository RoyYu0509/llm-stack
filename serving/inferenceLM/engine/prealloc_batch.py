"""Decode batch over a PRE-ALLOCATED KV cache (the alternative to RaggedDecodeBatch).

RaggedDecodeBatch keeps one rectangular, left-padded `(B, H, width, d)` tensor per layer and rebuilds it
with `torch.cat` on every step (and on every add / remove): the whole batch's KV is read and written
once per decode step, and every row pays for the widest row's padding.

Here the cache is allocated ONCE:

    k[l], v[l] : (n_slots, max_len, H, d)      per layer, never re-created
    lens       : (n_slots,) int32             tokens already in each slot   (GPU copy + a CPU mirror)

  add(rid, past)  copy the prompt's K/V into a free slot (one request, once)
  step(tokens)    ONE forward for all rows; each row's new K/V is written at position lens[slot] in place
                  by the attention kernel, which then attends to that row's own prefix only
  remove(rid)     hand the slot back (O(1), no copy)

No padding, no mask, no per-step reallocation. The price is memory reserved for `max_len` per slot up
front (no paging).

The class has the same surface the engine uses on RaggedDecodeBatch (add / remove / `in` / len / step) plus
`step_tokens`, which does the greedy sampling for the whole batch: one batched argmax and ONE device->host
copy per step instead of one `.item()` (one sync) per row.
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

import torch
import torch.nn as nn

KV = List[Tuple[torch.Tensor, torch.Tensor]]


class PreallocDecodeBatch:
    def __init__(self, model: nn.Module, n_slots: int, max_len: int, dtype: torch.dtype,
                 device: torch.device | str, impl: str = "auto", compile_decode: bool = False,
                 compile_backend: str = "inductor") -> None:
        """`compile_decode`: run the whole decode forward as one `torch.compile`d graph (fullgraph, dynamic batch
        size). Same weights, same math, fewer and fused kernels; the attention call goes through the
        `llmstack::kvcache_attention` custom op. `compile_backend="aot_eager"` is for CPU tests (traces, no codegen)."""
        if not getattr(model, "supports_prealloc_decode", False):
            raise RuntimeError("model has no forward_decode_kvcache (prealloc decode needs TransformerLM.forward_decode_kvcache)")
        self.model, self.n_slots, self.max_len, self.impl = model, n_slots, max_len, impl
        self.device = torch.device(device)
        n_layers = len(model.tf_layers)
        heads = model.tf_layers[0].MHA.heads_num
        d_head = model.tf_layers[0].MHA.d_k
        shape = (n_slots, max_len, heads, d_head)
        # zeros, not empty: positions past a row's length are never read by the kernel, but uninitialised
        # memory can hold NaN and the reference implementation / debugging must be deterministic.
        self.k = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(n_layers)]
        self.lens = torch.zeros(n_slots, dtype=torch.int32, device=self.device)   # GPU: fed to the kernel
        self._lens_cpu = [0] * n_slots                                            # CPU mirror: no sync to read
        self._free: List[int] = list(range(n_slots - 1, -1, -1))
        self._slot: Dict[Hashable, int] = {}
        self._pin = self.device.type == "cuda"
        self.stats = dict(steps=0, rows=0, adds=0, removes=0, compiled=int(compile_decode))
        self._decode = model.forward_decode_kvcache
        self.compiled = compile_decode
        if compile_decode:
            self._decode = torch.compile(model.forward_decode_kvcache, backend=compile_backend,
                                         fullgraph=True, dynamic=True)
            self.warm_compile()

    # --- surface shared with RaggedDecodeBatch -------------------------------------------------
    def __len__(self) -> int:
        return len(self._slot)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._slot

    @property
    def keys(self) -> List[Hashable]:
        return list(self._slot)

    def add(self, key: Hashable, row_past: KV) -> None:
        """Copy a request's prefill K/V (per layer `(1, H, S, d)`) into a free slot."""
        if key in self._slot:
            raise KeyError(f"row {key!r} is already in the batch")
        if not self._free:
            raise RuntimeError(f"no free KV slot: n_slots={self.n_slots} are all in use")
        S = row_past[0][0].shape[-2]
        if S >= self.max_len:
            raise RuntimeError(f"prompt of {S} tokens does not fit max_len={self.max_len}")
        slot = self._free.pop()
        for l, (k, v) in enumerate(row_past):
            self.k[l][slot, :S] = k[0].transpose(0, 1).to(self.k[l].dtype)
            self.v[l][slot, :S] = v[0].transpose(0, 1).to(self.v[l].dtype)
        self.lens[slot] = S
        self._lens_cpu[slot] = S
        self._slot[key] = slot
        self.stats["adds"] += 1

    def remove(self, key: Hashable) -> None:
        slot = self._slot.pop(key)
        self._free.append(slot)
        self._lens_cpu[slot] = 0
        self.stats["removes"] += 1

    def warm_compile(self) -> None:
        """Trigger the compiles before any request is timed. A batch of one is specialised separately from a batch
        of two or more (size-1 dims are never symbolic), so both are run. Dummy rows use FREE slots at length 0:
        the only side effect is a write at position 0 of those slots, which `add` overwrites before they are read."""
        for B in (1, 2):
            if len(self._free) < B:
                continue
            slots = self._free[-B:]
            tok = torch.zeros(B, 1, dtype=torch.long, device=self.device)
            idx = torch.tensor(slots, dtype=torch.int32, device=self.device)
            seqlens = torch.zeros(B, dtype=torch.int32, device=self.device)
            with torch.no_grad():
                self._decode(tok, seqlens.long().unsqueeze(1), self.k, self.v, seqlens, idx, impl=self.impl)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # --- one decode step -------------------------------------------------------------------------
    def _forward(self, tokens: Dict[Hashable, int]) -> Tuple[List[Hashable], torch.Tensor]:
        if set(tokens) != set(self._slot):
            raise KeyError(f"decode step got rows {sorted(map(str, tokens))}, batch has {sorted(map(str, self._slot))}")
        rids = list(tokens)
        slots = [self._slot[r] for r in rids]
        ctx = self.model.token_positions.shape[0]
        longest = max(self._lens_cpu[s] for s in slots)          # CPU mirror: this check costs no device sync
        if longest + 1 > min(self.max_len, ctx):
            raise RuntimeError(f"decode would reach position {longest + 1}, beyond max_len={self.max_len} / context {ctx}")

        tok_cpu = torch.tensor([[int(tokens[r])] for r in rids], dtype=torch.long)
        idx_cpu = torch.tensor(slots, dtype=torch.int32)
        if self._pin:
            tok_cpu, idx_cpu = tok_cpu.pin_memory(), idx_cpu.pin_memory()
        tok = tok_cpu.to(self.device, non_blocking=True)
        idx = idx_cpu.to(self.device, non_blocking=True)
        seqlens = self.lens.index_select(0, idx.long())                  # (B,) int32: lengths BEFORE this token
        positions = seqlens.long().unsqueeze(1)                   # RoPE position of the new token = its row's length
        with torch.no_grad():       # grad mode is part of torch.compile's guards: one mode, one set of graphs
            logits = self._decode(
                tok, positions, self.k, self.v, seqlens, idx, impl=self.impl
            )                                                     # (B, 1, V); K/V appended in place, layer by layer
        self.lens.index_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.int32))
        for s in slots:
            self._lens_cpu[s] += 1
        self.stats["steps"] += 1
        self.stats["rows"] += len(rids)
        return rids, logits[:, 0, :]

    def step(self, tokens: Dict[Hashable, int]) -> Dict[Hashable, torch.Tensor]:
        """Same contract as RaggedDecodeBatch.step: {row: logits (V,)}."""
        rids, logits = self._forward(tokens)
        return {r: logits[i] for i, r in enumerate(rids)}

    def step_tokens(self, tokens: Dict[Hashable, int]) -> Dict[Hashable, int]:
        """Greedy decode for the whole batch: one argmax kernel, one device->host copy."""
        rids, logits = self._forward(tokens)
        nxt = logits.argmax(dim=-1).tolist()
        return dict(zip(rids, nxt))
