"""One decode step of attention against a PRE-ALLOCATED KV cache. Inference only.

Layout (the one FlashAttention's `flash_attn_with_kvcache` uses):
    k_cache, v_cache : (n_slots, max_len, n_heads, d_head)   allocated once, never re-created
    cache_seqlens    : (B,) int32   how many keys each batch row already has in its slot
    cache_batch_idx  : (B,) int32   which slot each batch row lives in (None = rows 0..B-1)
    q, k_new, v_new  : (B, 1, n_heads, d_head)   the single new token of every row

The new key/value are written INTO the cache at position `cache_seqlens[b]` (in place, no copy of
the prefix), then the query attends to the first `cache_seqlens[b] + 1` keys of its own slot.

Two implementations with the same contract:
  * `flash`     -- flash_attn.flash_attn_with_kvcache (FlashAttention-2, split-KV decode). CUDA, fp16/bf16.
  * `reference` -- plain PyTorch (fp32 softmax). Slow (gathers the slots), runs anywhere; it exists so the
                   cache bookkeeping can be unit-tested on CPU and so the flash path can be checked against it.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def kvcache_attention_reference(q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx=None):
    B, _, H, D = q.shape
    dev = q.device
    idx = (cache_batch_idx.long() if cache_batch_idx is not None else torch.arange(B, device=dev))
    pos = cache_seqlens.long()
    # in-place append of the new token (what the flash kernel does when k / v are passed)
    k_cache[idx, pos] = k_new[:, 0].to(k_cache.dtype)
    v_cache[idx, pos] = v_new[:, 0].to(v_cache.dtype)
    kc, vc = k_cache[idx], v_cache[idx]                       # (B, max_len, H, D)   gather = reference only
    keys = torch.arange(kc.shape[1], device=dev)
    valid = keys.unsqueeze(0) <= pos.unsqueeze(1)             # (B, max_len): own prefix + the new key
    # slots past a row's length hold whatever was there before (possibly NaN / inf): 0 * NaN is NaN, so the
    # unused positions must be zeroed, not just given probability 0
    kc = kc.masked_fill(~valid[:, :, None, None], 0)
    vc = vc.masked_fill(~valid[:, :, None, None], 0)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), kc.float()) / math.sqrt(D)
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), vc.float())
    return out.to(q.dtype)


def kvcache_attention_flash(q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx=None):
    from flash_attn import flash_attn_with_kvcache  # lazy: not installed on CPU / Apple machines

    return flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k_new, v=v_new, cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx, causal=True,
    )


def _kvcache_attention_eager(q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx, impl):
    if impl == "auto":
        impl = "flash" if (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)) else "reference"
        if impl == "flash":
            try:
                import flash_attn  # noqa: F401
            except Exception:
                impl = "reference"
    fn = kvcache_attention_flash if impl == "flash" else kvcache_attention_reference
    return fn(q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx)


# torch.compile cannot trace into flash_attn's C++ entry point (and must be told that the caches are written in
# place), so under compilation the call goes through this registered op. `fake` only says "same shape as q".
@torch.library.custom_op("llmstack::kvcache_attention", mutates_args=("k_cache", "v_cache"))
def _kvcache_attention_op(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, k_new: torch.Tensor,
                          v_new: torch.Tensor, cache_seqlens: torch.Tensor, cache_batch_idx: torch.Tensor,
                          impl: str) -> torch.Tensor:
    return _kvcache_attention_eager(q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx, impl)


@_kvcache_attention_op.register_fake
def _kvcache_attention_op_fake(q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx, impl):
    return torch.empty_like(q)


def kvcache_attention(q, k_cache, v_cache, k_new, v_new, cache_seqlens,
                      cache_batch_idx: Optional[torch.Tensor] = None, impl: str = "auto"):
    """`impl`: "auto" (flash on CUDA fp16/bf16 when importable, else reference), "flash" or "reference"."""
    if torch.compiler.is_compiling():
        if cache_batch_idx is None:
            cache_batch_idx = torch.arange(q.shape[0], device=q.device, dtype=torch.int32)
        return _kvcache_attention_op(q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx, impl)
    return _kvcache_attention_eager(q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx, impl)
