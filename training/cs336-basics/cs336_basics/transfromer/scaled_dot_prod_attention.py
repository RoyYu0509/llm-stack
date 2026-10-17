import math
import torch
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, reduce, repeat, einsum
from cs336_basics import nvtx_compat as nvtx


def softmax(input: torch.Tensor, axis: int = -1):
    """
    Compute softmax along a specified axis, with numerical stability.

    Parameters:
        input: Tensor of any shape
        axis: Dimension along which to apply softmax
    """
    # 对每行减去最大值, 避免 exp 过大溢出
    max_val, _ = torch.max(input, dim=axis, keepdim=True) # Keep dim for Broadcasting
    input_stable = input - max_val

    # 在 safe matrix 下, 用 exp 和 sum 计算 softmax
    exp_x = torch.exp(input_stable)
    sum_exp = torch.sum(exp_x, dim=axis, keepdim=True)
    softmax_output = exp_x / sum_exp

    return softmax_output

@nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(query, key, value, is_causal: bool = False, attn_mask=None):
    """
    Return an output with the shape (batch_size, seq_len, d_model)

    Accepts 3D packed format:
    key:        (batch, seq_k, d_model)
    query:      (batch, seq_q, d_model)
    value:      (batch, seq_v, d_model)
    is_causal:  bool

    Return:
        - attention: Float[Tensor, "batch seq_q d_model"]
    """
    # Compute Normalized QtK & Apply causal mask
    # Note: 如果是 decode 阶段, seq_q=1, seq_k = cached_len + 1; norm_qk: shape = (..., 1, cached_len + 1)
    norm_qk: Float[torch.Tensor, "... seq_q, seq_k"]
    norm_qk = einsum(key, query, "... seq_k d_k, ... seq_q d_k -> ... seq_q seq_k") / math.sqrt(key.shape[-1])  # python float: no per-call host->device scalar copy / sync
    # query 能不能看到比它更靠后的 key: 用来区分 prefill 和 decode 阶段; prefill 有 causal_mask, decode 没有 causal_mask
    if is_causal:
        # 拿 sequence 长度 
        seq_q = query.shape[-2]
        seq_k = key.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_q, seq_k, device=query.device, dtype=torch.bool))
        norm_qk = norm_qk.masked_fill(~causal_mask, torch.finfo(norm_qk.dtype).min)

    # 这个 key 槽位是不是真的 token
    if attn_mask is not None:
        norm_qk = norm_qk.masked_fill(~attn_mask, torch.finfo(norm_qk.dtype).min)

    # Softmax
    with nvtx.range("Computing softmax"):
        softmax_qk: Float[torch.Tensor, "... seq_q seq_k"]
        softmax_qk = softmax(norm_qk, axis=-1)

    # Attention
    with nvtx.range("Multiplied V"):
        attention: Float[Tensor, "... seq_q d_v"]
        attention = einsum(softmax_qk, value, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")

    return attention

####################################################################
# Add Different FlashAttention Kernels 
####################################################################

from cs336_systems.FlashAttention.flash_attention_torch_vectorized import vectorized_attn_torch_fn
@nvtx.range("Vectorized-Attention-Torch")
def vectorized_attention_torch(query, key, value, is_causal: bool = False):
    return vectorized_attn_torch_fn(query, key, value, is_causal)  # Positional args only

# Triton kernel requires a CUDA GPU (compute capability 8.0+) and the `triton` package,
# neither of which exists on Apple Silicon / other non-CUDA machines. Import defensively
# so the rest of this module (and everything depending on it, e.g. TransformerLM) stays
# importable for local CPU/MPS development; only calling flash_attention_my_triton on
# such a machine raises, with the original import error attached.
try:
    from cs336_systems.FlashAttention.flash_attention_triton import flash_attn_triton_fn
    _TRITON_IMPORT_ERROR = None
except Exception as e:
    flash_attn_triton_fn = None
    _TRITON_IMPORT_ERROR = e

@nvtx.range("FlashAttention-MyTriton")
def flash_attention_my_triton(query, key, value, is_causal: bool = False):
    if flash_attn_triton_fn is None:
        raise RuntimeError(
            "flash_attention_triton kernel unavailable on this machine (requires a CUDA GPU + triton)."
        ) from _TRITON_IMPORT_ERROR
    return flash_attn_triton_fn(query, key, value, is_causal)  # Positional args only


####################################################################
# Kernel Registry — string → callable look-up for pipeline scripts
####################################################################
ATTENTION_KERNEL_REGISTRY: dict[str, callable] = {
    "scaled_dot_prod_attention": scaled_dot_product_attention,
    "scaled_dot_prod_attention_compiled": torch.compile(scaled_dot_product_attention),
    "vectorized_torch": vectorized_attention_torch,
    "vectorized_torch_compiled": torch.compile(vectorized_attention_torch),
    "flash_attention_triton": flash_attention_my_triton,
}
