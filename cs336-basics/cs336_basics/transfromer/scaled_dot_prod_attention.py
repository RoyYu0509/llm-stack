import torch
from jaxtyping import Float, Array
from torch import Tensor
from einops import rearrange, reduce, repeat, einsum
import torch.cuda.nvtx as nvtx


def softmax(input: torch.Tensor, axis: int = -1):
    """
    Compute softmax along a specified axis, with numerical stability.

    Parameters:
        input: Tensor of any shape
        axis: Dimension along which to apply softmax
    """
    # Subtract max for numerical stability
    max_val, _ = torch.max(input, dim=axis, keepdim=True) # Keep dim for Broadcasting
    input_stable = input - max_val

    # Compute exp and normalize
    exp_x = torch.exp(input_stable)
    sum_exp = torch.sum(exp_x, dim=axis, keepdim=True)
    softmax_output = exp_x / sum_exp

    return softmax_output

@nvtx.range("scaled dot product attention")
def scaled_dot_product_attention(query, key, value, is_causal: bool = False):
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
    with nvtx.range("Compute QK^t"):
        norm_qk: Float[torch.Tensor, "... seq_q, seq_k"]
        norm_qk = einsum(key, query, "... seq_k d_k, ... seq_q d_k -> ... seq_q seq_k") / torch.sqrt(torch.tensor(key.shape[-1], device=query.device))
        # Triangular Causal Mask
        if is_causal:
            seq_q = query.shape[-2]
            seq_k = key.shape[-2]
            causal_mask = torch.tril(torch.ones(seq_q, seq_k, device=query.device, dtype=torch.bool))
            norm_qk = norm_qk.masked_fill(~causal_mask, -1e9)
    
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

from cs336_systems.FlashAttention.FlashAttention.flash_attention_torch_vectorized import flash_attn_torch_vectorized_fn
@nvtx.range("Attention-FlashAttention-Torch")
def flash_attention_torch(query, key, value, is_causal: bool = False):
    return flash_attn_torch_vectorized_fn(query, key, value, is_causal=is_causal)

from cs336_systems.FlashAttention.FlashAttention.flash_attention_triton import flash_attn_triton_fn
@nvtx.range("Attention-FlashAttention-MyTriton")
def flash_attention_my_triton(query, key, value, is_causal: bool = False):
    return flash_attn_triton_fn(query, key, value, is_causal)