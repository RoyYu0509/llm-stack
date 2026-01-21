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
def scaled_dot_product_attention(query, key, value, bool_mask=None):
    """
    Return an output with the shape (batch_size,..., d_v)

    key:        (... n,d_k) = (batch_size, ..., seq_len, d_k) 
    query:      (... n,d_k) = (batch_size, ..., seq_len, d_k)
    value:      (... n,d_v) = (batch_size, ..., seq_len, d_v)
    bool_mask:  (seq_len, seq_len)

    Return:
        - attention: Float[Tensor, "... seq_q d_v"]
    """
    # Compute Normalized QtK & Apply Mask
    with nvtx.range("Compute QK^t"):
        norm_qk: Float[torch.Tensor, "... seq_q, seq_k"]
        norm_qk = einsum(key, query, "... seq_k d_k, ... seq_q d_k -> ... seq_q seq_k") / torch.sqrt(torch.tensor(key.shape[-1], device=query.device))
        if bool_mask is not None:
            norm_qk = norm_qk.masked_fill(~bool_mask, -1e9) # ~ is to invert F -> T, since F meaning we should masked them
    
    # Softmax
    with nvtx.range("Computing softmax"):
        softmax_qk: Float[torch.Tensor, "... seq_q seq_k"]
        softmax_qk = softmax(norm_qk, axis=-1)

    # Attention
    with nvtx.range("Multiplied V"):
        attention: Float[Tensor, "... seq_q d_v"]
        attention = einsum(softmax_qk, value, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")

    return attention
