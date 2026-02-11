import math

import torch
from jaxtyping import Float
from torch import Tensor


def _causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    q_idx = torch.arange(q_len, device=device)
    k_idx = torch.arange(k_len, device=device)
    return q_idx[:, None] >= k_idx[None, :]


def flash_attention_torch_fwd_vectorized(
    Q: Float[Tensor, "... N_q D"],
    K: Float[Tensor, "... N_k D"],
    V: Float[Tensor, "... N_k D"],
    is_causal: bool = False,
):
    """
    Vectorized (naive) attention forward pass.

    Returns:
        - O: attention output
        - L: logsumexp along key dimension (for backward)
    """
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)

    if is_causal:
        mask = _causal_mask(Q.shape[-2], K.shape[-2], Q.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    L = torch.logsumexp(scores, dim=-1)
    P = torch.exp(scores - L[..., None])
    O = torch.matmul(P, V)
    return O, L


class FlashAttentionTorchFunctionVectorized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        O, L = flash_attention_torch_fwd_vectorized(Q, K, V, is_causal=is_causal)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dLdO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        d = Q.shape[-1]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        if is_causal:
            mask = _causal_mask(Q.shape[-2], K.shape[-2], Q.device)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        P = torch.exp(scores - L[..., None])
        D = torch.sum(O * dLdO, dim=-1, keepdim=True)

        dLdV = torch.matmul(P.transpose(-2, -1), dLdO)
        dLdP = torch.matmul(dLdO, V.transpose(-2, -1))
        dLdS = P * (dLdP - D)
        dLdQ = torch.matmul(dLdS, K) / math.sqrt(d)
        dLdK = torch.matmul(dLdS.transpose(-2, -1), Q) / math.sqrt(d)

        return dLdQ, dLdK, dLdV, None


flash_attn_torch_vectorized_fn = torch.compile(FlashAttentionTorchFunctionVectorized.apply)
