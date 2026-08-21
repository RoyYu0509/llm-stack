"""
Flash Attention Triton kernel variants with different num_stages configurations.
This module provides separate kernel variants to benchmark the effect of num_stages
on performance. Each variant is compiled with a specific num_stages value.
"""

import triton
import triton.language as tl
import torch
from einops import rearrange


# ============================================================================
# Kernel Implementation (shared across all variants)
# ============================================================================

def _create_flash_fwd_kernel(num_stages_val):
    """Factory function to create a flash forward kernel with specific num_stages."""
    
    # Create autotune config with only the specified num_stages
    autotune_config = [
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, 
                      num_stages=num_stages_val, num_warps=8),
    ]
    
    @triton.autotune(configs=autotune_config, key=['N_QUERIES', 'N_KEYS'])
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr = False,
    ):
        """Flash Attention forward kernel with tiled computation and online softmax."""
        
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        O_block_ptr = tl.make_block_ptr(
            base=O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0)
        )

        L_block_ptr = tl.make_block_ptr(
            base=L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,)
        )

        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0)
        )

        m_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) - 1e20
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        for start_k_B_idx in range(0, N_KEYS, K_TILE_SIZE):
            K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale

            # Boundary mask
            KT_col_mask = start_k_B_idx + tl.arange(0, K_TILE_SIZE)
            Q_row_mask = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            boundary_mask = (KT_col_mask[None, :] < N_KEYS) & (Q_row_mask[:, None] < N_QUERIES)

            # Causal mask
            not_causal = tl.full(boundary_mask.shape, not IS_CAUSAL, dtype=tl.int1)
            causal_mask = (Q_row_mask[:, None] >= KT_col_mask[None, :]) | not_causal
            mask = boundary_mask & causal_mask
            S_ij = tl.where(mask, S_ij, -1e20)

            # Online softmax: update statistics
            curr_max = tl.max(S_ij, axis=1)
            prev_max = m_i
            m_i = tl.where(curr_max > m_i, curr_max, m_i)
            max_correct_scale = tl.exp(prev_max - m_i)

            P_ij = S_ij - m_i[:, None]
            P_ij = tl.exp(P_ij)

            l_i = l_i * max_correct_scale + tl.sum(P_ij, axis=1)

            V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            o_i = o_i * max_correct_scale[:, None] + tl.dot(P_ij, V_j)

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        o_i = o_i / l_i[:, None]
        o_i = o_i.to(tl.float32)
        lse_i = m_i + tl.log(l_i)
        lse_i = lse_i.to(tl.float32)

        tl.store(O_block_ptr, o_i, boundary_check=(0, 1))
        tl.store(L_block_ptr, lse_i, boundary_check=(0,))

    return flash_fwd_kernel


# ============================================================================
# Create kernel variants for num_stages 1-8
# ============================================================================

_kernels = {}
for stages in range(1, 9):
    _kernels[f'flash_fwd_kernel_stages_{stages}'] = _create_flash_fwd_kernel(stages)

# Extract individual kernels
flash_fwd_kernel_stages_1 = _kernels['flash_fwd_kernel_stages_1']
flash_fwd_kernel_stages_2 = _kernels['flash_fwd_kernel_stages_2']
flash_fwd_kernel_stages_3 = _kernels['flash_fwd_kernel_stages_3']
flash_fwd_kernel_stages_4 = _kernels['flash_fwd_kernel_stages_4']
flash_fwd_kernel_stages_5 = _kernels['flash_fwd_kernel_stages_5']
flash_fwd_kernel_stages_6 = _kernels['flash_fwd_kernel_stages_6']
flash_fwd_kernel_stages_7 = _kernels['flash_fwd_kernel_stages_7']
flash_fwd_kernel_stages_8 = _kernels['flash_fwd_kernel_stages_8']


# ============================================================================
# Wrapper functions for each variant
# ============================================================================

def _create_flash_fwd_triton_wrapper(kernel, stages_num):
    """Factory function to create a forward wrapper for a given kernel."""
    
    def flash_fwd_triton(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        """Flash Attention forward pass."""
        assert (Q.shape[-1] == K.shape[-1]) and (Q.shape[-1] == V.shape[-1]), \
            "Token embedding dimension inconsistent"
        assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, \
            f"Input should follow the shape B N D, Actual = {Q.shape}"
        
        DEVICE = Q.device
        B, Q_N, D = Q.shape
        K_N = K.shape[-2]

        OUT = torch.zeros((B, Q_N, D), dtype=Q.dtype, device=DEVICE, requires_grad=False)
        L = torch.zeros((B, Q_N,), dtype=Q.dtype, device=DEVICE, requires_grad=False)

        grid = lambda META: (triton.cdiv(Q_N, META["Q_TILE_SIZE"]), B)
        scale = 1 / D ** 0.5

        kernel[grid](
            Q, K, V,
            OUT, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            OUT.stride(0), OUT.stride(1), OUT.stride(2),
            L.stride(0), L.stride(1),
            Q_N, K_N,
            scale, D=D, IS_CAUSAL=is_causal
        )

        return OUT, L

    return flash_fwd_triton


# Create wrapper functions for all variants
flash_fwd_triton_stages_1 = _create_flash_fwd_triton_wrapper(flash_fwd_kernel_stages_1, 1)
flash_fwd_triton_stages_2 = _create_flash_fwd_triton_wrapper(flash_fwd_kernel_stages_2, 2)
flash_fwd_triton_stages_3 = _create_flash_fwd_triton_wrapper(flash_fwd_kernel_stages_3, 3)
flash_fwd_triton_stages_4 = _create_flash_fwd_triton_wrapper(flash_fwd_kernel_stages_4, 4)
flash_fwd_triton_stages_5 = _create_flash_fwd_triton_wrapper(flash_fwd_kernel_stages_5, 5)
flash_fwd_triton_stages_6 = _create_flash_fwd_triton_wrapper(flash_fwd_kernel_stages_6, 6)
flash_fwd_triton_stages_7 = _create_flash_fwd_triton_wrapper(flash_fwd_kernel_stages_7, 7)
flash_fwd_triton_stages_8 = _create_flash_fwd_triton_wrapper(flash_fwd_kernel_stages_8, 8)


# ============================================================================
# Autograd wrapper for each variant
# ============================================================================

class FlashAttentionTritonFunction(torch.autograd.Function):
    """Base autograd function for Flash Attention."""
    
    @staticmethod
    def create_with_kernel(kernel_fn):
        """Create a Flash Attention autograd function with a specific kernel."""
        
        class _FlashAttentionTriton(torch.autograd.Function):
            @staticmethod
            def forward(ctx, Q, K, V, is_causal=False):
                O, L = kernel_fn(Q, K, V, is_causal)
                ctx.save_for_backward(Q, K, V, O, L)
                ctx.is_causal = is_causal
                return O

            @staticmethod
            def backward(ctx, dLdO):
                Q, K, V, O, L = ctx.saved_tensors
                is_causal = ctx.is_causal
                d = Q.shape[-1]

                # Recompute P on GPU
                KT = rearrange(K, "B K_N D -> B D K_N")
                S = Q @ KT / d ** 0.5
                P = torch.exp(S - L[:, :, None])

                if is_causal:
                    B, Q_N, K_N = S.shape
                    mask = torch.tril(torch.ones(Q_N, K_N, device=Q.device, dtype=torch.bool))
                    P = P * mask[None, :, :]

                # Compute gradients
                PT = rearrange(P, "B Q_N K_N -> B K_N Q_N")
                dLdV = PT @ dLdO

                VT = rearrange(V, "B K_N D -> B D K_N")
                dLdP = dLdO @ VT

                D = torch.sum(dLdP * P, dim=-1, keepdim=True)
                dLdS = P * (dLdP - D)

                dLdQ = dLdS @ K / d ** 0.5

                dLdST = rearrange(dLdS, "B Q_N K_N -> B K_N Q_N")
                dLdK = dLdST @ Q / d ** 0.5

                return dLdQ, dLdK, dLdV, None

        return _FlashAttentionTriton

# Create autograd functions for each variant
_FlashAttention_Stages_1 = FlashAttentionTritonFunction.create_with_kernel(flash_fwd_triton_stages_1)
_FlashAttention_Stages_2 = FlashAttentionTritonFunction.create_with_kernel(flash_fwd_triton_stages_2)
_FlashAttention_Stages_3 = FlashAttentionTritonFunction.create_with_kernel(flash_fwd_triton_stages_3)
_FlashAttention_Stages_4 = FlashAttentionTritonFunction.create_with_kernel(flash_fwd_triton_stages_4)
_FlashAttention_Stages_5 = FlashAttentionTritonFunction.create_with_kernel(flash_fwd_triton_stages_5)
_FlashAttention_Stages_6 = FlashAttentionTritonFunction.create_with_kernel(flash_fwd_triton_stages_6)
_FlashAttention_Stages_7 = FlashAttentionTritonFunction.create_with_kernel(flash_fwd_triton_stages_7)
_FlashAttention_Stages_8 = FlashAttentionTritonFunction.create_with_kernel(flash_fwd_triton_stages_8)

# Application functions
flash_attn_triton_stages_1 = _FlashAttention_Stages_1.apply
flash_attn_triton_stages_2 = _FlashAttention_Stages_2.apply
flash_attn_triton_stages_3 = _FlashAttention_Stages_3.apply
flash_attn_triton_stages_4 = _FlashAttention_Stages_4.apply
flash_attn_triton_stages_5 = _FlashAttention_Stages_5.apply
flash_attn_triton_stages_6 = _FlashAttention_Stages_6.apply
flash_attn_triton_stages_7 = _FlashAttention_Stages_7.apply
flash_attn_triton_stages_8 = _FlashAttention_Stages_8.apply

# Dictionary for easy access
flash_attn_variants = {
    1: flash_attn_triton_stages_1,
    2: flash_attn_triton_stages_2,
    3: flash_attn_triton_stages_3,
    4: flash_attn_triton_stages_4,
    # 5: flash_attn_triton_stages_5,
    # 6: flash_attn_triton_stages_6,
    # 7: flash_attn_triton_stages_7,
    # 8: flash_attn_triton_stages_8,
}
