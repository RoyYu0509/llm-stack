import triton
import triton.language as tl
import triton
import triton.language as tl
import torch
import jaxtyping
from jaxtyping import Float
from torch import Tensor
from triton.language import tensor as tlTensor
from einops import rearrange


# Set of configurations to try during autotuning (Hyperparameter search space)
autotune_configs = [
    # Small tiles
    triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32, }, num_stages=4, num_warps=8),
    triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64, }, num_stages=4, num_warps=8),
    triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32, }, num_stages=4, num_warps=8),
]


@triton.autotune(configs=autotune_configs, key=['N_QUERIES', 'N_KEYS']) # Everytime M, N, K changes, we re-autotune
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
    Q_TILE_SIZE: tl.constexpr,  # Bq
    K_TILE_SIZE: tl.constexpr,  # Bk
    IS_CAUSAL: tl.constexpr = False,
):  
    """
    Parallelize over `Batch` and `Q_ROW`.


    """
    # ------------------------------------------------------------
    # Program IDs (Lunch grid: each Program owns a Q_TILE & O_TILE shape = [Q_TILE_SIZE, D]
    # number of tiles = (Q_N/Q_TILE_SIZE) * (batch_num)
    # ------------------------------------------------------------
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # ------------------------------------------------------------
    # Current Batch = batch_index
    # ------------------------------------------------------------
    Q_block_ptr = tl.make_block_ptr(
        # Matrix info
        base=Q_ptr + batch_index * stride_qb, # Each Q across batch
        shape=(N_QUERIES, D),   
        strides=(stride_qq, stride_qd),  
        # The Tiles
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # 
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        # Matrix info
        base=O_ptr + batch_index * stride_ob, 
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od,),
        # TILE info
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    L_block_ptr = tl.make_block_ptr(
        # Matrix info
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")  # "Q_TILE_SIZE, D"

    # ------------------------------------------------------------
    # KV block pointer: [Bk, D] later transpose
    # ------------------------------------------------------------
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,  # 
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
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )
        
    # ------------------------------------------------------------
    # Online Statistics
    # ------------------------------------------------------------
    # Initialize the row max tensor to -inf, because the Attention score could be negative. () 
    m_i = tl.zeros((Q_TILE_SIZE,), dtype = tl.float32) - 1e20    # "Q_TILE_SIZE, "
    l_i = tl.zeros((Q_TILE_SIZE,), dtype = tl.float32)          # "Q_TILE_SIZE, "
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype = tl.float32)        # "Q_TILE_SIZE, D"
    lse_i = tl.zeros((Q_TILE_SIZE,), dtype = tl.float32)        # "Q_TILE_SIZE, "

    # Compute online softmax, shifting tile block towards right side
    for start_k_B_idx in range(0, N_KEYS, K_TILE_SIZE):
        # 1. Compute pre-softmax
        K_j = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")  # "K_TILE_SIZE, D"
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale                                # "Q_TILE_SIZE, K_TILE_SIZE"
       
        # 2.  Masking
        # 2.1 Mask the out of bound entries to be -inf, so that row_max & Softmax is correct
        KT_col_mask = start_k_B_idx + tl.arange(0, K_TILE_SIZE)             # ", K_TILE_SIZE" - Along the KT cols
        Q_row_mask = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)     # "Q_TILE_SIZE, " - Along the Q rows
        boundary_mask = (KT_col_mask[None, :] < N_KEYS) & (Q_row_mask[:, None] < N_QUERIES)  # "Q_TILE_SIZE, K_TILE_SIZE"

        # 2.2 Causal Mask (Avoiding If Else logics branch)
        not_causal = tl.full(boundary_mask.shape, not IS_CAUSAL, dtype=tl.int1) # Initialize, set the dtype as bool int
        causal_mask = (Q_row_mask[:, None] >= KT_col_mask[None, :]) | not_causal  # True when IS_CAUSAL=False, else row>=col
        mask = boundary_mask & causal_mask
        S_ij = tl.where(mask, S_ij, -1e20)

        # 2. Update max (No need Boundary Mask)
        curr_max = tl.max(S_ij, axis = 1)               # "Q_TILE_SIZE, "
        prev_max = m_i  # "Q_TILE_SIZE"
        m_i = tl.where(curr_max > m_i, curr_max, m_i)   # "Q_TILE_SIZE, "
        max_correct_scale = tl.exp(prev_max-m_i)        # "Q_TILE_SIZE, "

        # 3. Compute the safe softmax (With Boundary Mask)
        P_ij = S_ij - m_i[:, None]  # "Q_TILE_SIZE, K_TILE_SIZE"
        P_ij = tl.exp(P_ij)

        # 4. Update sum
        l_i = l_i * max_correct_scale + tl.sum(P_ij, axis=1)  # "Q_TILE_SIZE, "

        # 5. Update OUT
        V_j = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")  # "K_TILE_SIZE, D"
        o_i = o_i * max_correct_scale[:,None] + tl.dot(P_ij, V_j)  # "Q_TILE_SIZE, D"

        # Advance pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
    # End for: Compute the Log Sum Exp
    o_i = o_i / l_i[:, None]
    o_i = o_i.to(tl.float32)
    lse_i = m_i + tl.log(l_i)
    lse_i = lse_i.to(tl.float32)

    # Write to OUT
    tl.store(O_block_ptr, o_i, boundary_check=(0,1))
    tl.store(L_block_ptr, lse_i, boundary_check=(0,))


def flash_fwd_triton(
    Q: torch.Tensor, 
    K: torch.Tensor, V: torch.Tensor, 
    is_causal=False
):
    assert (Q.shape[-1] == K.shape[-1]) and (Q.shape[-1] == V.shape[-1]), "Token embedding dimension inconsistent"
    assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, f"Input should follow the shape  B N D, Acutal = {Q.shape}"
    DEVICE = Q.device

    # Get shape
    B, Q_N, D = Q.shape
    K_N = K.shape[-2]

    
    # Create output buffers
    OUT = torch.zeros((B, Q_N, D), dtype=Q.dtype, device=DEVICE, requires_grad=True)
    L = torch.zeros((B, Q_N, ),  dtype=Q.dtype, device=DEVICE, requires_grad=True)

    grid = lambda META: (triton.cdiv(Q_N, META["Q_TILE_SIZE"]),  B)

    scale = 1/D**0.5

    flash_fwd_kernel[grid](
        Q, K, V,
        OUT, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        OUT.stride(0), OUT.stride(1), OUT.stride(2),
        L.stride(0), L.stride(1),
        Q_N, K_N,
        scale, D = D, IS_CAUSAL = is_causal
    )

    return OUT, L





class FlashAttentionTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass for FlashAttention using PyTorch operations.

        Parameters:
            - Q: Float[torch.Tensor, "N_q, d"] The Query matrix
            - K: Float[torch.Tensor, "N_k, d"] The Key matrix
            - V: Float[torch.Tensor, "N_k, d"] The Value matrix
            - B_q: int Query TILE_ROW
            - B_k: int Key TILE_ROW
        """
        B_q, B_k= 16, 16  # Example tile sizes; these can be tuned based on hardware
        O, L = flash_fwd_triton(Q, K, V,is_causal)
        ctx.save_for_backward(Q,K,V,O,L)
        ctx.B_q = B_q
        ctx.B_k = B_k
        print("Forward FlashAttention Torch done.")
        print("O:", O.shape)
        print("L:", L.shape)
        return O

    @staticmethod
    def backward(ctx, grad_O):
        """
        Backward pass for FlashAttention using PyTorch operations.

        Parameters:
            - grad_O: Gradient of the output from the forward pass.
        """
        Q, K, V, O, L = ctx.saved_tensors
        B_q = ctx.B_q
        B_k = ctx.B_k

        # Compute gradients w.r.t Q, K, V using similar tiled approach
        # This is a placeholder; actual implementation would mirror the forward pass logic.
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        # Implement gradient computations here...

        return grad_Q, grad_K, grad_V, None, None