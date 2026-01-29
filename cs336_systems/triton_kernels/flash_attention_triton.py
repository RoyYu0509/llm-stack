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

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

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

    Q_i: Float[tlTensor, "Q_TILE_SIZE, D"] = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")

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
    m_i:    Float[tlTensor, "Q_TILE_SIZE, "] = tl.zeros((Q_TILE_SIZE,), dtype = tl.float32)
    l_i:    Float[tlTensor, "Q_TILE_SIZE, "] = tl.zeros((Q_TILE_SIZE,), dtype = tl.float32)
    o_i:    Float[tlTensor, "Q_TILE_SIZE, D"] = tl.zeros((Q_TILE_SIZE, D), dtype = tl.float32)
    lse_i:  Float[tlTensor, "Q_TILE_SIZE, "] = tl.zeros((Q_TILE_SIZE,), dtype = tl.float32)

    # Compute online softmax, shifting tile block towards right side
    for key_idx in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Compute pre-softmax
        K_j:  Float[tlTensor, "K_TILE_SIZE, D"] = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        S_ij: Float[tlTensor, "Q_TILE_SIZE, K_TILE_SIZE"] = tl.dot(Q_i, tl.trans(K_j)) * scale
        # Mask the out of bound entries to be -inf, so that row_max & Softmax is correct
        row_mask = key_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        S_ij = tl.where(row_mask < N_KEYS, S_ij, -1e9)

        # Update max (No need Boundary Mask)
        curr_max:  Float[tlTensor, "Q_TILE_SIZE,"] = tl.max(S_ij, axis = 1)
        prev_max:  Float[tlTensor, "Q_TILE_SIZE,"] = m_i
        m_i:  Float[tlTensor, "Q_TILE_SIZE,"] = tl.where(curr_max > m_i, curr_max, m_i)
        max_correct_scale: Float[tlTensor, "Q_TILE_SIZE,"] = tl.exp(prev_max-m_i)

        # Compute the safe softmax (With Boundary Mask)
        P_ij: Float[tlTensor, "Q_TILE_SIZE, K_TILE_SIZE"] = S_ij - m_i[:, None]
        P_ij = tl.exp(P_ij)

        # Update sum
        l_i:  Float[tlTensor, "Q_TILE_SIZE, "] = l_i * max_correct_scale + tl.sum(P_ij, axis=1)

        # Update OUT
        V_j:    Float[tlTensor, "K_TILE_SIZE, D"] = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
        o_i:    Float[tlTensor, "Q_TILE_SIZE, D"] = o_i * max_correct_scale[:,None] + tl.dot(P_ij, V_j)

        # Advance pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
    # Compute the Log Sum Exp
    o_i = o_i / l_i[:, None]
    lse_i = m_i + tl.log(l_i)
    
    # Write to OUT
    tl.store(O_block_ptr, o_i, boundary_check=(0,1))
    tl.store(L_block_ptr, lse_i, boundary_check=(0,))


