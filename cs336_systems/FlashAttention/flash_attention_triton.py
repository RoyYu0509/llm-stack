import triton
import triton.language as tl
import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import jaxtyping
from jaxtyping import Float
from torch import Tensor
from triton.language import tensor as tlTensor
from einops import rearrange

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    SDPBackend = None
    sdpa_kernel = None

# Check for hardward compatibility for TF32 tensor-core on Triton
try:
    # This is a bit hacky, but Triton doesn't provide a direct API to check for TF32 support
    # We can attempt to compile a kernel with allow_tf32=True and see if it succeeds
    @triton.jit
    def tf32_check_kernel():
        x = tl.zeros((1,), dtype=tl.float16)
        y = tl.zeros((1,), dtype=tl.float16)
        z = tl.dot(x, y, out_dtype=tl.float32)
    tf32_check_kernel[(1,)]()  # Try to launch the kernel
except Exception:
    raise RuntimeError("This GPU does not support TF32, which is required for our Flash Attention implementation. Please use an Ampere or newer GPU.")


autotune_configs = [

    # =========================
    # 16x16 (latency focused)
    # =========================
    triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_stages=1, num_warps=1),
    # triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_stages=2, num_warps=1),
    # triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_stages=1, num_warps=2),
    # triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_stages=2, num_warps=2),
    # triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_stages=2, num_warps=4),

    # # =========================
    # # 32x32 (balanced)
    # # =========================
    # triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_stages=1, num_warps=2),
    # triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_stages=2, num_warps=2),
    # triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_stages=2, num_warps=4),
    # triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_stages=3, num_warps=4),

    # # =========================
    # # 64x32 (good for head_dim=256)
    # # =========================
    # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_stages=1, num_warps=4),
    # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_stages=2, num_warps=4),
    # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_stages=2, num_warps=8),

    # # =========================
    # # 32x64
    # # =========================
    # triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}, num_stages=1, num_warps=4),
    # triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}, num_stages=2, num_warps=4),
    # triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}, num_stages=2, num_warps=8),

    # # =========================
    # # 64x64 (throughput focused)
    # # =========================
    # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_stages=1, num_warps=4),
    # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_stages=2, num_warps=4),
    # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_stages=2, num_warps=8),
    # triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_stages=3, num_warps=8),

    # # =========================
    # # 128x64 (aggressive, may fail)
    # # =========================
    # triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_stages=1, num_warps=8),
    # triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_stages=2, num_warps=8),

    # # =========================
    # # 128x128 (very aggressive)
    # # =========================
    # triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_stages=1, num_warps=8),
]

# We need to include N_Q, N_K in the key since they affect control flow (masking)
# RULE: If changing the parameter affects the tile sizes, include them in the key as well
@triton.autotune(configs=autotune_configs, key=['N_QUERIES', 'N_KEYS', 'D'])
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
    D: tl.constexpr,  # This Feature Dim will be fixed at compile time; (Note: We are not tiling over D)
    Q_TILE_SIZE: tl.constexpr,  # Bq
    K_TILE_SIZE: tl.constexpr,  # Bk
    OUTPUT_DTYPE: tl.constexpr,  # Output dtype for compile-time specialization
    IS_CAUSAL: tl.constexpr = False,
):  
    """
    Flash Attention Forward Pass Kernel, with mannual autocasting to FP32 for stable softmax computation.
    
    Parallelize over `Batch` and `Q_ROW`.
    """
    # ------------------------------------------------------------
    # Program IDs (Lunch grid: each Program owns a Q_TILE & O_TILE shape = [Q_TILE_SIZE, D]
    # number of tiles = (Q_N/Q_TILE_SIZE) * (batch_num)
    # ------------------------------------------------------------
    Q_i_TILE_idx = tl.program_id(0)
    batch_index = tl.program_id(1)

    # ------------------------------------------------------------
    # Current Batch = batch_index
    # Current Q_i_TILE = [Q_i_TILE_idx * Q_TILE_SIZE : (Q_i_TILE_idx+1)*Q_TILE_SIZE]
    # ------------------------------------------------------------
    Q_block_ptr = tl.make_block_ptr(
        # Matrix info
        base=Q_ptr + batch_index * stride_qb, # Each Q across batch
        shape=(N_QUERIES, D),   
        strides=(stride_qq, stride_qd),  
        # The Tiles
        offsets=(Q_i_TILE_idx * Q_TILE_SIZE, 0), # Get the Q_i TILE starting offset
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # Load in the Q_i TILE 
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")  # "Q_TILE_SIZE, D"

    O_block_ptr = tl.make_block_ptr(
        # Matrix info
        base=O_ptr + batch_index * stride_ob, 
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od,),
        # TILE info
        offsets=(Q_i_TILE_idx * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    L_block_ptr = tl.make_block_ptr(
        # Matrix info
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(Q_i_TILE_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    
    # ------------------------------------------------------------
    # KV block pointer: [Bk, D] later transpose
    # ------------------------------------------------------------
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
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )

    # ------------------------------------------------------------
    # Record Casting Dtype for SoftMax & Output Dtype
    # ------------------------------------------------------------
    # High precision dtype for sensitive accumulation ops (compile-time constant)
    hp_dtype = tl.float32
    # Output dtype passed as constexpr for compile-time specialization
    o_dtype = OUTPUT_DTYPE

    # ------------------------------------------------------------
    # Online Statistics
    # ------------------------------------------------------------
    # Move Q_i to high precision once since it will be reused across all K_TILE blocks
    Q_i_hp = Q_i.to(hp_dtype)

    # Online Staitics (Initialized in High Precision)
    m_i = tl.zeros((Q_TILE_SIZE,), dtype = hp_dtype) - 1e20    # "Q_TILE_SIZE, "
    l_i = tl.zeros((Q_TILE_SIZE,), dtype = hp_dtype)          # "Q_TILE_SIZE, "
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype = hp_dtype)        # "Q_TILE_SIZE, D"
    lse_i = tl.zeros((Q_TILE_SIZE,), dtype = hp_dtype)        # "Q_TILE_SIZE, "

    # Compute online softmax, shifting tile block towards right side
    for start_k_B_idx in range(0, N_KEYS, K_TILE_SIZE):
        # 1. Compute pre-softmax
        K_j = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")  # "K_TILE_SIZE, D"
        K_j_hp = K_j.to(hp_dtype) # Upcast K to high precision
        S_ij = tl.dot(Q_i_hp, tl.trans(K_j_hp), out_dtype=hp_dtype) * scale  # "Q_TILE_SIZE, K_TILE_SIZE"
       
        # 2.  Masking
        # 2.1 Mask the out of bound entries to be -inf, so that row_max & Softmax is correct
        KT_col_mask = start_k_B_idx + tl.arange(0, K_TILE_SIZE)             # ", K_TILE_SIZE" - Along the KT cols
        Q_row_mask = Q_i_TILE_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)     # "Q_TILE_SIZE, " - Along the Q rows
        boundary_mask = (KT_col_mask[None, :] < N_KEYS) & (Q_row_mask[:, None] < N_QUERIES)  # "Q_TILE_SIZE, K_TILE_SIZE"

        # 2.2 Causal Mask (Avoiding If Else logics branch)
        not_causal = tl.full(boundary_mask.shape, not IS_CAUSAL, dtype=tl.int1) # Initialize, set the dtype as bool int
        causal_mask = (Q_row_mask[:, None] >= KT_col_mask[None, :]) | not_causal  # True when IS_CAUSAL=False, else row>=col
        mask = boundary_mask & causal_mask
        S_ij = tl.where(mask, S_ij, -1e20)  # Use large negative value for masking

        # 2. Update max (No need Boundary Mask)
        curr_max = tl.max(S_ij, axis = 1)               # "Q_TILE_SIZE, "
        prev_max = m_i  # "Q_TILE_SIZE"
        m_i = tl.where(curr_max > m_i, curr_max, m_i)   # "Q_TILE_SIZE, "
        max_correct_scale = tl.exp(prev_max-m_i)        # "Q_TILE_SIZE, "

        # 3. Compute the safe softmax (With Boundary Mask)
        P_ij = S_ij - m_i[:, None]  # "Q_TILE_SIZE, K_TILE_SIZE"
        P_ij = tl.exp(P_ij)

        # 4. Update sum (accumulation in hp_dtype)
        l_i = l_i * max_correct_scale + tl.sum(P_ij, axis=1)  # "Q_TILE_SIZE, "

        # 5. Update OUT with ValueMatrix
        V_j = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")  # "K_TILE_SIZE, D"
        V_j_hp = V_j.to(hp_dtype) 
        o_i = o_i * max_correct_scale[:,None] + tl.dot(P_ij, V_j_hp, out_dtype=hp_dtype)  # "Q_TILE_SIZE, D"

        # Advance pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
    # End for: Compute the Log Sum Exp
    # Final normalization in hp_dtype
    o_i = o_i / l_i[:, None]
    lse_i = m_i + tl.log(l_i)
    
    # Downcast to original output dtype before storing (memory bandwidth optimization)
    o_i = o_i.to(o_dtype)
    lse_i = lse_i.to(o_dtype)

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

    # Validate that the feature dimension used as a block_shape element
    # is a power of two. Triton's tl.make_block_ptr requires each
    # block_shape element to be a power of two; otherwise compilation
    # can fail with obscure errors. Raise a clear error so users can
    # either pick a power-of-two D or pad their tensors.
    if D & (D - 1) != 0:
        raise ValueError(
            f"Shape = (Head_dim, Q_N, K_N) = ({D}, {Q_N}, {K_N}) all dims should be pow of 2 and >= 16,\n"
        )

    
    # Create output buffers
    OUT = torch.zeros((B, Q_N, D), dtype=Q.dtype, device=DEVICE)
    L = torch.zeros((B, Q_N, ),  dtype=Q.dtype, device=DEVICE)

    grid = lambda META: (triton.cdiv(Q_N, META["Q_TILE_SIZE"]),  B)

    scale = 1/D**0.5
    
    # Map PyTorch dtype to Triton dtype for constexpr
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    output_dtype = dtype_map[Q.dtype]

    try:
        flash_fwd_kernel[grid](
        Q, K, V,
        OUT, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        OUT.stride(0), OUT.stride(1), OUT.stride(2),
        L.stride(0), L.stride(1),
        Q_N, K_N,
        scale, D=D, OUTPUT_DTYPE=output_dtype, IS_CAUSAL=is_causal
        )
    except Exception as e:
        # Surface a clearer, contextualized error to help debugging Triton
        raise RuntimeError(
            f"Triton flash_fwd_kernel failed. Shapes: Q={tuple(Q.shape)}, K={tuple(K.shape)}, V={tuple(V.shape)}, "
            f"dtypes: Q={Q.dtype}, K={K.dtype}, V={V.dtype}. Original error: {e}"
        ) from e

    return OUT, L

@torch.compile  # or torch.compile(mode="max-autotune") depending on your setup
def attn_bwd_torch(Q, K, V, O, L, dLdO, is_causal: bool):
    D = torch.sum(O * dLdO, dim=-1, keepdim=True)
    d = Q.shape[-1]

    KT = rearrange(K, "B N D -> B D N")
    S  = Q @ KT / (d ** 0.5)
    P  = torch.exp(S - L[:, :, None])

    if is_causal:
        B, Q_N, K_N = S.shape
        mask = torch.tril(torch.ones(Q_N, K_N, device=Q.device, dtype=torch.bool))
        P = P * mask[None, :, :]

    PT   = rearrange(P, "B Q_N K_N -> B K_N Q_N")
    dLdV = PT @ dLdO

    VT   = rearrange(V, "B K_N D -> B D K_N")
    dLdP = dLdO @ VT

    dLdS = P * (dLdP - D)
    dLdQ = dLdS @ K / (d ** 0.5)

    dLdST = rearrange(dLdS, "B Q_N K_N -> B K_N Q_N")
    dLdK  = dLdST @ Q / (d ** 0.5)
    return dLdQ, dLdK, dLdV



class FlashAttentionTorchFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        """
        Forward pass for FlashAttention using PyTorch operations.

        Parameters:
            - Q: Float[torch.Tensor, "N_q, d"] The Query matrix
            - K: Float[torch.Tensor, "N_k, d"] The Key matrix
            - V: Float[torch.Tensor, "N_k, d"] The Value matrix
        """
        O, L = flash_fwd_triton(Q, K, V,is_causal)
        ctx.save_for_backward(Q,K,V,O,L)
        ctx.is_causal = is_causal
        # print("Forward FlashAttention Torch done.")
        # print("O:", O.shape)
        # print("L:", L.shape)
        return O

    @staticmethod
    def backward(ctx, dLdO):
        Q, K, V, O, L = ctx.saved_tensors
        dLdQ, dLdK, dLdV = attn_bwd_torch(Q, K, V, O, L, dLdO, ctx.is_causal)
        return dLdQ, dLdK, dLdV, None
    
flash_attn_triton_fn = FlashAttentionTorchFunctionTriton.apply
