"""
- backward pass
- connect to pytorchgraph
- re-use intermediate values from fwd & bwd pass
- locks & atomics operations
- two sequential kernels (Better Than) one fused kernel
"""

import triton
import triton.language as tl
import torch
import jaxtyping
from jaxtyping import Float
from torch import Tensor
from einops import rearrange
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def _layernorm_forward(
    X_ptr, OUT_ptr, w_ptr, b_ptr, m_ptr, rstd_ptr, eps,
    X_N, X_D,
    stride_N, 
    # Customized meta_para
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Each TILEs.shape = [1, BLOCK_D], which is a portion of the entire row.

    Kernel is parallelized on rows: X_N
        TILEs should moves across columns.

        The block pointer must know:
        - ptr: ptr          The pointer to the first element of the tensor
        - shape:Tuple()     The overall shape of the original tensor (N, D) to handle out-of-bounds access
        - strides:Tuple()   The strides of each dimension (stride_R, stride_D) to use the memory layout properly
        - offsets:Tuple()   The ND coordinates of the starting block, i.e., "offsets"
        - block_shape:Tuple() The block shape to use load/store at a time
        - order:Tuple()     The order of the dimensions in memory from major to minor
            axes (= np.argsort(strides)) for optimizations

    """
    # Get which BLOCK is this program processing
    row_idx = tl.program_id(0)

    X_B_ptr = tl.make_block_ptr(
        X_ptr, shape=(X_N, X_D),
        strides=(1,), 
        offsets=(row_idx,0),
        block_shape=(1, BLOCK_SIZE_D),
        order=(1,0),
    )

    OUT_B_ptr = tl.make_block_ptr(
        OUT_ptr, shape=(X_N, X_D),
        strides=(1,),
        offsets=(row_idx,0),
        block_shape=(1, BLOCK_SIZE_D),
        order=(1,0),
    )

    w_B_ptr = tl.make_block_ptr(
        w_ptr, shape=(X_D,),
        strides=(1,),
        offsets=(row_idx,),
        block_shape=(BLOCK_SIZE_D,),
        order=(0,),
    )

    b_B_ptr = tl.make_block_ptr(
        b_ptr, shape=(X_D,),
        strides=(1,),
        offsets=(row_idx,),
        block_shape=(BLOCK_SIZE_D,),
        order=(0,),
    )

    m_B_ptr = tl.make_block_ptr(
        m_ptr, shape=(X_N,),
        strides=(1,),
        offsets=(row_idx,),
        block_shape=(1,),
        order=(0,),
    )

    rstd_B_ptr = tl.make_block_ptr(
        rstd_ptr, shape=(X_N,),
        strides=(1,),
        offsets=(row_idx,),
        block_shape=(1,),
        order=(0,),
    )
    
    # Compute the current row's mean
    row_tile_sum:   Float[Tensor,"1, BLOCK_SIZE_D"] = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32) # Collect x_i,1~X_D values
    for ith_block in range(cdiv(X_D, BLOCK_SIZE_D)):
        # Load TILE
        X_TILE: Float[Tensor, "1, BLOCK_SIZE_D"] = tl.load(X_B_ptr, boundary_check=(0,1), padding_option="zero")
        
        # Compute the TILE sum
        row_tile_sum + X_TILE

        # Advance the pointer
        X_B_ptr.advance((0, BLOCK_SIZE_D))

    row_sum:        Float[Tensor,"1"] = tl.sum(row_tile_sum, axis=0)
    row_mean:       Float[Tensor,"1"] = row_sum / X_D

    # Compute the current row's variance
    row_tile_diff_sqr:   Float[Tensor,"1, BLOCK_SIZE_D"] = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32) # Collect (x_i,1~X_D - mean)**2 value
    for ith_block in range(cdiv(X_D, BLOCK_SIZE_D)):
        # Load TILE
        X_TILE: Float[Tensor, "1, BLOCK_SIZE_D"] = tl.load(X_B_ptr, boundary_check=(0,1), padding_option="zero")
        
        # Compute the TILE variance
        ith_block_cols = (ith_block * BLOCK_SIZE_D) + tl.arange(0, BLOCK_SIZE_D)
        diff:   Float[Tensor, "1, BLOCK_SIZE_D"] = tl.where(ith_block_cols < X_D, X_TILE-row_mean, 0.0)
        row_tile_diff_sqr += diff * diff

        # Advance the pointer
        X_B_ptr.advance((0, BLOCK_SIZE_D))

    row_var = tl.sum(row_tile_diff_sqr, axis=0) / X_D
    row_rstd = 1/tl.sqrt(row_var + eps)

    # Write to mean ptr
    tl.store(m_B_ptr, row_mean, boundary_check=(0,))
    tl.store(rstd_B_ptr, row_rstd, boundary_check=(0,))

    # Apply transformation
    for ith_block in range(cdiv(X_D, BLOCK_SIZE_D)):
        OUT_TILE:   Float[Tensor,"1, BLOCK_SIZE_D"] = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32) # Collect y_i,1~X_D values
        X_TILE:     Float[Tensor, "1, BLOCK_SIZE_D"] = tl.load(X_B_ptr, boundary_check=(0,1), padding_option="zero")
        w_TILE:     Float[Tensor, "1, BLOCK_SIZE_D"] = tl.load(w_B_ptr, boundary_check=(0,1), padding_option="zero")
        b_TILE:     Float[Tensor, "1, BLOCK_SIZE_D"] = tl.load(b_B_ptr, boundary_check=(0,1), padding_option="zero")

        # Normalize it
        NORM_X_TILE: Float[Tensor, "1, BLOCK_SIZE_D"] = (X_TILE - row_mean) * row_rstd
        # Apply transformation
        OUT_TILE:    Float[Tensor, "1, BLOCK_SIZE_D"] = NORM_X_TILE * w_TILE + b_TILE
        # Write the result
        tl.store(OUT_B_ptr, OUT_TILE, boundary_check=(0,1))

        # Advance the pointers
        OUT_B_ptr.advance((0,BLOCK_SIZE_D))
        X_B_ptr.advance((0,BLOCK_SIZE_D))
        w_B_ptr.advance((BLOCK_SIZE_D, ))
        b_B_ptr.advance((BLOCK_SIZE_D, )) 

@triton.jit
def _layernorm_backward_dLdx(
    dLdy,
    X, dLdX, X_N, X_D, X_stride,
    w, grad_w_intermediate, 
    b, grad_b_intermediate,
    mean, rstd, 
    grad_w_locks, grad_b_locks,
    GROUP_SIZE: tl.constexpr, BLOCK_SIZE_D: tl.constexpr
):
    """
    Parallelize over rows X_N
    BLOCK.shape = [1, BLOCK_SIZE_D]
    """
    PID = tl.program_id(0)
    dLdy_B_ptr = tl.make_block_ptr(
        dLdy, shape=(X_N,),
        strides=(X_stride, ),
        offsets=(PID*X_N, ), block_shape=(1,),
        order=(0,)
    )

    X_B_ptr = tl.make_block_ptr(
        X, shape=(X_N, X_D),
        strides=(X_stride, 1),
        offsets=(PID*X_N, 0), block_shape=(1, BLOCK_SIZE_D),
        order=(1,0)
    )

    dLdX_B_ptr = tl.make_block_ptr(
        dLdX, shape=(X_N, X_D),
        strides=(X_stride, 1),
        offsets=(PID*X_N, 0), block_shape=(1,BLOCK_SIZE_D),
        order=(1,0)
    )

    w_B_ptr = tl.make_block_ptr(
        w, shape=(X_D,), strides=(1,),
        offsets=(PID*BLOCK_SIZE_D,), block_shape=(BLOCK_SIZE_D,),
        order=(0,)
    )

    grad_w_interm_B_ptr = tl.make_block_ptr(
        grad_w_intermediate, 
        shape=(X_D,), strides=(1,), 
        offsets=(PID*BLOCK_SIZE_D,), block_shape=(BLOCK_SIZE_D,),
        order=(0,),
    )

    b_B_ptr = tl.make_block_ptr(
        b,
        shape=(X_D,), strides=(1,), 
        offsets=(PID*BLOCK_SIZE_D), block_shape=(BLOCK_SIZE_D,),
        order=(0,)
    )

    grad_b_interm_B_ptr = tl.make_block_ptr(
        grad_b_intermediate,
        shape=(X_D,), strides=(1,), 
        offsets=(PID*BLOCK_SIZE_D), block_shape=(BLOCK_SIZE_D,),
        order=(0,)
    )

    mean_B_ptr = tl.make_block_ptr(
        mean, shape=(X_N,), strides=(1,),
        offsets=(PID,), block_shape=(1,),
        order=(0,)
    )


    rstd_B_ptr = tl.make_block_ptr(
        mean, shape=(X_N,), strides=(1,),
        offsets=(PID,), block_shape=(1,),
        order=(0,)
    )

    gradw_lock_ptr = tl.make_block_ptr(
        grad_w_locks, shape=(X_D,), strides=(1,),
        offsets=(PID*BLOCK_SIZE_D,), block_shape=(BLOCK_SIZE_D,),
        order=(0,)
    )

    gradb_lock_ptr = tl.make_block_ptr(
        grad_b_locks, shape=(X_D,), strides=(1,),
        offsets=(PID*BLOCK_SIZE_D,), block_shape=(BLOCK_SIZE_D,),
        order=(0,)
    )


    # Compute dL/dX
    OUT_dLdX = tl.zeros((BLOCK_SIZE_D), dtype=tl.float32)
    for i_th_tile in range(tl.cdiv(X_D, BLOCK_SIZE_D)):
        X_TILE:     Float[Tensor, "1 BLOCK_SIZE_D"] = tl.load(X_B_ptr, boundary_check=(0,1), padding_option="zero")
        w_TILE:     Float[Tensor, "BLOCK_SIZE_D, "] = tl.load(w_B_ptr, boundary_check=(0,), padding_option="zero")
        dLdy_TILE:  Float[Tensor, "BLOCK_SIZE_D, "] = tl.load(dLdy_B_ptr, boundary_check=(0,), padding_option="zero")
        mean_TILE:  Float[Tensor, "1, "] = tl.load(mean_B_ptr)
        rstd_TILE:  Float[Tensor, "1, "] = tl.load(rstd_B_ptr)

        # Mask outof cols entries
        entries_cols = i_th_tile * BLOCK_SIZE_D + tl.arange(BLOCK_SIZE_D)
        
        # Compute dL/dX in bound
        norm_x:     Float[Tensor, "1 BLOCK_SIZE_D"] = tl.where(entries_cols < X_D, X_TILE-mean_TILE, 0.0) # Compute the normalized x
        dydnorm_x:  Float[Tensor, "BLOCK_SIZE_D, "] = tl.where(entries_cols < X_D, w_TILE*dLdy_TILE, 0.0) # Compute the grad w.r.t. normalized x
        row_sum_dydnorm_x:  Float[Tensor, "1, "] = tl.sum(dydnorm_x, axis=0)
        OUT_dLdX:   Float[Tensor, "1 BLOCK_SIZE_D"] = rstd_TILE * (dydnorm_x - row_sum_dydnorm_x/X_D - row_sum_dydnorm_x*norm_x/X_D)

        # Store the TILE result
        tl.store(dLdX_B_ptr, OUT_dLdX, boundary_check=(1,0))

        # Advance all pointers
        X_B_ptr.advance()
        w_B_ptr.advance()
        dLdy_B_ptr.advance()
        mean_B_ptr.advance()
        rstd_B_ptr.advance()
        
        dLdX_B_ptr.advance()

    # Compute intermediate dL/dw
    for i_th_tile in range(tl.cdiv(X_D, BLOCK_SIZE_D)):
        X_TILE:     Float[Tensor, "1 BLOCK_SIZE_D"] = tl.load(X_B_ptr, boundary_check=(0,1), padding_option="zero")
        dLdy_TILE:  Float[Tensor, "BLOCK_SIZE_D, "] = tl.load(dLdy_B_ptr, boundary_check=(0,), padding_option="zero")
        mean_TILE:  Float[Tensor, "1, "] = tl.load(mean_B_ptr)
        rstd_TILE:  Float[Tensor, "1, "] = tl.load(rstd_B_ptr)

        # Mask outof cols entries & Compute the normalized X
        entries_cols = i_th_tile * BLOCK_SIZE_D + tl.arange(BLOCK_SIZE_D)
        norm_x:     Float[Tensor, "1 BLOCK_SIZE_D"] = tl.where(entries_cols < X_D, X_TILE-mean_TILE, 0.0)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_X, normalized_shape, w, b, eps):
        # Batch all leading dimensions, reshape to a 2D tensor.
        X: Float[Tensor, "N, D"] = rearrange(in_X, "... N D -> (... N) D")
        N, D = X.shape
        OUT = torch.empty_like(X)
        mean = torch.empty(D, dtype=torch.float32, device=DEVICE)
        rstd = torch.empty(D, dtype=torch.float32, device=DEVICE)

        BLOCK_SIZE_D = triton.next_power_of_2(D) # Each block shape = [1, BLOCK_SIZE]
        # Some random num_wrap config
        num_warps = min(max(BLOCK_SIZE_D//256, 1), 8)

        # Define grid: Parrallelized over rows
        # Each block is shape = [1, BLOCK_SIZE_D]
        _layernorm_forward[(N,)](
            X, OUT, w, b, mean, rstd, eps,
            N, D,
            X.stride(0),
            # self-defined meta-parameters
            BLOCK_SIZE_D = BLOCK_SIZE_D, 
            # triton official meta-parameters
            num_warps=num_warps
        )

        # Here, ctx is to cache intermediate value for backward pass
        ctx.save_for_backward(X, w, b, rstd)
        ctx.BLOCK_SIZE_D = BLOCK_SIZE_D
        ctx.num_warps = num_warps
        ctx.eps = eps

        return OUT



    @staticmethod
    def backward(ctx, dLdy):
        X, w, b, mean, rstd, = ctx.saved_tensors
        N, D = X.shape

        dLdX: Float[Tensor, "N, D"] = torch.empty_like(X)
        dLdw: Float[Tensor, "D,"] = torch.empty_like(X)
        dldb: Float[Tensor, "D,"] = torch.empty_like(X)

        # Define the intermediate reduce step size
        GROUP_SIZE = 64 # Reduce 64 block into 1 output
        grad_w_intermediate = torch.zeros((GROUP_SIZE, N), dtype=X.dtype, device=DEVICE)
        grad_b_intermediate = torch.zeros((GROUP_SIZE, N), dtype=X.dtype, device=DEVICE)
        # Create a R/W lock for intermediate gradient tensor
        grad_w_locks = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=DEVICE)
        grad_b_locks = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=DEVICE)
        # Compute the gradient backward
        _layernorm_backward_dLdx[(N,)](
            dLdy,
            X, dLdX, N, D, X.stride(0),
            w, grad_w_intermediate,
            b, grad_b_intermediate,
            mean, rstd, 
            grad_w_locks, grad_w_locks,
            GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE_D=ctx.BLOCK_SIZE_D, num_warps=ctx.num_warps,
        )
        grid = lambda META: (triton.cdiv(D, META['BLOCK_SIZE_D']),)
        _w_b_reduce_kernel[grid](
            dLdw, grad_w_intermediate, 
            dldb, grad_b_intermediate, 
            min(GROUP_SIZE, N), D,
            BLOCK_SIZE_N=32, BLOCK_SIZE_D=64
        )

        retrun dLdX, None, dLdw, dldb, None



# Testing against torch implementation
def test_layernorm_kernel(
    X_Shape:tuple, # 2D input
    dtype,
    eps=1e-5,
    device=DEVICE      
):  
    N, D = X_Shape
    X = -2.3 + 0.5 * torch.randn(X_Shape, dtype=dtype, device=DEVICE)
    X.requires_grad_(True)
    w = torch.rand((N,), dtype=dtype, device=DEVICE)
    b = torch.rand((N,), dtype=dtype, device=DEVICE)

    print("Testing Forward Prop")
    y_triton = layer_norm_triton(X, (N, ), w, b, eps)
    y_torch = torch.nn.functional.layer_norm(X, (N,), w, b, eps).to(dtype)
    torch.testing.assert_close(y_triton, y_torch)
    print("Forward pass test PASS!")
    

    print("Testing Back Prop")
    dLdy = 0.1 * torch.randn_like(X)
    y_triton.backward(dLdy, retain_graph = True) # Retain the graph, or the PyTorch will reset the graph after call backprop
    dLdx_tri, dLdw_tri, dLdb_tri = [tensor.grad.clone() for tensor in [X, w, b]]
    # Reset the gradient
    X.grad, w.grad, b.grad = None, None, None,
    y_torch.backward(dLdy, retain_graph=True)
    dLdx_tor, dLdw_tor, dLdb_tor = [tensor.grad.clone() for tensor in [X, w, b]]

    torch.testing.assert_close(dLdx_tor, dLdx_tri, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_tor, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_tor, atol=1e-2, rtol=0)

    print("Backward pass test PASS!")

