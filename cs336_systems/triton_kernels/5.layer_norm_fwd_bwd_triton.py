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
    X_ptr, OUT_ptr, w_ptr, b_ptr, rstd_ptr,
    X_N, X_D,
    stride_N, eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    """
    Compute the sum in multiple TILEs.shape = [BLOCK_N, BLOCK_D]
    """
    # Get which BLOCK is this program processing
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(0)

    X_B_ptr = tl.make_block_ptr(
        X_ptr, shape=(X_N, X_D),
        strides=(stride_N, 1),
        offsets=(row_idx,col_idx),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1,0),
    )

    OUT_B_ptr = tl.make_block_ptr(
        OUT_ptr, shape=(X_N, X_D),
        strides=(stride_N, 1),
        offsets=(row_idx,col_idx),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
        order=(1,0),
    )

    w_B_ptr = tl.make_block_ptr(
        w_ptr, shape=(X_D,),
        strides=(1,),
        offsets=(col_idx,),
        block_shape=(BLOCK_SIZE_D,),
        order=(0,),
    )

    b_B_ptr = tl.make_block_ptr(
        b_ptr, shape=(X_D,),
        strides=(1,),
        offsets=(col_idx,),
        block_shape=(BLOCK_SIZE_D,),
        order=(0,),
    )

    rstd_B_ptr = tl.make_block_ptr(
        rstd_ptr, shape=(X_D,),
        strides=(1,),
        offsets=(col_idx,),
        block_shape=(BLOCK_SIZE_D,),
        order=(0,),
    )

    # Initialize a complete BLOCK_ROW
    SUM = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    for i in range(tl.cdiv(N, BLOCK_SIZE_N)):
        for j in range(tl.cdiv(D, BLOCK_SIZE_D):)

    

    


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_X, normalized_shape, w, b, eps):
        # Batch all leading dimensions, reshape to a 2D tensor.
        X: Float[Tensor, "N, D"] = rearrange(in_X, "... N D -> (... N) D")
        N, D = X.shape
        OUT = torch.empty_like(X)
        mean = torch.empty(normalized_shape, dtype=torch.float32, device=DEVICE)
        rstd = torch.empty(normalized_shape, dtype=torch.float32, device=DEVICE)

        MAX_FUSED_SIZE = 65536 // X.element_size() # number of element that can fits in a 64kb SRAM
        BLOCK_SIZE_N = 1
        BLOCK_SIZE_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(BLOCK_SIZE_N*D)) # Each block shape = [BLOCK_SIZE_N, BLOCK_SIZE]
        if N >  BLOCK_SIZE_D:
            raise RuntimeError("Runtime slowed when one row of X (size >= 64kb) in our kernel")
        # Some random num_wrap config
        num_warps = min(max(BLOCK_SIZE_D//256, 1), 8)

        # Define grid
        num_TILE_N = tl.cdiv(N, BLOCK_SIZE_N)
        num_TILE_D = tl.cdiv(D, BLOCK_SIZE_D)
        _layernorm_fwd[(num_TILE_N,num_TILE_D)](
            X, OUT, w, b, mean, rstd,
            N, D,
            X.stride(0),
            # self-defined meta-parameters
            BLOCK_SIZE_N = BLOCK_SIZE_N,
            BLOCK_SIZE_D = BLOCK_SIZE, 
            # triton official meta-parameters
            num_warps=num_warps
        )

        # Here, ctx is to cache intermediate value for backward pass
        ctx.save_for_backwad(X, w, b, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps

        return OUT



    @staticmethod
    def backward():





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

