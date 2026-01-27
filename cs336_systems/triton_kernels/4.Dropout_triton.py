import triton
import triton.language as tl
import torch
import jaxtyping
from jaxtyping import Float
from torch import Tensor
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def _dropout(
    x_ptr, OUT_ptr,
    N,
    x_stride, OUT_stride,
    TILE_N: tl.constexpr,
    num_stges: tl.constexpr
):
    # Get the PID
    TILE_IDX = tl.program_id(0)
    # Initialize pointer
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(N,),
        strides=(x_stride,),
        offsets=(TILE_IDX * TILE_N,),
        block_shape=(TILE_N,),
        order=(0,)
    )

    out_block_ptr = tl.make_block_ptr(
        OUT_ptr,
        shape=(N,),
        strides=(OUT_stride,),
        offsets=(TILE_IDX * TILE_N),
        block_shape=(TILE_N,),
        order=(0,)
    )

    # Prepare a buffer
    OUT = tl.zeros((N,), dtype = tl.float32)

    for i in range(tl.cidv(N, TILE_N)):
        # Load the block
        x_TILE = tl.load(x_block_ptr, boundary_check=(0,), padding_option="zero")




def dropout(x:torch.Tensor, p, seed):
    """
    Dropout applied on a vector x.
    """
    OUT = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    # The triton kernel
    _dropout[grid](
        x, OUT,
        n_elements,
        p, seed,
        BLOCK_SIZE=1024
    )
    return OUT
