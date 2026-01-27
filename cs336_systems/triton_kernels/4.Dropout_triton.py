import triton
import triton.language as tl
import torch
import jaxtyping
from jaxtyping import Float
from torch import Tensor
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

autotune_configs = [
    # Small tiles
    triton.Config({'TILE_N': 32,}),
]


@triton.autotune(configs=autotune_configs, key=['N']) # Everytime N changes, we re-autotune
@triton.jit
def _dropout(
    x_ptr, OUT_ptr,
    N,
    x_stride, OUT_stride,
    p, # Dropout rate
    seed,
    TILE_N: tl.constexpr,
):
    """
    Note: the mask should be computed inside the triton kernel, rather than precomputed
    in the wrapper function, which will leads to extra memmove cost of a whole masking 
    tensor of size (N,)
    """
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
    OUT = tl.zeros((TILE_N,), dtype = tl.float32)


    # Load the block
    x_TILE = tl.load(x_block_ptr, boundary_check=(0,), padding_option="zero")
    
    # Build dropout mask
    offsets = TILE_IDX * TILE_N + tl.arange(0, TILE_N)
    TILE_mask = tl.rand(seed, offsets)
    # Apply dropout
    x_TILE_drop = TILE_mask < p

    # Write the result
    OUT = tl.where(x_TILE_drop, 0.0, x_TILE/(1-p)) # Re-Normalize
    
    # Write to the correct base pointer
    tl.store(out_block_ptr, OUT, boundary_check=(0,))
    

        


def dropout(x:torch.Tensor, p, seed):
    """
    Dropout applied on a vector x.
    """
    OUT = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["TILE_N"]),)
    # The triton kernel
    _dropout[grid](
        x, OUT,
        n_elements,
        1,1,
        p, seed,
    )
    return OUT


if __name__ == "__main__":
    x = torch.randn((8,), device=DEVICE, dtype=torch.float32)
    for i in range(10):
        output = dropout(x, p=0.1, seed=i)
        print(output)
        output = dropout(x, p=0.1, seed=i)
        print(output)
