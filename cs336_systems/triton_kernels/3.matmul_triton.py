"""
- Automatic performance optimization 
- PID re-ordering for improved SRAM shared memory utilization between PIDs
- Multi-dimensional grid launch
- Triton interpreter for improved debugging experience

A @ B = C
(M x K) @ (K x N) = (M x N)


==== Algorithm: Block-wise MatMul ====

- TILE_SIZE: TILE_A.shape = (BLOCK_M, BLOCK_K) and TILE_B.shape = (BLOCK_K, BLOCK_N)

OUT = torch.zeros((M, N))
for M_START in range(0, M, BLOCK_M):
    for N_START in range(0, N, BLOCK_N):
        rslt_MN_accumulated = torch.zeros((BLOCK_M, BLOCK_N))
        for K_START in range(0, K, BLOCK_K):
            A_TILE = A[M_START:M_START+BLOCK_M, K_START:K_START+BLOCK_K]
            B_TILE = B[K_START:K_START+BLOCK_K, N_START:N_START+BLOCK_N]
            rslt_MN_accumulated += A_TILE @ B_TILE
        
        # Write to the OUT buffer
        OUT[M_START:M_START+BLOCK_M, N_START:N_START+BLOCK_N] = rslt_MN_accumulated     # Apply mask
""" 
import torch
import triton
import triton.language as tl
DEIVCE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Testing function
def test_matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1):
    """
    atol & rtol is higher in matmul as floating point multiplication & addition
    is not associative.
    """
    torch.manual_seed(0)
    assert len(size) == 3, "Size should be a tuple of (M, N, K)"
    M, N, K = size
    A = torch.randn((M, K), device=DEIVCE).to(torch.float16)
    B = torch.randn((K, N), device=DEIVCE).to(torch.float16)
    OUT_torch = torch.matmul(A, B)
    OUT_triton = matmul_triton(A, B)

    torch.testing.assert_allclose(OUT_torch, OUT_triton, atol=atol, rtol=rtol)/
    print("PASS!")


# Optional: Enable Triton interpreter for better debugging
import os
os.environ['TRITON_INTERPRETER'] = '1'  # Enable Triton interpreter for better debugging

# Autotuning
autotune_configs = [
    triton.Confi
]


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using Triton kernel.
    A: (M, K)
    B: (K, N)
    Returns:
    OUT: (M, N)
    """
    assert A.dim() == 2 and B.dim() == 2, "Input tensors must be 2D matrices."
    assert A.shape[1] == B.shape[0], "Inner dimensions must match for matrix multiplication."
    M, K = A.shape
    K, N = B.shape
    OUT = torch.zeros((M, N), device=A.device, dtype=A.dtype)

    # Define the launch grid and block sizes
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']))

    _matmul_triton[grid](
        A, B, OUT,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
    )

    return OUT