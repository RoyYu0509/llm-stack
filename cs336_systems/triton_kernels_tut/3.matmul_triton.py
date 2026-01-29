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
import triton.testing as tt
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

    torch.testing.assert_close(OUT_torch, OUT_triton, atol=atol, rtol=rtol)
    print("PASS!")


# Optional: Enable Triton interpreter for better debugging
import os
os.environ['TRITON_INTERPRETER'] = '1'  # Enable Triton interpreter for better debugging

""" Autotuning configurations for the Triton kernel """
# Set of configurations to try during autotuning (Hyperparameter search space)
autotune_configs = [
    # Small tiles
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    # Medium tiles
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    # Large tiles
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    # Different group sizes
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE': 4}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE': 16}, num_stages=4, num_warps=8),
    # Different K dimensions
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
]


@triton.autotune(configs=autotune_configs, key=['M', 'N', 'K']) # Everytime M, N, K changes, we re-autotune
@triton.jit
def _matmul_triton(
    A_ptr, B_ptr, OUT_ptr,
    M, N, K,
    stride_A_M, stride_A_K,
    stride_B_K, stride_B_N,
    stride_OUT_M, stride_OUT_N,
    # Set Customized Hyperparameters to be autotuned
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, # Tile.shape = [BLOCK_M, BLOCK_K] & [BLOCK_K, BLOCK_N]
    GROUP_SIZE: tl.constexpr  # How many rows of PIDs in one SM
):  
    ########################################################
    # Intuition: #
    ########################################################
    """
    The matrix has been grouped rearraging PIDs for better shared memory utilization. 
    
    * Each cell computed by which PID?
    ------------------------------------------------------
            Ungrouped                 Grouped
        [0,   1,   2,  3]        [0,  2, |  8, 10]
        [4,   5,   6,  7]  -->   [1,  3, |  9, 11]
        [8,   9,  10, 11]        --------|--------
        [12, 13,  14, 15]        [4,  6, | 12, 14]
                                 [5,  7, | 13, 15] 

    * Ungrouped Resulted DataLoading:
    ------------------------------------------------------
        - SM1 = {PID_0,1,2,3}        A[r0], B[c0,c1,c2,c3]
        - SM2 = {PID_4,5,6,7}        A[r1], B[c0,c1,c2,c3]
        - SM3 = {PID_8,9,10,11}      A[r2], B[c0,c1,c2,c3]
        - SM4 = {PID_12,13,14,15}    A[r3], B[c0,c1,c2,c3]

    Therefore, our goal is to reassign each program in groups to compute OUTPUT in a specific way
    so that the overall loading can be minimized.

    * Grouped Resulted DataLoading:
    ------------------------------------------------------
        - SM1 = {PID_0,1,2,3}        A[r0, r1], B[c0, c1]
        - SM2 = {PID_4,5,6,7}        A[r2, r3], B[c0, c1]
        - SM3 = {PID_8,9,10,11}      A[r0, r1], B[c2, c3]
        - SM4 = {PID_12,13,14,15}    A[r2, r3], B[c2, c3]

    We see that the data loading has been decrease.
    """
    ########################################################
    # The Actual: #
    ########################################################
    """
    1. Flat PIDs (never change)
    PID = program_id(0) = 0,1,2,3,4,5,...

    2. Grouping (who runs together)
    group_id = PID // (GROUP_SIZE * num_tiles_N)

    3. Tile mapping (what each PID computes) decides which output tile each PID owns
    PID_M = first_row_of_group + local_row      # ROW
    PID_N = local_col                           # COL

    
    The matrix is computed by regrouping program IDs so that PIDs assigned to the same SM 
    cooperate on a rectangular block of the output matrix.

    Grouping does not change program IDs themselves; instead, it changes how each PID maps 
    to (row, column) tiles in the output.

    This allows each SM to reuse A rows and B columns across multiple PIDs, reducing redundant 
    global memory loads and improving shared memory efficiency.
    """
    # Compute the PID and overall grid dimensions
    PID = tl.program_id(0)
    num_TILES_along_M = tl.cdiv(M, BLOCK_M) # Number of tiles along A's M dimension
    num_TILES_along_N = tl.cdiv(N, BLOCK_N) # Number of tiles along B's N dimension
    # Check which group this PID belongs to
    num_PID_per_group = GROUP_SIZE * num_TILES_along_N # Total PIDs per SM(group) = (# rows of PIDs per group) * (# PIDs along columns)
    original_group_id = PID // num_PID_per_group # Which SM is this PID assigned to
    new_group_starting_row = original_group_id * GROUP_SIZE # The new starting row that this group of PIDs will handle
    # Adjust group size for boundary conditions
    GROUP_SIZE_Adjusted = min(num_TILES_along_M - new_group_starting_row, GROUP_SIZE)
    """
    For example, if the current PID == 5
        - PID = 5
        - num_PID_per_group = 2 * 2 = 4
        - original_group_id = 5 // 4  = 1
        - local_PID = 1
        - new_group_starting_row = 1 * 2 = 2
        - GROUP_SIZE_Adjusted = 2 (not the last row tile)

        - PID_M = (the new starting row handled by this group) + (rank of this PID in its own group) % (# PID_row per group) = NEW_ROW
            - num_PID_per_group = 4; 
            - PID % num_PID_per_group = rank of this PID in its own group = {0,1,2,3} = 5 % 4 = 1
                = 2 + 1 % 2 = 3

        - PID_N = (rank of this PID in its own group) // (# PID_row per group) = NEW_COLUMN
                = 1 // 2 = 0

    Now, PID 5 will write at OUT[PID_M, PID_N] = OUT[3, 0]
    The whole group = {PID4, PID5, PID6, PID7} now 
    """
    PID_M = new_group_starting_row + ((PID % num_PID_per_group) % GROUP_SIZE_Adjusted)
    PID_N = (PID % num_PID_per_group) // GROUP_SIZE_Adjusted


    # Compute the starting offsets of A, B, and OUT for this PID
    offsets_A_M = PID_M * BLOCK_M + tl.arange(0, BLOCK_M) # grab the tile rows of A
    offsets_B_N = PID_N * BLOCK_N + tl.arange(0, BLOCK_N) # grab the tile columns of B
    offsets_K = tl.arange(0, BLOCK_K)                     # grab the K dimension

    # Compute the starting offsets of A, B, and OUT for this PID
    """
    offsets_A_M[:, None]: shape (BLOCK_M, 1) = [[m1], [m2], ..., [mBLOCK_M]]
    offsets_K[None, :]: shape (1, BLOCK_K) = [[k1, k2, ..., kBLOCK_K]]
    
    A_tile_offsets: shape (BLOCK_M, BLOCK_K) = 
        [
            [m1*k1, m1*k2, ..., m1*kBLOCK_K], # Tile A row 1
            [m2*k1, m2*k2, ..., m2*kBLOCK_K], # Tile A row 2
            ...
        ]
    
    B_tile_offsets: shape (BLOCK_K, BLOCK_N) =
        [
            [k1*n1, k1*n2, ..., k1*nBLOCK_N], # Tile B column 1
            [k2*n1, k2*n2, ..., k2*nBLOCK_N], # Tile B column 2
            ...
        ]
    
    OUT_tile_offsets: shape (BLOCK_M, BLOCK_N) =
        [
            [m1*n1, m1*n2, ..., m1*nBLOCK_N],      # Tile OUT row 1
            [m2*n1, m2*n2, ..., m2*nBLOCK_N],      # Tile OUT row 2
            [ ...                          ],      # Tile OUT row ...
            [mBLOCK_M*n1, ..., mBLOCK_M*nBLOCK_N]  # Tile OUT row BLOCK_M
        ]
    """
    A_tile_offsets = offsets_A_M[:, None] * stride_A_M + offsets_K[None, :] * stride_A_K
    B_tile_offsets = offsets_K[:, None] * stride_B_K + offsets_B_N[None, :] * stride_B_N
    

    # Initialize the accumulator for the output tile
    # Accumulate in fp32
    OUT_tile_accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension TILE
    for k in range(0, K, BLOCK_K):
        # Define masks for boundary conditions on matrix A and B
        mask_A = (offsets_A_M[:, None] < M) & ( (k + offsets_K[None, :]) < K) # shape (BLOCK_M, BLOCK_K), mask out element > M or > K
        mask_B = ( (k + offsets_K[:, None]) < K) & (offsets_B_N[None, :] < N) # shape (BLOCK_K, BLOCK_N), mask out element > K or > N
        # Load tiles from A and B into shared memory
        A_tile = tl.load(A_ptr + A_tile_offsets, mask=mask_A, other=0.0)
        B_tile = tl.load(B_ptr + B_tile_offsets, mask=mask_B, other=0.0)

        # Perform the matrix multiplication for the tile
        OUT_tile_accumulator += tl.dot(A_tile, B_tile)

        # Move the K dimension offset
        A_tile_offsets += BLOCK_K * stride_A_K
        B_tile_offsets += BLOCK_K * stride_B_K
    
    # Stored in fp16
    OUT_tile_accumulator = OUT_tile_accumulator.to(tl.float16)

    # Write in
    OUT_TILE_offsets = offsets_A_M[:, None] * stride_OUT_M + offsets_B_N[None, :] * stride_OUT_N
    OUT_mask = (offsets_A_M[:, None] < M) & (offsets_B_N[None, :] < N)
    tl.store(OUT_ptr + OUT_TILE_offsets, OUT_tile_accumulator, mask = OUT_mask)



def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using Triton kernel.
    A: (M, K)
    B: (K, N)
    Returns:
    OUT: (M, N)

    Algorithm Overview:
    -------------------
    1. Divide matrices A and B into tiles of ROW_TILEs (BLOCK_M,) and COL_TILEs (BLOCK_N,) respectively.
    2. Flatten the 2D grid of tiles into a 1D grid for Triton kernel launch, ie: [ {A[r1], B[c1]}, {A[r1], B[c2]}, {...} ] 
    3. Each program take an A[row_i] and B[col_j] tile to compute OUT[row_i, col_j].
    4. Regroup PIDs to SMs to improve shared memory utilization.

    """
    assert A.dim() == 2 and B.dim() == 2, "Input tensors must be 2D matrices."
    assert A.shape[1] == B.shape[0], "Inner dimensions must match for matrix multiplication."
    M, K = A.shape
    K, N = B.shape
    OUT = torch.zeros((M, N), device=A.device, dtype=A.dtype)

    # How the META is constructed in Python (see below, full details see notion)
    """
    Python runtime:
    1. Define fn grid = lambda META: ...
        → grid exists, META does not exist, nothing executed

    2. Call _matmul_triton[grid](...)
        → Triton:
            a. picks config
          * b. builds META
            c. calls grid(META)
            d. compiles _matmul_triton[grid](...)
            e. runs benchmark
            f. picks best config
        → GPU runs kernel with compiled code
    """
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    _matmul_triton[grid](
        A, B, OUT,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        OUT.stride(0), OUT.stride(1),
    )

    return OUT

# ====== Benchmarking TFLOPs  =======
configs_tflops = tt.Benchmark(
    x_names=["M", "N", "K"],
    x_vals=[128 * i for i in range(2, 33)],
    line_arg="provider",
    line_vals=["triton", "torch"],
    line_names=["Triton", "Torch"],
    plot_name="matmul-tflops",
    args={},
)

@tt.perf_report(configs_tflops)
def benchmark_matmul_tflops(M, N, K, provider):
    device = DEIVCE
    dtype = torch.float16

    A = torch.randn((M, K), device=device, dtype=dtype)
    B = torch.randn((K, N), device=device, dtype=dtype)

    ms_min, ms_med, ms_max = tt.do_bench(
        lambda: matmul_triton(A, B) if provider == "triton"
        else torch.matmul(A, B),
        quantiles=(0.0, 0.5, 1.0),
    )

    flops = 2 * M * N * K

    def perf(ms):
        return flops / (ms * 1e-3) / 1e12

    mean = perf(ms_med)
    min_ = perf(ms_max)
    max_ = perf(ms_min)

    return mean, min_, max_


# ====== Benchmarking Memory Bandwidth (GB/s)  =======
configs_memory = tt.Benchmark(
    x_names=["M", "N", "K"],
    x_vals=[128 * i for i in range(2, 33)],
    line_arg="provider",
    line_vals=["triton", "torch"],
    line_names=["Triton", "Torch"],
    plot_name="matmul-memory-bandwidth",
    args={},
)

@tt.perf_report(configs_memory)
def benchmark_matmul_memory(M, N, K, provider):
    device = DEIVCE
    dtype = torch.float16

    A = torch.randn((M, K), device=device, dtype=dtype)
    B = torch.randn((K, N), device=device, dtype=dtype)

    ms_min, ms_med, ms_max = tt.do_bench(
        lambda: matmul_triton(A, B) if provider == "triton"
        else torch.matmul(A, B),
        quantiles=(0.0, 0.5, 1.0),
    )

    # Memory transferred: A (M x K) + B (K x N) + OUT (M x N)
    # Each element is float16 (2 bytes)
    bytes_transferred = 2 * (M * K + K * N + M * N)

    def memory_bandwidth(ms):
        # ms is in milliseconds, convert to seconds
        # GB/s = bytes / (ms * 1e-3) / 1e9
        return bytes_transferred / (ms * 1e-3) / 1e9

    mean = memory_bandwidth(ms_med)
    min_ = memory_bandwidth(ms_max)
    max_ = memory_bandwidth(ms_min)

    return mean, min_, max_


# ====== Benchmarking Arithmetic Intensity (FLOPs/byte)  =======
configs_intensity = tt.Benchmark(
    x_names=["M", "N", "K"],
    x_vals=[128 * i for i in range(2, 33)],
    line_arg="provider",
    line_vals=["triton", "torch"],
    line_names=["Triton", "Torch"],
    plot_name="matmul-arithmetic-intensity",
    args={},
)
@tt.perf_report(configs_intensity)
def benchmark_matmul_intensity(M, N, K, provider):
    device = DEIVCE
    dtype = torch.float16

    A = torch.randn((M, K), device=device, dtype=dtype)
    B = torch.randn((K, N), device=device, dtype=dtype)

    ms_min, ms_med, ms_max = tt.do_bench(
        lambda: matmul_triton(A, B) if provider == "triton"
        else torch.matmul(A, B),
        quantiles=(0.0, 0.5, 1.0),
    )

    # FLOPs and bytes transferred
    flops = 2 * M * N * K
    # Arithmetic Intensity = FLOPs / Bytes (input data only: A + B, not output)
    # A: M x K, B: K x N, each element is float16 (2 bytes)
    bytes_transferred = 2 * (M * K + K * N)

    # Arithmetic Intensity = FLOPs / Bytes (in units of FLOPs per byte)
    intensity = flops / bytes_transferred

    # Return intensity (constant for fixed M, N, K regardless of execution time)
    # Return as (mean, min_, max_) where they're all the same value
    return intensity, intensity, intensity


if __name__ == "__main__":
    test_matmul_kernel(size=(512, 512, 64))
    import sys
    if "--benchmarking" in sys.argv:
        os.makedirs("cs336_systems/triton_kernels/benchmarking", exist_ok=True)
        benchmark_matmul_tflops.run(save_path='cs336_systems/triton_kernels/benchmarking', print_data=False)
        benchmark_matmul_memory.run(save_path='cs336_systems/triton_kernels/benchmarking', print_data=False)
        benchmark_matmul_intensity.run(save_path='cs336_systems/triton_kernels/benchmarking', print_data=False)