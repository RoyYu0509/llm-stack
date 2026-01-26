"""
In this example, we compute the softmax by loading in the entire row as one tile.
TILE_SIZE refers to the `next_power_of_2(n_cols)`

"""

import triton
import triton.language as tl
import torch
import jaxtyping
from jaxtyping import Float
from torch import Tensor
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def memo_naive_softmax(X):
    """
    A memory inefficient softmax implementation
    """
    # Reads MN
    X: Float[Tensor, "M,N"] = X
    # Write M
    R_MAX: Float[Tensor, "M, 1"] = X.max(dim=1).values.unsqueeze(-1)
    # Read MN + M, Write MN
    z = X-R_MAX
    # Read MN, Write MN
    numerator: Float[Tensor, "M, N"] = torch.exp(z)
    # Read MN, Write M
    denominator: Float[Tensor, "M, 1"] = numerator.sum(dim=-1).unsqueeze(-1)
    # Read MN + M, Write MN
    out = numerator / denominator
    # In total 8MN + 4M Memory Access.
    return out


@triton.jit
def _softmax_fwd_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    N, D,
    BLOCK_SIZE: tl.constexpr, # Compiled time variable
    num_stages: tl.constexpr,
):
    # Shape (N, D)
    # BLCOK_SIZE = next power of 2 > D; So that it fits the entire row.
    PID = tl.program_id(0)

    TILE_SIZE = tl.num_programs(0)
    # If we have 4 programs running, TILE_SIZE = 4 meaning we are processing 4 rows at a time
    # If there are at most 6 rows, N = 6, then:
    # Wave 1:
    #   pid 0 -> row 0
    #   pid 1 -> row 1
    #   pid 2 -> row 2
    #   pid 3 -> row 3
    # Wave 2:
    #   pid 0 -> row 4
    #   pid 1 -> row 5
    #   pid 2 -> Idel...
    #   pid 3 -> Idel...
    # 
    # We see all pid increament by TILE_SIZE

    # Iterate over rows
    for row_idx in tl.range(PID, N, step=TILE_SIZE, num_stages=num_stages): # num_stages: the maximum code lines allowed to run
        # Get the starting row and columns of current PID
        pid_input_row_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE) # Note: BLOCK_SIZE >= D

        # Load the entire row
        input_ptrs = pid_input_row_ptr + col_offsets

        # Mask out the extra columns
        mask = col_offsets < D

        # Load a row
        # NOTE: The value of `others` need to be choose carefully.
        # For row_max, `-inf` has no effect
        # For row_sum, `exp(-inf) = 0` has no effect
        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 
        
        # Safe Softmax Computation
        z = row - tl.max(row, axis = -1)  # [,BLOCK_SIZE] - [,1] = [,BLOCK_SIZE]
        numerator = tl.exp(z)   # shape [,BLOCK_SIZE]
        denominator = tl.sum(numerator, axis = -1)  # shape [,1]
        out = numerator/denominator # [,BLOCK_SIZE] / [,1] = [,BLOCK_SIZE]
        
        # Get the starting row of the current PID
        pid_output_row_ptr = output_ptr + row_idx * output_row_stride

        # Write in values
        tl.store(pid_output_row_ptr + col_offsets, out, mask=mask) # only write the mask in values



        

# Fetch the GPU sepcifics
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]     # The number of SM
NUM_REGS = properties["max_num_regs"]           # Total registers available per Streaming Multiprocessor (SM).
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]# The total SRAM in a SM
WARP_SIZE = properties["warpSize"]              # The number of threads in a warp, ie 32

def softmax_triton(X):
    assert X.ndim == 2
    assert X.is_contiguous()
    n_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Guess the number of warps we want to assign per sm 
    warps_per_sm = 4
    # The number of code lines that allowed be executed simultaneously
    num_stages = 2

    y = torch.empty_like(X)

    # Warm-up Compilation
    kernel = _softmax_fwd_kernel.warmup(
        X,y,
        n_rows, n_cols,
        X.stride(0), # X.stride(1) == 1 implicitly
        y.stride(0), # Same
        BLOCK_SIZE = BLOCK_SIZE,
        num_stages = num_stages,
        num_warps = warps_per_sm,
        grid = (1,)
    )
    kernel._init_handles()
    n_reg_needed_per_program = kernel.n_regs
    sram_needed_per_progarm = kernel.metadata.shared
    
    # Compute # of Programs you can run simultaneously on an SM given the register usage
    """
    If:  
        - Each Streaming Multiprocessor (SM) has 65536 registers 

    The Program:
        - Execute on 8 warps at a time.
        - Each warps has 32 threads.
        - Each thread uses 32 registers to execute the program.
        
    Then:
        Each program needs: 32 registers * 
    """
    reg_occupancy = NUM_REGS // (n_reg_needed_per_program * WARP_SIZE * warps_per_sm)
    # Compute # of Programs you can run simultaneously on an SM given the SRAM
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_progarm

    # The maximum # of Programs you can run given all the constraints
    num_programs_per_sm = min(reg_occupancy, sram_occupancy)
    
    # Compute the total number programs we can compute in one wave
    num_programs = min(NUM_SM * num_programs_per_sm, n_rows) # Cap out at `n_rows`

    grid = (num_programs, 1, 1)

    kernel[grid](
        X, y, 
        X.stride(0), # X.stride(1) == 1 implicitly
        y.stride(0), # Same
        n_rows, n_cols
        # The compile-time variables are hidden, as they are already initialized
        # during the warm-up phase.
    )

    return y


# Testing 
def test_softmax_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size, device=device)
    S_triton = softmax_triton(x)
    S_torch = torch.softmax(x, axis=-1)
    torch.testing.assert_close(S_triton, S_torch, atol=atol, rtol=rtol)
    print("PASS!")


# -------------------------------
# Memory GB/s benchmark
# -------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128*i for i in range(2, 50)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'naive'],
        line_names=['Triton', 'Torch', 'Naive'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="GB/s",
        plot_name='softmax-triton-kernel',
        args={'M': 4096}
    )
)
def benchmark(M, N, provider):
    X = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    
    # Make sure the benchmarking is accurate by setting the GPU stream (Optional)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(X, axis=-1))
        # 1 read & 1 write for the torch kernel
        GB_per_sec = lambda ms: 2 * X.numel() * X.element_size() * 1e-9 / (ms * 1e-3)
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax_triton(X))
        # 1 read & 1 write for the triton kernel
        GB_per_sec = lambda ms: 2 * X.numel() * X.element_size() * 1e-9 / (ms * 1e-3)
    if provider == 'naive':
        ms = triton.testing.do_bench(lambda: memo_naive_softmax(X))
        # 8MN + 4M for the naive version
        GB_per_sec = lambda ms: (8 * X.numel() + 4 * X.shape[0])* X.element_size() * 1e-9 / (ms * 1e-3)
    
    return GB_per_sec(ms)


# -------------------------------
# GFLOPs benchmark
# -------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128*i for i in range(2, 50)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'naive'],
        line_names=['Triton', 'Torch', 'Naive'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="GFLOPs/s",
        plot_name='softmax-triton-flops',
        args={'M': 4096}
    )
)
def benchmark_flops(M, N, provider):
    X = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(X, dim=-1))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax_triton(X))
    elif provider == 'naive':
        ms = triton.testing.do_bench(lambda: memo_naive_softmax(X))

    # Softmax FLOPs ≈ 5 * M * N
    flops = 5 * M * N
    gflops_per_sec = flops / (ms * 1e-3) * 1e-9
    return gflops_per_sec


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    test_softmax_kernel(size=(1823, 781))
    # Add benchmarking flag
    import sys
    if "--benchmark" in sys.argv:
        benchmark.run(save_path='cs336_systems/triton_kernels', print_data=False)
    if "--benchflops" in sys.argv:
        benchmark_flops.run(save_path='cs336_systems/triton_kernels', print_data=False)