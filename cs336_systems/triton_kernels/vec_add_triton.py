import triton
import triton.language as tl
import torch
import jaxtyping
from jaxtyping import Float
from torch import Tensor
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def add_kernel(
    x_ptr, y_ptr,
    out_ptr, 
    x_ROW, 
    BLOCK_SIZE: tl.constexpr,  # Set this to be compiled time constant; Change will leads to another kernel compilation.
): 
    """
    In triton, multiple 
    """
    # Get the program
    PID = tl.program_id(axis=0)
    # Define the entries Start and the Range of the current Program
    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
   
    # This mask is for the program when called on the last tile
    # In case the `x_ROW` is not perfectly divided, each will create an extra block
    # that a part of it is out of the tensor.
    mask = offsets < x_ROW 
    
    # Load data to SRAM (on-chip) from HBM (off-chip), with safe masking
    x = tl.load(x_ptr+offsets, mask=mask, other=None)
    y = tl.load(y_ptr+offsets, mask=mask, other=None)
    
    # Do computation on-ship
    output = x + y

    # Write data back to HBM (off-chip), with safe masking
    tl.store(out_ptr + offsets, output,  mask=mask)




def triton_add(x, y):
    # Prepare buffer
    out = torch.empty_like(x)

    # Checks
    assert x.device == y.device

    # Define lunch grid
    X_ROW = out.numel()
    grid = lambda properties: (triton.cdiv(X_ROW, properties['BLOCK_SIZE']),) # Return a tuple represeting grid structure
    # Calls the kernel, indexed by `grid`
    add_kernel[grid](
        x, y, 
        out,
        X_ROW,
        BLOCK_SIZE=1024
    )

    return out


def test_add_kernel(
    size, 
    atol=1e-3, # Absolute raw tol; useful when the true is 0
    rtol=1e-3, # Error relative to the size tol; useful when the true is extreme
    device = DEVICE
):
    # Create test data
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)
    out_triton = triton_add(x, y)
    out_ref = x + y
    # Compare
    torch.testing.assert_close(out_triton, out_ref, atol=atol, rtol=rtol)
    print("PASS")
    return 0



# This is a Triton benchmarking wrapper that does the followings:
"""
for size in x_vals:                 # sizes = 2^12 ... 2^27
    for provider in line_vals:      # ['triton', 'torch']
        result = benchmark(size=size, provider=provider)
        record(result)
"""
# When we call benchmark.run(), it will execute the wrapper function like this:
"""       
benchmark.run()
 ├─ benchmark(size=4096, provider='triton')
 │    └─ do_bench(lambda: triton_add(x, y))
 ├─ benchmark(size=4096, provider='torch')
 │    └─ do_bench(lambda: x + y)
 ├─ benchmark(size=8192, provider='triton')
 │    └─ do_bench(lambda: triton_add(x, y))
 ├─ benchmark(size=8192, provider='torch')
 │    └─ do_bench(lambda: x + y)
"""
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], 
        x_vals=[2**i for i in range(12, 28, 1)], # Try different vector sizes
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton Vec Add", "Torch Vec Add"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel = 'GB/s',
        plot_name="vec-add_performance",
        args={},
    )
)
def benchmark(size, provider):
    # Create input data
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95]

    # Record the execution runtime: meadian, 5th-percentile, 95th-percentile
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x+y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_add(x,y), quantiles=quantiles)

    # Benchmark the Data Transferation Efficiency in GB/s: 
    # == [num_of_transf_ops * sizeof(x) * convert_to_GB] / [runtime * convert_to_second]
    gbps = lambda ms: 3 * (x.numel() * x.element_size()) * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # Test correctness
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=4098)

    # Test time complexity
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=True)