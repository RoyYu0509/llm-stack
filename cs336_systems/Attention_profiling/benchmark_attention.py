import argparse
import os
import timeit
from typing import Callable

import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
from tqdm import tqdm

from cs336_basics.transfromer.scaled_dot_prod_attention import (
    flash_attention_my_triton,
    flash_attention_torch,
    scaled_dot_product_attention,
)


DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

KERNELS: dict[str, Callable] = {
    "Naive Attention": scaled_dot_product_attention,
    "MyTriton": flash_attention_my_triton,
    "VecTorch": flash_attention_torch,
}


def _sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark_attention_kernel(
    kernel_name: str,
    attention_fn: Callable,
    warmup_iter: int,
    profile_iter: int,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    is_causal: bool,
) -> dict:
    """
    Benchmark the attention kernel with separate forward and backward timing.
    
    Procedures:
        1. Create random Q, K, V tensors
        2. Run several warm-up iterations
        3. Run several benchmarking iterations, measuring forward and backward time separately
        4. Return the average forward, backward, and total time per iteration in milliseconds
    """
    print(f"\n=== Benchmarking {kernel_name} ===")
    
    # Create Q, K, V tensors
    shape = (batch_size * num_heads, seq_len, head_dim)
    print(f"Using shape: {shape}, device: {device}, dtype: {dtype}")
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    
    # Warm-up iterations
    with nvtx.range(f"Warm-up [{kernel_name}]"):
        for _ in tqdm(range(warmup_iter), desc=f"Warmup [{kernel_name}]", unit="iter"):
            output = attention_fn(q, k, v, is_causal)
            loss = output.square().mean()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None

    # Prepare timing buffers
    forward_times = []
    backward_times = []
    
    # Benchmarking iterations
    with nvtx.range(f"Benchmark [{kernel_name}]"):
        for _ in tqdm(range(profile_iter), desc=f"Benchmark [{kernel_name}]", unit="iter"):
            # Forward pass
            
            with nvtx.range(f"{kernel_name}:forward"):
                _sync_device(device)
                t0 = timeit.default_timer()
                output = attention_fn(q, k, v, is_causal)
                _sync_device(device)
                t1 = timeit.default_timer()
                loss = output.square().mean()

            # Backward pass
            with nvtx.range(f"{kernel_name}:backward"):
                _sync_device(device)
                t2 = timeit.default_timer()
                loss.backward()
                _sync_device(device)
                t3 = timeit.default_timer()

            q.grad = None
            k.grad = None
            v.grad = None

            forward_times.append(t1 - t0)
            backward_times.append(t3 - t2)

    # Record the average times from the benchmarking iterations
    result = {
        "kernel": kernel_name,
        "forward_ms": 1000.0 * sum(forward_times) / len(forward_times),
        "backward_ms": 1000.0 * sum(backward_times) / len(backward_times),
        "total_ms": 1000.0 * (sum(forward_times) + sum(backward_times)) / len(forward_times),
    }

    del q, k, v
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark attention kernel runtimes")
    parser.add_argument("--DEVICE", type=str, default="cuda")
    parser.add_argument("--DTYPE", type=str, default="float16", choices=list(DTYPE_DICT.keys()))
    parser.add_argument("--BATCH_SIZE", type=int, default=8)
    parser.add_argument("--NUM_HEADS", type=int, default=16)
    parser.add_argument("--SEQ_LEN", type=int, default=256)
    parser.add_argument("--HEAD_DIM", type=int, default=64)
    parser.add_argument("--WARMUP_ITER", type=int, default=20)
    parser.add_argument("--PROFILE_ITER", type=int, default=100)
    parser.add_argument("--IS_CAUSAL", action="store_true")
    parser.add_argument("--SEED", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.SEED)

    if args.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.DEVICE.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")

    dtype = DTYPE_DICT[args.DTYPE]

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_attention")
    os.makedirs(output_dir, exist_ok=True)

    # Benchmark all kernels
    results = []
    for kernel_name, kernel_fn in KERNELS.items():
        # Clean up CUDA memory from previous kernel
        if args.DEVICE.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        # Benchmark the kernel
        row = benchmark_attention_kernel(
            kernel_name=kernel_name,
            attention_fn=kernel_fn,
            warmup_iter=args.WARMUP_ITER,
            profile_iter=args.PROFILE_ITER,
            batch_size=args.BATCH_SIZE,
            num_heads=args.NUM_HEADS,
            seq_len=args.SEQ_LEN,
            head_dim=args.HEAD_DIM,
            device=args.DEVICE,
            dtype=dtype,
            is_causal=args.IS_CAUSAL,
        )
        row["status"] = "success"
        results.append(row)

    # Save benchmark results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    print("\nBenchmark summary:")
    print(df)
    print(f"\nSaved results to {csv_path}")
    
    # Cleanup CUDA context to prevent nsys from hanging on orphaned processes
    if args.DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Force garbage collection before exit
    import gc
    gc.collect()



if __name__ == "__main__":
    main()