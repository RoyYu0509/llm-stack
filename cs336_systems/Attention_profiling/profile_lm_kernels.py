import argparse
import os
import timeit

import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
from tqdm import tqdm

from cs336_basics.lm import TransformerLM
from cs336_basics.train.data_loader import data_loading
from cs336_basics.train.loss import cross_entropy
from cs336_basics.train.optimizer import AdamW, grad_clip, lr_scheduler
from cs336_basics.transfromer.scaled_dot_prod_attention import (
    flash_attention_my_triton,
    vectorized_attn_torch_fn,
    scaled_dot_product_attention,
)


DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

ATTN_KERNELS = [
    ("CompTorch", vectorized_attn_torch_fn),
    ("Naive Attention", scaled_dot_product_attention),
    ("MyTriton", flash_attention_my_triton),
]


def _sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _load_np_tokens(path: str, device: str) -> torch.Tensor:
    arr = np.load(path, mmap_mode="r")
    tensor = torch.from_numpy(arr).long()
    if device.startswith("cuda"):
        tensor = tensor.pin_memory()
    return tensor


def profile_lm_with_kernel(
    kernel_name: str,
    attention_fn,
    args,
    train_data: torch.Tensor,
    offsets: torch.Tensor,
):
    """
    Bench the Transformer LM with a specific attention forward kernel {Triton, Compiled Torch, Naive}

    Procedures:
        1. Create the Transformer LM model with the specified attention kernel
        2. Run several warm-up iterations
        3. Run several benchmarking iterations, measuring the forward and backward time separately
        4. Return the average forward, backward, and total time per iteration in milliseconds
    """
    print(f"\n=== Profiling LM with {kernel_name} ===")
    
    # Create the Transformer LM model
    model = TransformerLM(
        args.VOCAB_SIZE,
        args.CONTEXT_LENGTH,
        args.NUM_LAYERS,
        args.D_MODEL,
        args.NUM_HEADS,
        args.D_FF,
        args.ROPE_THETA,
        device=args.DEVICE,
        dtype=args.DTYPE,
        attention_fn=attention_fn,
    )
    # Move the model to the specified device and dtype
    model.to(device=args.DEVICE, dtype=args.DTYPE)
    forward_fn = model.forward
    param_list = list(model.parameters())

    # Create the optimizer
    optimizer = AdamW(
        param_list,
        args.LR,
        args.WEIGHT_DECAY,
        (args.BETA1, args.BETA2),
        eps=args.ADAM_EPS,
    )

    # Prepare to record times
    forward_times = []
    backward_times = []

    # Warm-up iterations
    with nvtx.range(f"Warm-up [{kernel_name}]"):
        for _ in tqdm(range(args.WARM_UP_ITER), desc=f"Warmup [{kernel_name}]", unit="iter"):
            inputs, targets = data_loading(
                train_data,
                args.TR_BAT_SIZE,
                args.CONTEXT_LENGTH,
                args.DEVICE,
                offsets,
            )
            optimizer.zero_grad()
            with torch.autocast(device_type=args.DEVICE, dtype=torch.float16):
                logits = forward_fn(x=inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            grad_clip(param_list, args.GRAD_CLIP)
            optimizer.step()

    # Benchmarking iterations
    with nvtx.range(f"Profile [{kernel_name}]"):
        for step in tqdm(range(args.PROFILE_ITER), desc=f"Profile [{kernel_name}]", unit="iter"):
            # Load data
            inputs, targets = data_loading(
                train_data,
                args.TR_BAT_SIZE,
                args.CONTEXT_LENGTH,
                args.DEVICE,
                offsets,
            )
            optimizer.zero_grad()

            # Forward pass
            _sync_device(args.DEVICE)
            with nvtx.range(f"{kernel_name}:forward"):
                t0 = timeit.default_timer()
                with torch.autocast(device_type=args.DEVICE, dtype=torch.float16):
                    logits = forward_fn(x=inputs)
                _sync_device(args.DEVICE)
                t1 = timeit.default_timer()

            loss = cross_entropy(logits, targets)
            
            # Backward pass
            _sync_device(args.DEVICE)
            with nvtx.range(f"{kernel_name}:backward"):
                t2 = timeit.default_timer()
                loss.backward()
                _sync_device(args.DEVICE)
                t3 = timeit.default_timer()
            
            # Gradient clipping and optimizer step
            grad_clip(param_list, args.GRAD_CLIP)
            optimizer.step()

            lr = lr_scheduler(
                it=step,
                max_learning_rate=args.LR,
                min_learning_rate=args.LR * 0.2,
                warmup_iters=args.WARMUP_ITERS,
                cosine_cycle_aiters=args.MAX_ITERS,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            forward_times.append(t1 - t0)
            backward_times.append(t3 - t2)

    # Record the average times from the profiling iterations
    result = {
        "kernel": kernel_name,
        "forward_ms": 1000.0 * sum(forward_times) / len(forward_times),
        "backward_ms": 1000.0 * sum(backward_times) / len(backward_times),
        "total_ms": 1000.0 * (sum(forward_times) + sum(backward_times)) / len(forward_times),
    }

    del model, optimizer
    if args.DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile LM with different attention kernels")

    # User-defined profiling parameters
    parser.add_argument("--WARM_UP_ITER", type=int, required=True)
    parser.add_argument("--PROFILE_ITER", type=int, required=True)
    parser.add_argument("--TRAIN_PATH", type=str, required=True)
    parser.add_argument("--DEVICE", type=str, default="cpu")
    parser.add_argument("--DTYPE", type=str, default="float32", choices=list(DTYPE_DICT.keys()))
    parser.add_argument("--ALL_KERNEL",  action="store_true")
    parser.add_argument("--ONE_KERNEL", type=str, default="Naive Attention", choices=[name for name, _ in ATTN_KERNELS])
    parser.add_argument("--TORCH_MEM_PROF", action="store_true")


    # DEFAULTs for LM training
    parser.add_argument("--TR_BAT_SIZE", type=int, default=8)
    parser.add_argument("--CONTEXT_LENGTH", type=int, default=518) 

    parser.add_argument("--VOCAB_SIZE", type=int, default=10000)
    parser.add_argument("--ROPE_THETA", type=float, default=10_000.0)
    parser.add_argument("--NUM_LAYERS", type=int, default=18)
    parser.add_argument("--D_MODEL", type=int, default=256)
    parser.add_argument("--NUM_HEADS", type=int, default=64)
    parser.add_argument("--D_FF", type=int, default=3072)

    parser.add_argument("--LR", type=float, default=3e-4)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01)
    parser.add_argument("--BETA1", type=float, default=0.9)
    parser.add_argument("--BETA2", type=float, default=0.999)
    parser.add_argument("--ADAM_EPS", type=float, default=1e-8)
    parser.add_argument("--GRAD_CLIP", type=float, default=1.0)
    parser.add_argument("--MAX_ITERS", type=int, default=10_000)
    parser.add_argument("--WARMUP_ITERS", type=int, default=2_000)

    args = parser.parse_args()
    args.DTYPE = DTYPE_DICT[args.DTYPE]

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiling_lm")
    os.makedirs(output_dir, exist_ok=True)

    # Online loading tokenized training data
    train_data = _load_np_tokens(args.TRAIN_PATH, args.DEVICE)
    offsets = torch.arange(args.CONTEXT_LENGTH, dtype=torch.long, device=train_data.device)

    # If profiling all kernels choices
    if args.ALL_KERNEL:
        kernels = ATTN_KERNELS
    # Only profile the naive attention
    else:  
        if args.ONE_KERNEL is None:
            raise ValueError("Please specify the attention kernel to profile using --ONE_KERNEL") 
        kernels = [kernel for kernel in ATTN_KERNELS if kernel[0] == args.ONE_KERNEL]

    # Start profiling
    rows = []
    for kernel_name, kernel_fn in kernels:
        # Clean up CUDA memory from previous kernel
        if args.DEVICE.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        # Start memory recording AFTER cleanup (if enabled)
        if args.TORCH_MEM_PROF:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        
        # Call the LM Kernel profiling function
        row = profile_lm_with_kernel(kernel_name, kernel_fn, args, train_data, offsets)
        
        # Stop memory recording and dump snapshot BEFORE cleanup
        if args.TORCH_MEM_PROF:
            pickle_path = os.path.join(output_dir, f"lm_attention_{kernel_name}_memory_profile.pickle")
            torch.cuda.memory._dump_snapshot(pickle_path)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"Saved memory snapshot to {pickle_path}")
        
        # Record the profiling results
        row["status"] = "success"
        rows.append(row)

    # Save profiling results to CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "lm_attention_kernel_profile.csv")
    df.to_csv(csv_path, index=False)

    print("\nProfiling summary:")
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
