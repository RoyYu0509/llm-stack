import argparse
import os
import timeit
from typing import Callable

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function

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


def _make_qkv(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size * num_heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def _run_one_step(
    attention_fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = attention_fn(q, k, v, is_causal)
    loss = output.square().mean()
    return output, loss


def _update_progress(label: str, step: int, total: int) -> None:
    end_char = "\n" if step >= total else "\r"
    print(f"{label} {step}/{total}", end=end_char, flush=True)


def _benchmark_kernel(
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
    Benchmark the `attention_fn` kernel under the defined configuration.
    """
    q, k, v = _make_qkv(batch_size, num_heads, seq_len, head_dim, device, dtype)

    for step in range(1, warmup_iter + 1):
        _, loss = _run_one_step(attention_fn, q, k, v, is_causal)
        loss.backward()
        q.grad = None
        k.grad = None
        v.grad = None
        _update_progress(f"[{kernel_name}] Warmup", step, warmup_iter)

    forward_times = []
    backward_times = []

    for step in range(1, profile_iter + 1):
        _sync_device(device)
        t0 = timeit.default_timer()
        _, loss = _run_one_step(attention_fn, q, k, v, is_causal)
        _sync_device(device)
        t1 = timeit.default_timer()

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
        _update_progress(f"[{kernel_name}] Benchmark", step, profile_iter)

    return {
        "kernel": kernel_name,
        "forward_ms": 1000.0 * sum(forward_times) / len(forward_times),
        "backward_ms": 1000.0 * sum(backward_times) / len(backward_times),
        "total_ms": 1000.0 * (sum(forward_times) + sum(backward_times)) / len(forward_times),
    }


def _profile_kernel_once(
    kernel_name: str,
    attention_fn: Callable,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    is_causal: bool,
    output_dir: str,
) -> None:
    q, k, v = _make_qkv(batch_size, num_heads, seq_len, head_dim, device, dtype)

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function(f"{kernel_name}_forward"):
            _, loss = _run_one_step(attention_fn, q, k, v, is_causal)
        with record_function(f"{kernel_name}_backward"):
            loss.backward()

    table = prof.key_averages().table(sort_by="self_cuda_time_total" if "CUDA" in [a.name for a in activities] else "self_cpu_time_total", row_limit=80)
    with open(os.path.join(output_dir, f"profile_{kernel_name.replace(' ', '_')}.txt"), "w", encoding="utf-8") as f:
        f.write(table)

    trace_path = os.path.join(output_dir, f"trace_{kernel_name.replace(' ', '_')}.json")
    prof.export_chrome_trace(trace_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark attention kernel runtimes and profile one kernel once")
    parser.add_argument("--DEVICE", type=str, default="cuda")
    parser.add_argument("--DTYPE", type=str, default="float16", choices=list(DTYPE_DICT.keys()))
    parser.add_argument("--BATCH_SIZE", type=int, default=8)
    parser.add_argument("--NUM_HEADS", type=int, default=16)
    parser.add_argument("--SEQ_LEN", type=int, default=256)
    parser.add_argument("--HEAD_DIM", type=int, default=64)
    parser.add_argument("--WARMUP_ITER", type=int, default=20)
    parser.add_argument("--PROFILE_ITER", type=int, default=100)
    parser.add_argument("--IS_CAUSAL", action="store_true")
    parser.add_argument("--OUTPUT_DIR", type=str, default="")
    parser.add_argument("--SEED", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.SEED)

    if args.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.DEVICE.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")

    dtype = DTYPE_DICT[args.DTYPE]

    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.OUTPUT_DIR or os.path.join(
        here,
        f"benchmark_attention_{args.DEVICE}_{args.DTYPE}_B{args.BATCH_SIZE}_H{args.NUM_HEADS}_T{args.SEQ_LEN}_Dh{args.HEAD_DIM}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over configurations
    results = []
    kernel_items = list(KERNELS.items())
    for idx, (kernel_name, kernel_fn) in enumerate(kernel_items, start=1):
        print(f"\nRunning kernel {idx}/{len(kernel_items)}: {kernel_name}")
        row = _benchmark_kernel(
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
        row["status"] = "ok"
        results.append(row)

        _profile_kernel_once(
            kernel_name=kernel_name,
            attention_fn=kernel_fn,
            batch_size=args.BATCH_SIZE,
            num_heads=args.NUM_HEADS,
            seq_len=args.SEQ_LEN,
            head_dim=args.HEAD_DIM,
            device=args.DEVICE,
            dtype=dtype,
            is_causal=args.IS_CAUSAL,
            output_dir=output_dir,
        )
        profile_kernel_file = kernel_name.replace(" ", "_")
        print(
            f"Saved one-shot profile for {kernel_name} to "
            f"{os.path.join(output_dir, f'profile_{profile_kernel_file}.txt')} and "
            f"{os.path.join(output_dir, f'trace_{profile_kernel_file}.json')}"
        )

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(df)
    print(f"Saved runtime benchmark to {csv_path}")



if __name__ == "__main__":
    main()