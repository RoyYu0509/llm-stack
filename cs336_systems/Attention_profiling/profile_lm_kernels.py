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
    flash_attention_torch,
    scaled_dot_product_attention,
)


DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

ATTN_KERNELS = [
    ("Naive Attention", scaled_dot_product_attention),
    ("MyTriton", flash_attention_my_triton),
    ("VecTorch", flash_attention_torch),
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

    model.to(device=args.DEVICE, dtype=args.DTYPE)
    forward_fn = torch.compile(model.forward) if args.COMPILED else model.forward
    param_list = list(model.parameters())

    optimizer = AdamW(
        param_list,
        args.LR,
        args.WEIGHT_DECAY,
        (args.BETA1, args.BETA2),
        eps=args.ADAM_EPS,
    )

    forward_times = []
    backward_times = []

    for _ in tqdm(range(args.WARM_UP_ITER), desc=f"Warmup [{kernel_name}]", unit="iter"):
        inputs, targets = data_loading(
            train_data,
            args.TR_BAT_SIZE,
            args.CONTEXT_LENGTH,
            args.DEVICE,
            offsets,
        )
        optimizer.zero_grad()
        logits = forward_fn(x=inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        grad_clip(param_list, args.GRAD_CLIP)
        optimizer.step()

    for step in tqdm(range(args.PROFILE_ITER), desc=f"Profile [{kernel_name}]", unit="iter"):
        inputs, targets = data_loading(
            train_data,
            args.TR_BAT_SIZE,
            args.CONTEXT_LENGTH,
            args.DEVICE,
            offsets,
        )
        optimizer.zero_grad()

        _sync_device(args.DEVICE)
        with nvtx.range(f"{kernel_name}:forward"):
            t0 = timeit.default_timer()
            logits = forward_fn(x=inputs)
            _sync_device(args.DEVICE)
            t1 = timeit.default_timer()

        loss = cross_entropy(logits, targets)

        _sync_device(args.DEVICE)
        with nvtx.range(f"{kernel_name}:backward"):
            t2 = timeit.default_timer()
            loss.backward()
            _sync_device(args.DEVICE)
            t3 = timeit.default_timer()

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

    parser.add_argument("--WARM_UP_ITER", type=int, required=True)
    parser.add_argument("--PROFILE_ITER", type=int, required=True)

    parser.add_argument("--TRAIN_PATH", type=str, required=True)
    parser.add_argument("--TR_BAT_SIZE", type=int, default=4)
    parser.add_argument("--CONTEXT_LENGTH", type=int, default=256)

    parser.add_argument("--VOCAB_SIZE", type=int, required=True)
    parser.add_argument("--ROPE_THETA", type=float, default=10_000.0)
    parser.add_argument("--NUM_LAYERS", type=int, default=36)
    parser.add_argument("--D_MODEL", type=int, default=1280)
    parser.add_argument("--NUM_HEADS", type=int, default=20)
    parser.add_argument("--D_FF", type=int, default=5120)

    parser.add_argument("--LR", type=float, default=3e-4)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01)
    parser.add_argument("--BETA1", type=float, default=0.9)
    parser.add_argument("--BETA2", type=float, default=0.999)
    parser.add_argument("--ADAM_EPS", type=float, default=1e-8)
    parser.add_argument("--GRAD_CLIP", type=float, default=1.0)
    parser.add_argument("--MAX_ITERS", type=int, default=10_000)
    parser.add_argument("--WARMUP_ITERS", type=int, default=2_000)

    parser.add_argument("--DEVICE", type=str, default="cpu")
    parser.add_argument("--DTYPE", type=str, default="float32", choices=list(DTYPE_DICT.keys()))
    parser.add_argument("--COMPILED", action="store_true")
    parser.add_argument("--ONLY_KERNEL", type=str, default="", choices=["", "Naive Attention", "MyTriton", "VecTorch"])

    args = parser.parse_args()
    args.DTYPE = DTYPE_DICT[args.DTYPE]

    compiled_str = "compiled" if args.COMPILED else "uncompiled"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"profiling_lm_{compiled_str}")
    os.makedirs(output_dir, exist_ok=True)

    train_data = _load_np_tokens(args.TRAIN_PATH, args.DEVICE)
    offsets = torch.arange(args.CONTEXT_LENGTH, dtype=torch.long, device=train_data.device)

    kernels = ATTN_KERNELS
    if args.ONLY_KERNEL:
        kernels = [k for k in ATTN_KERNELS if k[0] == args.ONLY_KERNEL]

    rows = []
    for kernel_name, kernel_fn in kernels:
        print(f"\n=== Profiling LM with {kernel_name} (compiled={args.COMPILED}) ===")
        try:
            row = profile_lm_with_kernel(kernel_name, kernel_fn, args, train_data, offsets)
            row["status"] = "ok"
        except Exception as exc:
            row = {
                "kernel": kernel_name,
                "forward_ms": float("nan"),
                "backward_ms": float("nan"),
                "total_ms": float("nan"),
                "status": f"failed: {exc}",
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "lm_attention_kernel_profile.csv")
    df.to_csv(csv_path, index=False)

    print("\nProfiling summary:")
    print(df)
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
