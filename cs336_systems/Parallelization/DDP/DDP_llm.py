import argparse
import os
import time
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import tqdm

from cs336_basics.lm import TransformerLM
from cs336_basics.train.loss import cross_entropy
from cs336_basics.train.optimizer import AdamW
from cs336_basics.transfromer.scaled_dot_prod_attention import (
    flash_attention_my_triton,
    scaled_dot_product_attention,
    vectorized_attention_torch,
)

DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

ATTN_KERNELS = [
    ("CompTorch", vectorized_attention_torch),
    ("Naive Attention", scaled_dot_product_attention),
    ("MyTriton", flash_attention_my_triton),
]


class TokenStreamDataset(Dataset):
    def __init__(self, path: str, context_len: int):
        self.tokens_stream = np.load(path, mmap_mode="r")
        self.context_len = context_len
        self.N = self.tokens_stream.shape[0]

        if self.N <= self.context_len:
            raise RuntimeError("context_length is too large for the provided data.")

    def __len__(self) -> int:
        return self.N - self.context_len - 1

    def __getitem__(self, idx: int):
        x_np = self.tokens_stream[idx : idx + self.context_len]
        y_np = self.tokens_stream[idx + 1 : idx + 1 + self.context_len]

        # Convert only this sample to int64 for embedding lookup.
        x = torch.tensor(x_np, dtype=torch.long)
        y = torch.tensor(y_np, dtype=torch.long)
        return x, y


def set_dist_env(rank: int, world_size: int, backend: str = "gloo") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    init_process_group(backend=backend, rank=rank, world_size=world_size)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def naive_LLM_DDP(
    rank: int,
    world_size: int,
    data_path: str,
    context_length: int,
    model_args: dict,
    optimizer_args: dict,
    loss_fn: Callable,
    epochs: int,
    batch_size: int,
    backend: str,
    print_every: int = 10,
):
    set_dist_env(rank, world_size, backend=backend)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    print(f"Rank {rank} initializing dataset and dataloader...")
    dataset = TokenStreamDataset(data_path, context_length)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loc_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
    )

    print(f"Rank {rank} initializing model and optimizer...")
    model = TransformerLM(**model_args).to(device)

    # Ensure all ranks start from identical weights.
    print(f"Rank {rank} broadcasting initial model parameters...")
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    optimizer = AdamW(model.parameters(), **optimizer_args)

    total_avg_comm_time = 0.0
    total_avg_epoch_time = 0.0

    model.train()
    for epoch in range(epochs):
        print(f"Rank {rank} starting epoch {epoch + 1}/{epochs}")
        sampler.set_epoch(epoch)
        acc_loss = 0.0
        comm_time_epoch = 0.0

        _synchronize_if_cuda(device)
        t0 = time.perf_counter()

        i = 0
        for bat_X, bat_y_ref in tqdm.tqdm(loc_loader, desc=f"Rank {rank} Epoch {epoch + 1}", unit="batch"):
            bat_X = bat_X.to(device, non_blocking=True)
            bat_y_ref = bat_y_ref.to(device, non_blocking=True)

            optimizer.zero_grad()
            bat_y_pred = model(bat_X)
            loss = loss_fn(bat_y_pred, bat_y_ref)
            loss.backward()

            _synchronize_if_cuda(device)
            t1 = time.perf_counter()
            for para in model.parameters():
                if para.grad is None:
                    continue
                dist.all_reduce(para.grad, op=dist.ReduceOp.SUM)
                para.grad /= world_size
            _synchronize_if_cuda(device)
            t2 = time.perf_counter()
            comm_time_epoch += t2 - t1

            optimizer.step()
            acc_loss += loss.item()

            i += 1
            if i == 100:  # Only do detailed timing for the first 10 batches to reduce overhead.
                break

        _synchronize_if_cuda(device)
        t3 = time.perf_counter()
        epoch_time = t3 - t0

        timing = torch.tensor([comm_time_epoch, epoch_time], device=device)
        dist.all_reduce(timing, op=dist.ReduceOp.SUM)
        timing /= world_size

        avg_comm_epoch = timing[0].item()
        avg_epoch_time = timing[1].item()

        total_avg_comm_time += avg_comm_epoch
        total_avg_epoch_time += avg_epoch_time

        if ((epoch + 1) % print_every == 0 or epoch == 0):
            print(
                f"Rank {rank} | "
                f"Epoch {epoch + 1:04d} | Avg comm: {avg_comm_epoch:.4f}s | "
                f"Avg total: {avg_epoch_time:.4f}s | Local loss sum: {acc_loss:.4f}"
            )

            print(
                f"Parameters sample (rank {rank}): {model.parameters().__next__()[0, :5].tolist()}"
            )

    if rank == 0:
        print("\n=== Final Timing Results (avg across ranks) ===")
        print(f"Average communication time per epoch: {total_avg_comm_time / epochs:.4f} seconds")
        print(f"Average total time per epoch: {total_avg_epoch_time / epochs:.4f} seconds")

    dist.barrier()
    destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Training TransformerLM with manual DDP-style gradient sync and lazy data loading"
    )

    assert torch.cuda.is_available(), "CUDA is required for this script."
    parser.add_argument("--TR_BAT_SIZE", type=int, default=8)
    parser.add_argument("--EPOCHES", type=int, default=3)
    parser.add_argument("--TRAIN_PATH", type=str, required=True)
    parser.add_argument("--DTYPE", type=str, default="float32", choices=list(DTYPE_DICT.keys()))
    parser.add_argument( "--ATTN_KERNEL", type=str, default="Naive Attention",
        choices=[name for name, _ in ATTN_KERNELS],
    )

    
    parser.add_argument("--CONTEXT_LENGTH", type=int, default=256)
    
    parser.add_argument("--PRINT_EVERY", type=int, default=1)
    parser.add_argument("--VOCAB_SIZE", type=int, default=10000)
    parser.add_argument("--ROPE_THETA", type=float, default=10_000.0)
    parser.add_argument("--NUM_LAYERS", type=int, default=16)
    parser.add_argument("--D_MODEL", type=int, default=256)
    parser.add_argument("--NUM_HEADS", type=int, default=8)
    parser.add_argument("--D_FF", type=int, default=256*4)

    parser.add_argument("--LR", type=float, default=3e-4)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01)
    parser.add_argument("--BETA1", type=float, default=0.9)
    parser.add_argument("--BETA2", type=float, default=0.999)
    parser.add_argument("--ADAM_EPS", type=float, default=1e-8)

    args = parser.parse_args()

    dtype = DTYPE_DICT[args.DTYPE]
    attention_fn = dict(ATTN_KERNELS)[args.ATTN_KERNEL]
    backend = "nccl"
    world_size = torch.cuda.device_count()

    model_kwargs = dict(
        vocab_size=args.VOCAB_SIZE,
        context_length=args.CONTEXT_LENGTH,
        num_layers=args.NUM_LAYERS,
        d_model=args.D_MODEL,
        heads_num=args.NUM_HEADS,
        d_ff=args.D_FF,
        theta=args.ROPE_THETA,
        device="cpu",
        dtype=dtype,
        attention_fn=attention_fn,
    )

    optim_kwargs = dict(
        lr=args.LR,
        weight_decay=args.WEIGHT_DECAY,
        betas=(args.BETA1, args.BETA2),
        eps=args.ADAM_EPS,
    )

    mp.spawn(
        fn=naive_LLM_DDP,
        args=(
            world_size,
            args.TRAIN_PATH,
            args.CONTEXT_LENGTH,
            model_kwargs,
            optim_kwargs,
            cross_entropy,
            args.EPOCHES,
            args.TR_BAT_SIZE,
            backend,
            args.PRINT_EVERY,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
