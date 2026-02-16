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
    trDataset: Dataset,
    valDataset: Dataset,
    model: torch.nn.Module,
    optimizer_args: dict,
    loss_fn: Callable,
    epochs: int,
    eval_interval: int,
    tr_batch_size: int,
    val_batch_size: int,
    backend: str,
    print_every = None,
):
    """
    Training TransformerLM with with same `model_args` using DDP-style gradient sync and lazy data loading.
    
    Important: Only supports GPU training with NCCL backend for now.
    
    Args:
        - rank: the rank of the current process (0 to world_size-1)
        - world_size: total number of processes participating in the training
        
        - context_length: the context length for the language model
        - model_args: a dict of arguments to initialize the TransformerLM
        - optimizer_args: a dict of arguments to initialize the AdamW optimizer
        - loss_fn: the loss function to use for training (e.g. cross_entropy)
        - epochs: total number of training epochs
        - eval_interval: how many epochs to wait before running evaluation on the validation set
        - tr_batch_size: batch size for training
        - val_batch_size: batch size for validation
        - backend: the distributed backend to use (e.g. "nccl" for GPU training)
        - print_every: Mianly for inspecting gradient and parameter values. If None, only print at eval intervals.
    """
    set_dist_env(rank, world_size, backend=backend)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        raise NotImplementedError("This function is only implemented for GPU training with NCCL backend.")

    print(f"Rank {rank} initializing dataset and dataloader...")
    
    # Training Data Loader & DDP Sampler
    tr_sampler = DistributedSampler(
        dataset=trDataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    tr_loader = DataLoader(
        dataset=trDataset,
        batch_size=tr_batch_size,
        sampler=tr_sampler,
        shuffle=False,
        num_workers=2,
    )

    # Validation Data Loader & DDP Sampler
    val_sampler = DistributedSampler(
        dataset=valDataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=valDataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
    )

    # Move model to device (cuda:rank)
    print(f"Rank {rank} initializing model and optimizer...")
    model = model.to(device)

    # Boardcast the initial model parameters from rank 0 to other ranks
    print(f"Rank {rank} broadcasting initial model parameters...")
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # Init optimizer
    optimizer = AdamW(model.parameters(), **optimizer_args)

    # Timing logs
    total_avg_comm_time = 0.0
    total_avg_epoch_time = 0.0

    model.train()
    for epoch in range(epochs):
        # Set up DDP sampler, ensuring no data leaks across ranks
        print(f"Rank {rank} starting epoch {epoch + 1}/{epochs}")
        tr_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        acc_loss = 0.0
        comm_time_epoch = 0.0

        # Start timing
        _synchronize_if_cuda(device)
        t0 = time.perf_counter()

        # Train on local batch
        i = 0
        for bat_X, bat_y_ref in tqdm.tqdm(tr_loader, desc=f"Rank {rank} Epoch {epoch + 1}", unit="batch"):
            # Move to device
            bat_X = bat_X.to(device, non_blocking=True)
            bat_y_ref = bat_y_ref.to(device, non_blocking=True)

            # Compute local gradients
            optimizer.zero_grad()
            bat_y_pred = model(bat_X)
            loss = loss_fn(bat_y_pred, bat_y_ref)
            loss.backward()
            
            # All-reduce gradients sync, summing + avg
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

            # Step optimizer after gradient sync
            optimizer.step()    
            acc_loss += loss.item()

            i += 1
            if i == 100:  # TODO: Remove it
                break
        
        # Log Timing Info
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

        # Debugging logs
        if print_every is not None and i < 100:
            print(f"Inspect parameters sample (rank {rank}): {model.parameters().__next__()[0, :5].tolist()}")
            print(f"Inspecting gradient sample (rank {rank}): {[p.grad[0, :5].tolist() for p in model.parameters() if p.grad is not None][0]}")
        
        # Evaluation on validation set every eval_interval epochs
        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_bat_X, val_bat_y_ref in val_loader:
                    val_bat_X = val_bat_X.to(device, non_blocking=True)
                    val_bat_y_ref = val_bat_y_ref.to(device, non_blocking=True)
                    val_bat_y_pred = model(val_bat_X)
                    val_loss += loss_fn(val_bat_y_pred, val_bat_y_ref).item()

            # Average validation loss across all batches and ranks.
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
            # Print eval info
            print(
                f"Rank {rank} | "
                f"Epoch {epoch + 1:04d} | Avg comm: {avg_comm_epoch:.4f}s | "
                f"Avg total: {avg_epoch_time:.4f}s | Local loss sum: {acc_loss:.4f} | Val loss: {avg_val_loss:.4f}"
            )
            print(f"Inspect parameters sample (rank {rank}): {model.parameters().__next__()[0, :5].tolist()}")

        # ---- Epoch End ----

    # Finished Training, print final timing results averaged across ranks.
    if rank == 0:
        print("\n=== Final Timing Results (avg across ranks) ===")
        print(f"Average communication time per epoch: {total_avg_comm_time / epochs:.4f} seconds")
        print(f"Average total time per epoch: {total_avg_epoch_time / epochs:.4f} seconds")

    dist.barrier()
    destroy_process_group()



