from DDP_batch import naive_DDP
from cs336_basics.lm import TransformerLM
from cs336_basics.train.data_loader import data_loading
from cs336_basics.train.loss import cross_entropy
from cs336_basics.train.optimizer import AdamW, grad_clip, lr_scheduler
from cs336_basics.transfromer.scaled_dot_prod_attention import (
    flash_attention_my_triton,
    vectorized_attn_torch_fn,
    scaled_dot_product_attention,
)
import time
from typing import List
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import argparse

from cs336_systems.Attention_profiling.profile_lm_kernels import _load_np_tokens
import numpy as np

DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class LazyTokenDataset(torch.utils.data.Dataset):
    """
    Lazy-loading dataset that loads data from disk on-demand.
    Each process only loads the data it needs, when it needs it.
    """
    def __init__(self, data_path: str, context_length: int, device: str = "cpu"):
        self.data_path = data_path
        self.context_length = context_length
        self.device = device
        
        # Load just the metadata (file size) without loading actual data
        self.tokens = np.load(data_path, mmap_mode='r')  # Memory-mapped, doesn't load into RAM
        self.total_tokens = len(self.tokens)
        
        # Calculate number of valid samples
        # Each sample needs context_length + 1 tokens (input + target)
        self.num_samples = max(0, self.total_tokens - context_length)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Load only the specific tokens needed for this sample.
        This is called on-demand by the DataLoader.
        """
        # Load only context_length + 1 tokens from disk
        start_idx = idx
        end_idx = idx + self.context_length + 1
        
        # Memory-mapped array allows efficient slicing without loading full file
        chunk = self.tokens[start_idx:end_idx]
        
        # Convert to tensor
        chunk_tensor = torch.from_numpy(np.array(chunk)).long()
        
        # Split into input and target
        inputs = chunk_tensor[:-1]  # First context_length tokens
        targets = chunk_tensor[1:]  # Next context_length tokens (shifted by 1)
        
        return inputs, targets[:-1]  # Return (context_length,) for both

ATTN_KERNELS = [
    ("CompTorch", vectorized_attn_torch_fn),
    ("Naive Attention", scaled_dot_product_attention),
    ("MyTriton", flash_attention_my_triton),
]

def set_dist_env(rank, world_size, backend="gloo"):
    # Set the master node address and port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Initialize the the proc - `rank` into the group
    init_process_group(
        backend=backend, 
        rank=rank, 
        world_size=world_size
    )



def naive_DDP(
        rank: int, world_size: int, 
        data_path: str,
        context_length: int,
        model: nn.Module, 
        loss_fn: nn.Module,
        lr: float, epochs: int,
        batch_size: int,
        backend: str
    ):
    """DDP training with lazy data loading - each GPU only loads its portion"""
    set_dist_env(rank, world_size, backend=backend)
    DEVICE = f"cuda:{rank}" if backend == "nccl" else "cpu"

    # Create lazy-loading dataset (doesn't load data into memory yet)
    # All processes reference the same file, but DistributedSampler ensures
    # each process only loads different samples
    dataset = LazyTokenDataset(data_path, context_length, device=DEVICE)
    
    # Use DistributedSampler to partition data across ranks
    # Each rank will get different indices, and LazyTokenDataset will only
    # load those specific samples from disk when __getitem__ is called
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,  # Shuffle within each epoch
        drop_last=False
    )
    
    # Create DataLoader with the sampler
    # Data is loaded lazily: only when batches are requested during training
    # NOTE: Don't use shuffle=True in DataLoader when using DistributedSampler
    loc_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # Can increase for parallel data loading
        pin_memory=(backend == "nccl")
    )
    print(f"Rank {rank} will process {len(sampler)} samples (lazy-loaded from disk)")

    # Training Loop
    optim = torch.optim.AdamW(model.parameters(), lr = lr)
    model.to(DEVICE)
    model.train()
    for epoch in range(epochs):
        # Set epoch for sampler to ensure different shuffling each epoch
        sampler.set_epoch(epoch)
        
        for bat_X, bat_y_ref in loc_loader:
            # Data is lazy-loaded here - only these specific batches are read from disk!
            # Move data to device (if not already there)
            bat_X = bat_X.to(DEVICE, non_blocking=True)
            bat_y_ref = bat_y_ref.to(DEVICE, non_blocking=True)
            # Forward
            bat_y_pred = model(bat_X)
            # Compute
            loss = loss_fn(bat_y_pred, bat_y_ref)

            # Backward
            optim.zero_grad()
            loss.backward()

            # Perform All-reduce on the gradients, equivalently reduce() + broadcast() / all_reduce(.)
            for para in model.parameters():
                # Reduce -> Rank 0
                dist.reduce(para.grad, dst=0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    para.grad /= world_size

                # Broadcast -> All ranks
                dist.broadcast(para.grad, src = 0)
                
            # Step the optimizer
            optim.step()

        # Inspect parameters
        print(f" Epoch: {epoch} | Rank {rank} | parameters: {list(model.parameters())[0]}")
    dist.barrier()
    destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Profile LM with different attention kernels")

    # DDP parameters
    parser.add_argument('--GPU', action='store_true', help='use GPU if available')
    parser.add_argument('--cpu_nproc', type=int, help='number of CPU processes to spawn', default=1)

    # User-defined profiling parameters
    parser.add_argument("--WARM_UP_ITER", type=int, required=True)
    parser.add_argument("--PROFILE_ITER", type=int, required=True)
    parser.add_argument("--TRAIN_PATH", type=str, required=True)
    parser.add_argument("--VAL_PATH", type=str, required=True)
    parser.add_argument("--DEVICE", type=str, default="cpu")
    parser.add_argument("--DTYPE", type=str, default="float32", choices=list(DTYPE_DICT.keys()))
    parser.add_argument("--ATTN_KERNEL", type=str, default="Naive Attention", choices=[name for name, _ in ATTN_KERNELS])
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
    attention_fn = dict(ATTN_KERNELS)[args.ATTN_KERNEL]
    backend = "nccl" if args.GPU and torch.cuda.is_available() else "gloo"
    world_size = args.cpu_nproc if backend == "gloo" else torch.cuda.device_count()

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

    # Don't load training data here - let each process load lazily!
    # Just pass the path to the data file
    
    # Define training parameters
    loss_fn = cross_entropy
    lr = args.LR
    epochs = args.WARM_UP_ITER + args.PROFILE_ITER  # Or however many epochs you want
    batch_size = args.TR_BAT_SIZE
    
    # Train LM using DDP with lazy data loading
    # Each GPU process will load only its portion of data from args.TRAIN_PATH
    mp.spawn(
        fn = naive_DDP,
        args = (
            world_size, 
            args.TRAIN_PATH,  # Pass file path, not loaded data
            args.CONTEXT_LENGTH,
            model, 
            loss_fn, 
            lr, 
            epochs, 
            batch_size, 
            backend
        ),
        nprocs = world_size,
        join = True
    )


