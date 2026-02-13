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
import numpy as np
import torch
from torch.utils.data import Dataset
from cs336_basics.train.data_loader import data_loading

# Helper data loader
class TokenStreamDataset(Dataset):
    def __init__(self, path, context_len):
        self.tokens_stream = self._load_np_tokens(path)   # create the lazy data stream
        self.context_len = context_len  # per-sample length defined
        self.N = self.tokens_stream.shape[0]
    
    def __len__(self):
        """
        Return the total number of tokens in the dataset
        The last sample starts at N - context_len - 1.
        """
        return self.N - self.context_len - 1

    def _load_np_tokens(self, path):
        """Create the lazy data stream"""
        arr = np.load(path, mmap_mode="r")
        tensor = torch.from_numpy(arr).long()
        return tensor
    
    def __getitem__(self, idx):
        """Get the idx-th sample"""
        # Safe check for max context length < data stream length
        max_start = self.N - self.context_len - 1
        if max_start < 0:
            raise RuntimeError("context_length is too large for the provided data.")
        # Indexing
        x = self.tokens_stream[idx : idx + self.context_len]
        y = self.tokens_stream[idx + 1 : idx + 1 + self.context_len]
        # Data type conversion
        if x.dtype != torch.long:
            x = x.long().contiguous()
            y = y.long().contiguous()
        return x, y

        

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


def naive_LLM_DDP(
        rank: int, world_size: int, 
        data_path: str,
        context_length: int,
        model_args: dict,
        optimizer_args: dict,
        loss_fn: nn.Module,
        epochs: int,
        batch_size: int,
        backend: str
    ):
    """DDP training with lazy data loading - each GPU only loads its portion"""
    set_dist_env(rank, world_size, backend=backend)
    torch.cuda.set_device(rank)
    DEVICE = f"cuda:{rank}" if backend == "nccl" else "cpu"

    # Create lazy-loading dataset (doesn't load data into memory yet)
    # All processes reference the same file, but DistributedSampler ensures
    # each process only loads different samples
    dataset = TokenStreamDataset(data_path, context_length)
    
    # Use DistributedSampler to partition data across ranks
    # Each rank will get different indices, and TokenStreamDataset will only
    # load those specific samples from disk when __getitem__ is called
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    # Create DataLoader with the sampler
    # Data is loaded lazily: only when batches are requested during training
    # NOTE: Don't use shuffle=True in DataLoader when using DistributedSampler
    loc_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )
    print(f"Rank {rank} will process {len(sampler)} samples (lazy-loaded from disk)")
    
    # Create model and optimizer
    model = TransformerLM(**model_args)
    # Boardcast model parameters from rank 0 to all other ranks
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), **optimizer_args)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Shuffle data differently each epoch
        acc_loss = 0.0
        for bat_X, bat_y_ref in loc_loader:
            bat_X = bat_X.to(DEVICE)
            bat_y_ref = bat_y_ref.to(DEVICE)
            # Clear last gradients
            optimizer.zero_grad()
            # Forward
            bat_y_pred = model(bat_X)
            # Compute
            loss = loss_fn(bat_y_pred, bat_y_ref)
            # Backward
            loss.backward()

            # Parallelism: Perform All-reduce on the gradients on each parameter
            for para in model.parameters():
                # Continue if no need to update
                if para.grad is None: 
                    continue
                # Reduce -> Rank 0
                dist.reduce(para.grad, dst=0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    para.grad /= world_size
                # Broadcast -> All ranks
                dist.broadcast(para.grad, src = 0)
            
            # Accumulate loss for monitoring
            acc_loss += loss.item()

            # Step optimizer
            optimizer.step()

        # Inspect parameters
        print(f" Epoch: {epoch} | Rank {rank} | parameters: {list(model.parameters())[0]} | Training Loss: {acc_loss:.4f}")
    dist.barrier()
    destroy_process_group()



def main():
    parser = argparse.ArgumentParser(description="Training TransformerLM with FlashAttn, CUDA DDP, and lazy data loading")

    # Enforce CUDA availability
    assert torch.cuda.is_available(), "CUDA is required for this script."

    # User-defined profiling parameters
    parser.add_argument("--EPOCHES", type=int, default=3)
    parser.add_argument("--TRAIN_PATH", type=str, required=True)
    parser.add_argument("--VAL_PATH", type=str, required=True)
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
    backend = "nccl"
    world_size = torch.cuda.device_count()

    # Initialize model and optimizer args
    model_kwargs = dict(
        vocab_size=args.VOCAB_SIZE,
        context_length=args.CONTEXT_LENGTH,
        num_layers=args.NUM_LAYERS,
        d_model=args.D_MODEL,
        num_heads=args.NUM_HEADS,
        d_ff=args.D_FF,
        rope_theta=args.ROPE_THETA,
        device="cpu",     # Moved to CUDA after DDP init on each rank   
        dtype=args.DTYPE,
        attention_fn=attention_fn,
    )

    optim_kwargs = dict(
        lr=args.LR,
        weight_decay=args.WEIGHT_DECAY,
        betas=(args.BETA1, args.BETA2),
        eps=args.ADAM_EPS,
    )

    loss_fn = cross_entropy
    batch_size = args.TR_BAT_SIZE
    
    # Train LM using DDP with lazy data loading
    # Each GPU process will load only its portion of data from args.TRAIN_PATH
    mp.spawn(
        fn=naive_LLM_DDP,
        args=(world_size, args.TRAIN_PATH, args.CONTEXT_LENGTH, model_kwargs, optim_kwargs, loss_fn, args.EPOCHES, batch_size, backend),
        nprocs=world_size,
        join=True,
    )

