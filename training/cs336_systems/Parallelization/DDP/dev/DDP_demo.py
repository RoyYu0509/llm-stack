import time
from typing import List
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
import os
import argparse

class LinearModel(nn.Module):
    """A simple linear model"""
    def __init__(self, in_dim, out_dim, device, dtype, default_init = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim).to(device=device, dtype=dtype)
        if default_init:
            self.init_linear()
    

    def forward(self, x):
        return self.linear(x)

    def init_linear(self):
        """Linear model parameters initialization"""
        for m in self.modules(): # Returns all parameters
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


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
        data:tuple[torch.Tensor, torch.Tensor],
        model: nn.Module, 
        loss_fn: nn.Module,
        lr: float, epochs: int,
        backend: str
    ):
    """1. Setting up process group"""
    set_dist_env(rank, world_size, backend=backend)
    DEVICE = f"cuda:{rank}" if backend == "nccl" else "cpu"

    # Shard the data batch
    B_N, B_D = data[0].shape
    sharded_batch_size = B_N // world_size
    # Shard the data chunk
    B_start = rank * sharded_batch_size
    B_end = (rank + 1) * sharded_batch_size
    # Build the dataloader
    X_loc, y_loc = data[0][B_start:B_end, :], data[1][B_start:B_end]
    X_loc = X_loc.to(DEVICE)
    y_loc = y_loc.to(DEVICE)
    loc_loader = DataLoader(TensorDataset(X_loc, y_loc), batch_size=32, shuffle=True)
    print(f"Rank {rank} processing data from {B_start} to {B_end}")

    # Training Loop
    optim = torch.optim.AdamW(model.parameters(), lr = lr)
    model.to(DEVICE)
    model.train()
    for epoch in range(epochs):
        for bat_X, bat_y_ref in loc_loader:
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

def get_random_dataset(N:int, D:int, device:str, dtype=torch.float32):
    X = torch.randn((N,D), device=device, dtype=dtype) * 238.32134
    y = torch.randn((N,1), device=device, dtype=dtype)
    return X, y

def main(backend: str):
    DEVICE = "cuda" if backend == "nccl" else "cpu"
    DTYPE = torch.float32
    X, y = get_random_dataset(10000, 10, device=DEVICE, dtype=DTYPE)
    data = (X, y)

    epochs = 10

    model = LinearModel(
        10, 1, 
        device=DEVICE, dtype=DTYPE,
        default_init=True
    )

    lr = 0.00001
    loss_fn = nn.MSELoss()

    world_size = args.cpu_nproc if backend == "gloo" else torch.cuda.device_count()

    mp.spawn(
        fn = naive_DDP,
        args = (world_size, data, model, loss_fn, lr, epochs, backend),
        nprocs = world_size,
        join = True
    )

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', action='store_true', help='use GPU if available')
    parser.add_argument('--cpu_nproc', type=int, help='number of CPU processes to spawn', default=1)
    args = parser.parse_args()
    backend = "nccl" if args.GPU and torch.cuda.is_available() else "gloo"

    main(backend)