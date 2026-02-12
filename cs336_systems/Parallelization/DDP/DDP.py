from Linear_model import LinearModel, 
import os
import time
from typing import List
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--GPU', action='store_true', help='use GPU if available')
parser.add_argument('--cpu_nproc', type=int, help='number of CPU processes to spawn', default=1)
args = parser.parse_args()

BACKEND = "nccl" if args.GPU and torch.cuda.is_available() else "gloo"
CPU_NPROC = args.cpu_nproc

def dist_setup(rank, world_size):
    # Set the master node address and port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Initialize the the proc - `rank` into the group
    init_process_group(
        backend=BACKEND, 
        rank=rank, 
        world_size=world_size
    )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        rank: int,
        save_every: int,
        inspect_p: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.rank = rank
        self.save_every = save_every
        self.loss_fn = loss_fn
        self.inspect_p = inspect_p

    def _run_sample(self,input, target):
        """ Fit model on a single sample """
        self.optimizer.zero_grad()
        predictions = self.model(input)
        loss = self.loss_fn(predictions, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _run_batch(self, epoch: int):
        """ Fit model on the entire batch """
        loss_val = []
        for input, target in self.dataloader:
            loss_val.append(self._run_sample(input, target))
        if self.rank == 0:
            print(f"Rank {self.rank} Epoch {epoch} | Batch Loss = {sum(loss_val)/len(loss_val)}")
    
    def train(self, epochs: int):
        for epoch in range(epochs):
            self._run_batch(epoch)
            if epoch % self.save_every == 0:
                md_ckp = self.model.state_dict()
                os.makedirs("./lm_ckp", exist_ok=True)
                torch.save(md_ckp, "./lm_ckp")
                if self.inspect_p:
                    print(f"Rank {self.rank} Epoch {epoch} | Model Weights overview = {self.model.weight[:5,:5]}")

    def compute_local_gradients(self, epoch: int):
        """ Compute local gradients on the current batch, without updating the model """
        for input, target in self.dataloader:


def _naive_DDP_training_iter(
        rank: int, world_size: int,
        model: LinearModel, 
        dataset: tuple[torch.Tensor, torch.Tensor],
        epochs: int, lr: float
    ) -> None:
    """
    A full step model update using batch data parallelization.
    
    Steps:
        1. Shard the batch into n/d
        2. Each process computes local gradients on its shard
        3. All-Reduce the gradients across all processes
        4. Each process updates its local model with the averaged gradients
    """
    # Intialize Proc
    dist_setup(rank, world_size)
    DEVICE = f"cuda:{rank}" if BACKEND == "nccl" else "cpu"
    model.to(device=DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Load the whole dataset on rank 0
    if rank == 0:
        X, y = dataset[0], dataset[1]
        N,D = X.shape[0], X.shape[1]
        B_N = N // world_size
    else:


    # Keep only a batch shard
    for i in range(world_size):
        if rank == i:
            X = X[i*B_N : (i+1)*B_N, :]
            y = y[i*B_N : (i+1)*B_N, :]

    # Shard data, each process keeps its own shard
    N = X.shape[0]
    batch_size = N // world_size
    X, y = X[rank:rank*batch_size, :].to(DEVICE), y[rank:rank*batch_size,:].to(DEVICE)

    # Build Dataloader & Trainer
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    trainer = Trainer(model, optimizer, dataloader, nn.CrossEntropyLoss(), rank, save_every=100)

    # Compute local gradients
    trainer.train(epochs)

    # All-Reduce collections


    # Model updates
    

def get_random_dataset(N:int, D:int, device:str):
    X = torch.randint(0, 1000, (N,D), device=device)
    y = torch.randint(0, 10, (N,), device=device)
    return X, y



def scatter_data(rank: int, world_size: int, N, D):
    """
    Load and Scatter data from rank 0
    """
    # Intialize Proc
    dist_setup(rank, world_size)
    DEVICE = f"cuda:{rank}" if BACKEND == "nccl" else "cpu"

    # Load the data on to rank0
    if rank == 0:
        X, y = get_random_dataset(N, D, device=DEVICE)
    else:
        X, y = None, None
    
    



def main():
    X, y = get_random_dataset(10000, 10, device='cpu')
