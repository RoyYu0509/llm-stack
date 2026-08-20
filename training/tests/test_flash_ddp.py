import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cs336_systems.Parallelization.FlashDDP.FlashDDP import DDPOverlapBucketed

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping NCCL test")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class MediumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def check_consistency(model, rank, world_size, iteration):
    # Check 1: Gradients should be synchronized and equal across ranks
    for name, param in model.module.named_parameters():
        if param.grad is None:
            continue
            
        grad = param.grad.data.clone()
        
        # Verify grad is not Zero (unless it happens to be naturally) - simple sanity check
        # assert torch.count_nonzero(grad) > 0, f"Gradient for {name} is zero/None on rank {rank}"
        
        # All-gather gradients to check equality
        grad_list = [torch.zeros_like(grad) for _ in range(world_size)]
        dist.all_gather(grad_list, grad)
        
        reference_grad = grad_list[0]
        for i, other_grad in enumerate(grad_list[1:]):
            assert torch.allclose(reference_grad, other_grad, atol=1e-5), \
                f"Iter {iteration}: Gradients for {name} mismatch between rank 0 and rank {i+1}"

    # Check 2: Parameters should stay synchronized across ranks
    for name, param in model.module.named_parameters():
        data = param.data.clone()
        data_list = [torch.zeros_like(data) for _ in range(world_size)]
        dist.all_gather(data_list, data)
        
        reference_data = data_list[0]
        for i, other_data in enumerate(data_list[1:]):
            if not torch.allclose(reference_data, other_data, atol=1e-5):
                print(f"Rank {rank}, Iter {iteration}, Parameter {name} mismatch")
                # print(f"Rank {rank}, Parameter {name}, Data: {data}")
                # print(f"Rank {rank}, Parameter {name}, Other Data: {other_data}")
            assert torch.allclose(reference_data, other_data, atol=1e-5), \
                f"Iter {iteration}: Parameters {name} mismatch between rank 0 and rank {i+1}"

def run_ddp_check(rank, world_size, model_cls=MediumModel, bucket_size_mb=0.1, iterations=10):
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Set seed for reproducibility of initialization
    torch.manual_seed(42)
    model = model_cls().to(device)
    
    # Wrap model in DDP
    ddp_model = DDPOverlapBucketed(model, bucket_size_mb)
    
    if hasattr(ddp_model, 'add_bucketing_hooks'):
        ddp_model.add_bucketing_hooks()
    
    optimizer = torch.optim.SGD(ddp_model.module.parameters(), lr=0.01)
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    for i in range(iterations):
        # Input data - different for each rank to ensure all-reduce is actually doing work
        torch.manual_seed(rank + i * 100)
        # MediumModel expects input (batch, 128)
        input_data = torch.randn(16, 128).to(device)
        # Target for CrossEntropy (batch)
        target = torch.randint(0, 10, (16,)).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = ddp_model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Trigger synchronization explicitly
        if hasattr(ddp_model, 'finish_gradient_synchromnization'):
            ddp_model.finish_gradient_synchromnization()
            
        # Check consistency (gradients and params)
        check_consistency(ddp_model, rank, world_size, i)
        
        # Optimizer step
        optimizer.step()
        
        # Check params consistency after step
        check_consistency(ddp_model, rank, world_size, i)

    cleanup()

def test_ddp_medium_model_10_iters():
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} GPUs to run DDP test")
    mp.spawn(run_ddp_check,
             args=(world_size, MediumModel, 0.1, 10),
             nprocs=world_size,
             join=True)

def test_ddp_small_bucket_stress():
    # Force many small buckets
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} GPUs to run DDP test")
    mp.spawn(run_ddp_check,
             args=(world_size, MediumModel, 0.001, 5),
             nprocs=world_size,
             join=True)

def test_ddp_large_bucket_single():
    # Force single large bucket
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires at least {world_size} GPUs to run DDP test")
    mp.spawn(run_ddp_check,
             args=(world_size, MediumModel, 50.0, 5),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    test_ddp_overlap_bucketed()
