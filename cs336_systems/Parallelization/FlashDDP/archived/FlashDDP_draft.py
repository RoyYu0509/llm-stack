import argparse
import os
import time
from typing import Callable
from xml.parsers.expat import model

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import parameters_to_vector, vector_to_parameters
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



class flashDDPModule(torch.nn.Module):
    """
    An optimized DDP with Bucketing and Overlapping Communication and Computation.
    """
    def __init__(self, module: torch.nn.Module, buket_size: int, rank: int, world_size: int):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, 
        construct a DDP container that will handle gradient synchronization across ranks.
        
        Args:
            - module: an instantiated PyTorch nn.Module, e.g. TransformerLM(model_args)
        """
        super().__init__()
        self.module = module
        self.rank = rank
        self.world_size = world_size
        self.hooks = []
        self.comm_work_handles = []
        self.bucket_size = buket_size
        self.para_2_size_reverse = self._build_bucket()
        self.acc_bucket_mb = 0
        self._safe = torch.zeros(world_size, device=torch.device(f"cuda:{rank}"), dtype=torch.int32)
        self.bucket = []
        self._start_sync = False
    
    def _build_bucket(self):
        """Build the parameters bucket list."""
        para_2_size_reverse = {}
        for para in reversed(list(self.module.parameters())):
            para_2_size_reverse[para] = para.numel() * para.element_size()

        return para_2_size_reverse
    
    def forward(self, *inputs, **kwargs):
        """
        A standard self.module forward function.

        TODO: Implement parallelized forward. (Tensor parallelism, pipeline parallelism, etc.)
        """
        return self.module(*inputs, **kwargs)

    def _board_cast_params(self):
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def call_and_register_handle(self, collective_fn, tensor):
        """
        Perform the collective_fn AND register the collective handle
        """
        handle = collective_fn(tensor)
        self.comm_work_handles.append(handle)
        return handle

    def _hook_async_all_reduce(self):
        """
        Hook Fn: Issue an async all-reduce communication when the bucket is full.
        
        Precondition:
            - all ranks' buckets are full and ready to sync
            - self.bucket has been materialized to a flatten tensor
        """
        if self._start_sync:
            work = dist.all_reduce(self.bucket, op=dist.ReduceOp.AVG, async_op=True)
            self.comm_work_handles.append(work)
            return self.bucket
    
    def _hook_add_to_bucket(self, grad_tensor):
        """Hook Fn: Add the tensor to the bucket when recieved grad"""
        print("Add parameter to bucket, size (MB): ", grad_tensor.numel() * grad_tensor.element_size() / 1e6)
        self.bucket.append(grad_tensor)
        self.acc_bucket_mb += grad_tensor.numel() * grad_tensor.element_size() / 1e6
        
        # Check if self rank bucket size == threshold
        if self.acc_bucket_mb >= self.bucket_size:
            # Record the current rank bucket as safe
            self._safe[self.rank] = 1
            
            # Wait until all ranks are safe
            while not sum(self._safe) == self.world_size:
                print(f"Rank {self.rank} waiting...")
                self._hook_try_flatten_grads()
            
            # All ranks are safe, start the all-reduce communication with the bucket
            print(f"Rank {self.rank} starting all-reduce with bucket size (MB): {self.acc_bucket_mb:.2f}")

    
    def _hook_try_flatten_grads(self,):
        """
        Check if all ranks are safe. 
        If all ranks safe, mature self.bucket to the actual bucket tensor.
        """
        if sum(self._safe) < self.world_size:
            return
        else:
            # All ranks'buckets is ready
            # Replace the bucket with the actual flatten tensor
            self.bucket = torch._utils._flatten_dense_tensors(self.bucket)
            self._start_sync = True
            print(f"Rank {self.rank} bucket is full, start all-reduce with size (MB): {self.bucket.numel() * self.bucket.element_size() / 1e6}")
        
    def add_grad_allredu_hook(self):
        """
        Add a post-accumulate hook to each parameter's gradient tensor, 
        so that after the local gradient is computed in each backward pass, 
        it will automatically trigger an all-reduce communication to sync gradients across ranks.


        Hooks:
            1. Check if the bucket of this parameter is ready
            2. If ready, fatten all tensor gradient and issue a all-reduce comm with overlapping.
        """
        # Check model device
        assert next(self.module.parameters()).device == torch.device(f"cuda:{self.rank}"), "Model parameters must be on CUDA."
        # Iterate and add post-hooks
        for para in self.bucket:
            if para.requires_grad:
                # Hook Fn 1: Record safe when recieved the gradient and add to bucket
                hook = para.register_post_accumulate_grad_hook(self._hook_add_to_bucket)
                # Hook Fn 2: All-reduce when all ranks' bucket are ready.
                hook = para.register_post_accumulate_grad_hook(self._hook_async_all_reduce)
                self.hooks.append(hook)

    def wait_on_queue(self):
        """
        Wait for all communications to finish. Mainly used for timing the communication time.
        """
        for handle in self.comm_work_handles:
            handle.wait()

        self.comm_work_handles.clear()



def parallel_train(
    rank: int,
    parallel_wrapper: torch.nn.Module,
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
    time_warmup_ep = None,
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

    # Boardcast the initial model parameters from rank 0 to other ranks
    model = model.to(device)
    print(f"Rank {rank} broadcasting initial model parameters...")
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model.train()
    model = DDPOverlapBucketed
    # Timing logs
    total_avg_comm_time = 0.0
    total_avg_epoch_time = 0.0

    # Move model to device (cuda:rank)
    print(f"Rank {rank} initializing optimizer...")

    # Init optimizer
    optimizer = AdamW(model.parameters(), **optimizer_args)
    for epoch in range(epochs):
        # Set up DDP sampler, ensuring no data leaks across ranks
        print(f"Rank {rank} starting epoch {epoch + 1}/{epochs}")
        tr_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        acc_loss = 0.0

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
            # Backward is synchronous, it return after all para.grad are updated; thus, async ops are queued.
            loss.backward()
            
            # wait for grad tensor sync
            model.wait_on_queue()

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

        if time_warmup_ep is not None and epoch < time_warmup_ep:
            timing = torch.tensor([epoch_time], device=device)
            dist.all_reduce(timing, op=dist.ReduceOp.SUM)
            timing /= world_size

            avg_epoch_time = timing[0].item()

            total_avg_epoch_time += avg_epoch_time

        # Debugging logs
        if print_every is not None and i < 100:
            print(f"Inspect parameters sample (rank {rank}): {model.parameters().__next__()[0, :5].tolist()}")
            print(f"Inspecting gradient sample (rank {rank}): {[p.grad[0, :5].tolist() for p in model.parameters() if p.grad is not None][0]}")
        
        # Evaluation on validation set every eval_interval epochs
        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            print(f"Rank {rank} starting evaluation on validation set...")
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_bat_X, val_bat_y_ref in tqdm.tqdm(val_loader, desc=f"Rank {rank} Evaluating Epoch {epoch + 1}", unit="batch"):
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
                f"Epoch {epoch + 1:04d}"
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

    