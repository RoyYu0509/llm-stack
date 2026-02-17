import argparse
import os
import time
from typing import Callable, List, Tuple, Dict
from xml.parsers.expat import model

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import tqdm

from cs336_basics.lm import TransformerLM
from cs336_basics.train.loss import cross_entropy
from cs336_basics.train.optimizer import AdamW

def log_dist(msg):
    if os.environ.get("DEBUG_DDP") == "1":
        rank = dist.get_rank() if dist.is_initialized() else -1
        print(f"[Rank {rank}] {msg}")

def _get_tensor_size(tensor):
    """Return the size of the `tensor`."""
    return tensor.numel() * tensor.element_size() / 1e6  # Size in MB


class Bucket:
    """
    The communicative tensor bucket object.
    """
    def __init__(self, bucket_size_mb: float, bound: List[int]):
        self.bkt_size_mb = bucket_size_mb
        self._initialize_bkt_size_mb = 0.0
        self.para_tensor_dict = {}
        self._initialize_full = False
        self._full_para_set = set()
        self._updated_grad_para= set()
        self.bound = bound
        self.para_list = []
    
    def add_tensor(self, para):
        """
        Create a dictionary: {para: para.grad}
        Set self.full = True if `self.bkt_size_mb == self.curr_bkt_size_mb`.
        """
        # Add the parameter tensor to the bucket dict
        self.para_tensor_dict[para] = para.grad
        self._initialize_bkt_size_mb += _get_tensor_size(para.grad)
        self._full_para_set.add(para)
        self.para_list.append(para)
        # Check if the bucket is full
        if self._initialize_bkt_size_mb >= self.bkt_size_mb:
            self._initialize_full = True

    def is_full(self):
        return self._initialize_full
    
    def grad_ready(self):
        """
        Check if all the gradient tensors in the bucket are ready (i.e., have been computed).
        Return True if all gradients are ready, False otherwise.
        """
        return self._updated_grad_para == self._full_para_set
    
    def record_updates(self, para):
        """
        Add para into the `self._updated_grad_para` set, to record that the para has been updated
        """
        self._updated_grad_para.add(para)

    def yeild_flatten_grad(self):
        # Dynamically retrieve the CURRENT gradients from the keys (parameters)
        # We iterate over the keys (parameters) of the dictionary to get the latest .grad attribute
        grads = [p.grad for p in self.para_list]
        flatten_grad = parameters_to_vector(grads)
        return flatten_grad


class DDPOverlapBucketed(nn.Module):
    """
    An optimized DDP with Bucketing and Overlapping Communication and Computation.
    """
    def __init__(self, module: torch.nn.Module, bucket_size_mb: int):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, 
        construct a DDP container that will handle gradient synchronization across ranks.
        
        Args:
            - module: an instantiated PyTorch nn.Module, e.g. TransformerLM(model_args)
        """

        self.para_2_bucket: Dict[torch.nn.Parameter, Bucket]
        super().__init__()
        self.module = module
        self.dtype = next(module.parameters()).dtype
        self.device = next(module.parameters()).device
        self.hooks = []
        self.comm_work_handles = []

        # A list of buckets
        self.bkt_sz_mb = bucket_size_mb
        self.buckets = []
        self.para_2_bucket = {}

        # Keep a reference to the model parameters in the same order as the buckets, 
        # for correct flattening/unflattening (since the bucket boundaries are determined by the order of parameters).
        self._reversed_paras = []

        # Initialize the bucket 
        self._build_bucket()
        # Add the parameter hooks for overlapping communication and computation
        self.add_bucketing_hooks()
    
    def _build_bucket(self):
        """Build the parameters bucket dict."""
        # Initialize an empty bucket
        temp_bucket = Bucket(self.bkt_sz_mb, [0,0])
        start = 0
        end = 0

        # Build the bucket in reverse order
        for i, para in enumerate(reversed(list(self.module.parameters()))):
            # Initialize gradient vector only for `requires_grad == True` parameters
            if para.requires_grad:
                # Initialize gradient tensor
                para.grad = torch.zeros_like(para, dtype=self.dtype, device=self.device)

                # Record the order of bucketed parameters
                self._reversed_paras.append(para)
            
                # Register the para to the bucket
                temp_bucket.add_tensor(para)
                self.para_2_bucket[para] = temp_bucket
                
                prev_end = end
                end += para.numel()
                log_dist(f"Rank {dist.get_rank()} Added {i}-th param (size {para.numel()}) to bucket. Current specific bound in loop: [{prev_end}, {end}]")

                # Yield & New a bucket if is full
                if temp_bucket.is_full():
                    temp_bucket.bound = [start, end] # Update the bucket bound
                    log_dist(f"Rank {dist.get_rank()}, Bucket registered full. Bound: {temp_bucket.bound}")
                    self.buckets.append(temp_bucket) # add to the bucket list
                    # Init a new bucket
                    temp_bucket = Bucket(self.bkt_sz_mb, [start, end])
                    start = end # Update the start of the new bucket as the old end
            
        # Add the last bucket is not empty
        if len(temp_bucket.para_tensor_dict.keys()) > 0:
            temp_bucket.bound = [start, end] # Update the bucket bound
            self.buckets.append(temp_bucket)

        # Initialize a flattened gradient vector for the built buckets
        self.temp_flatten_grad = torch.zeros(end, dtype=self.dtype, device=self.device)
        log_dist(f"Rank {dist.get_rank()}, Finished building buckets. Total bucketss: {len(self.buckets)}")
        
    def forward(self, *inputs, **kwargs):
        """
        A standard self.module forward function.

        TODO: Implement parallelized forward. (Tensor parallelism, pipeline parallelism, etc.)
        """
        return self.module(*inputs, **kwargs)

    def sync_grad(self, bucket:Bucket):
        """Sync the gradient bucket"""
        # Assert the bucket is ready for synchronization
        assert bucket.grad_ready(), "The gradient is not ready"        
        
        # Get corresponding boundary in the flattened gradient vector for this bucket
        s,e = bucket.bound
        log_dist(f"Syncing bucket with bound [{s}, {e}] on rank {dist.get_rank()} with {bucket.yeild_flatten_grad().shape} grad tensor.")
        # Yield a flattened gradient vector from the bucket and replace the part in the full model's flat grad 
        self.temp_flatten_grad[s:e] = bucket.yeild_flatten_grad()
        log_dist(f"Syncing bucket bound [{s}, {e}]")
        # Inplace all-reduce the full model's flat grad 
        work = dist.all_reduce(self.temp_flatten_grad[s:e], op=dist.ReduceOp.SUM, async_op=True)
        self.comm_work_handles.append(work)

    def _register_grad_ready(self, para):
        """register the parameter as recieved gradient in its bucket"""
        log_dist(f"Parameter of size {para.numel()} ready/accumulated.")
        bucket = self.para_2_bucket[para]
        bucket.record_updates(para)
        if bucket.grad_ready():
            log_dist("Bucket full & ready. Triggering sync.")
            self.sync_grad(bucket)

    def add_bucketing_hooks(self):
        """
        Add Overlapping Bucketed DDP Hooks to `self.module.parameters()`
        """
        # Iterate and add post-hooks
        for para in self.module.parameters():
            if para.requires_grad:
                # Hook Fn 1: Record the para as received gradient in its bucket
                #            and sync the gradient if the bucket is ready
                hook = para.register_post_accumulate_grad_hook(self._register_grad_ready)
                self.hooks.append(hook)

    def finish_gradient_synchromnization(self):
        """
        All communication work handles have been queued.
        """
        # If no communication work was queued this step, do not overwrite
        # parameter gradients with the (possibly stale/zero) flattened buffer.
        # if len(self.comm_work_handles) == 0:
        #     log_dist("No queued all-reduce work handles in this step; skipping unflatten.")
        #     for bucket in self.buckets:
        #         bucket._updated_grad_para.clear()
        #     return

        # Wait for all comm to finish
        for handle in self.comm_work_handles:
            handle.wait()
        self.comm_work_handles.clear()

        # Normalize the gradients
        if dist.is_initialized():
             self.temp_flatten_grad /= dist.get_world_size()

        # After allreduce updates, unflatten the gradient vector into every para.grad
        self.unflatten_grads()

        # Clear the bucket records
        for bucket in self.buckets:
            bucket._updated_grad_para.clear()

    def unflatten_grads(self):
        """
        Unflatten the gradient vector into the original parameter shapes 
        after all-reduce the model's full flattened gradient vector.
        """
        original_grad = [p.grad for p in self._reversed_paras]
        vector_to_parameters(self.temp_flatten_grad, original_grad)