import argparse
import os
import time
from typing import Callable
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
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from typing import Callable, List, Tuple, Dict

import tqdm

from cs336_basics.lm import TransformerLM
from cs336_basics.train.loss import cross_entropy
from cs336_basics.train.optimizer import AdamW

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
        self._updated_grad_para= set()
        self.bound = bound
    
    def add_tensor(self, para):
        """
        Create a dictionary: {para: para.grad}
        Set self.full = True if `self.bkt_size_mb == self.curr_bkt_size_mb`.
        """
        # Add the parameter tensor to the bucket dict
        self.para_tensor_dict[para] = para.grad
        self._initialize_bkt_size_mb += _get_tensor_size(para.grad)
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
        return self._updated_grad_para == set(self.para_tensor_dict.keys())
    
    def record_updates(self, para):
        """
        Add para into the `self._updated_grad_para` set, to record that the para has been updated
        """
        self._updated_grad_para.add(para)


class DDPOverlapBucketed(nn.Module):
    """
    An optimized DDP with Bucketing and Overlapping Communication and Computation.
    """
    def __init__(self, module: torch.nn.Module, buket_size_mb: int, rank: int, world_size: int):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, 
        construct a DDP container that will handle gradient synchronization across ranks.
        
        Args:
            - module: an instantiated PyTorch nn.Module, e.g. TransformerLM(model_args)
        """

        self.para_2_bucket: Dict[torch.nn.Parameter, Bucket]
        super().__init__()
        self.module = module
        self.rank = rank
        self.world_size = world_size
        self.hooks = []
        self.comm_work_handles = []
        # A list of buckets
        self.bkt_sz_mb = buket_size_mb
        self.buckets = []
        self.para_2_bucket = {}

        # Initialize the bucket 
        self._build_bucket()

        # Create a flatten gradient tensor
        self.flatten_grad = parameters_to_vector(self.module.parameters())
    
    def _build_bucket(self):
        """Build the parameters bucket dict."""
        # Initialize an empty bucket
        temp_bucket = Bucket(self.bkt_sz_mb, [0,0])
        start = 0
        end = 0
        # Iterate over model parameters to create gradient bucket.
        for para in reversed(list(self.module.parameters())):
            # Initialize gradient vector
            para.grad = torch.zeros_like(para, dtype=para.dtype, device=para.device)
            # Register the para to the bucket
            temp_bucket.add_tensor(para)
            self.para_2_bucket[para] = temp_bucket
            end += para.numel()
            # Yield & New a bucket if is full
            if temp_bucket.is_full():
                self.buckets.append(temp_bucket) # add to the bucket list
                temp_bucket = Bucket(self.bkt_sz_mb, [start, end]) # Init a new bucket
                start = end # Update the start of the new bucket as the old end
        
        # Add the last bucket is not empty
        if len(temp_bucket.para_tensor_dict.keys()) > 0:
            self.buckets.append(temp_bucket)

    def forward(self, *inputs, **kwargs):
        """
        A standard self.module forward function.

        TODO: Implement parallelized forward. (Tensor parallelism, pipeline parallelism, etc.)
        """
        return self.module(*inputs, **kwargs)

    def sync_grad(self, bucket:Bucket):
        """Sync the gradient bucket"""
        assert bucket.grad_ready(), "The gradient is not ready"
        s,e = bucket.bound
        work = dist.all_reduce(self.flatten_grad[s:e], op=dist.ReduceOp.SUM, async_op=True)
        self.comm_work_handles.append(work)

    def _register_grad_ready(self, para):
        """register the parameter as recieved gradient in its bucket"""
        bucket = self.para_2_bucket[para]
        bucket.record_updates(para)
        if bucket.grad_ready():
            self.sync_grad(bucket)

    def add_bucketing_hooks(self):
        """
        Add Overlapping Bucketed DDP Hooks to `self.module.parameters()`
        """
        # Check model device
        assert next(self.module.parameters()).device == torch.device(f"cuda:{self.rank}"), "Model parameters must be on CUDA."
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
        # Wait for all comm to finish
        for handle in self.comm_work_handles:
            handle.wait()
        self.comm_work_handles.clear()

        # After allreduce updates, unflatten the gradient vector into every para.grad
        self.unflatten_grads()

    def unflatten_grads(self):
        """Unflatten the gradient vector into the original parameter shapes."""
        param_list = list(self.module.parameters())
        vector_to_parameters(self.flatten_grad, param_list)