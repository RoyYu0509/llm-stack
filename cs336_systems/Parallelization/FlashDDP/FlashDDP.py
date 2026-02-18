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

    Instance Attributes:
        - offsets:      the starting offset of the bucket in the global flattened gradient vector (for correct slicing)
        - numel:        the number of elements in the bucket (for global tensor slicing)
        - bkt_size_mb:  the fixed bucket size threshold in MB

        - _initialize_bkt_size_mb: accumulated bucket size during initialization (for bucket initialzation)
        - _initialize_full: bool flag to indicate if the bucket is full during initialization (for bucket initialzation)

        - _full_para_set: the set of bucketed parameters (for quick lookup)
        - _updated_grad_para_set: the set of parameters that have received gradients (for checking if the bucket is ready for synchronization)
        
        - para_list: the list of bucketed parameters (for correct flattening/unflattening order)
    """
    def __init__(self, bucket_size_mb: float, offsets:int):
        self.offsets = offsets
        self.numel = 0
        self.bkt_size_mb = bucket_size_mb
        self._initialize_bkt_size_mb = 0.0
        self._initialize_full = False
        self._full_para_set = set()
        self._updated_grad_para_set= set()
        self.para_list = []
    
    def add_tensor(self, para):
        """
        Register para into the bucket:
            1. Update current bucket size in MB
            2. Add para into the bucket's parameter set (for quick lookup)
            3. Add para into the bucket's parameter list (for correct flattening order)
            4. Update self.numel the number of parameter elements in the bucket (for global tensor slicing)
        
        Update the full flag if the bucket size exceeds.
        """
        self._initialize_bkt_size_mb += _get_tensor_size(para.grad)
        self._full_para_set.add(para)
        self.para_list.append(para)
        self.numel += para.numel()
        # Check if the bucket is full
        if self._initialize_bkt_size_mb >= self.bkt_size_mb:
            self._initialize_full = True

    def is_full(self):
        """
        Return `True` if the bucket is full, False otherwise.
        """
        return self._initialize_full
    
    def grad_ready(self):
        """
        Return `True` if all parameters in the bucket have received bwd gradients.
        """
        return self._updated_grad_para_set == self._full_para_set
    
    def record_updates(self, para):
        """
        Add para into the `self._updated_grad_para_set` set, to record that the para has been updated
        """
        self._updated_grad_para_set.add(para)

    def yield_flatten_grad(self):
        """
        Return the flattened gradient vectors of the bucket from self.para_list (SAME ORDER).
        """
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
            - bucket_size_mb: the bucket size threshold in MB for bucketing the model parameters.
        
        Instance Attributes:
            - module: the original nn.Module passed in, used for forward and backward
            - dtype: the data type of the model parameters (for initializing the bucket gradient tensors)
            - device: the device of the model parameters (for initializing the bucket gradient tensors)
            - hooks: a list of registered hook handles for overlapping communication and computation (for cleanup)
            - comm_work_handles: a list of communication work handles for asynchronous communication (for synchronization)
            - buckets: a list of Bucket objects for bucketing the model parameters
            - para_2_bucket: a dictionary mapping each parameter to its corresponding bucket (for quick lookup during gradient synchronization)
            - _reversed_paras: a list of model parameters in the same order as inside the buckets (for correct flattening/unflattening of the write in bucket grad vectors)
            - temp_flatten_grad: a local full flattened gradient vector for the whole model, used for bucketing communication and synchronization   
        """

        
        super().__init__()
        self.module: torch.nn.Module = module
        self.dtype: torch.dtype = next(module.parameters()).dtype
        self.device: torch.device = next(module.parameters()).device
        self.hooks: List[torch.nn.utils.hooks.RemovableHandle] = []
        self.comm_work_handles: List[torch.distributed.ProcessGroup] = []

        # A list of buckets
        self.bkt_sz_mb: int = bucket_size_mb
        self.buckets: List[Bucket] = []
        self.para_2_bucket: Dict[torch.nn.Parameter, Bucket] = {}

        # A local full parameters list in the same order as the buckets for correct flattening/unflattening of the write in bucket grad vectors.
        self._reversed_paras: List[torch.nn.Parameter] = []
        # A local full flattened gradient vector for the whole model, used for bucket grad replacing and synchronization.
        self.temp_flatten_grad: torch.Tensor = None

        # Initialize the bucket 
        self._build_bucket()

        # Add the parameter hooks for overlapping communication and computation
        self._add_grad_ready_hooks()
    
    def _build_bucket(self):
        """
        Build model's parameters buckets for bucketed communication.
        
        """
        # Initialize an empty bucket
        offsets = 0
        temp_bucket = Bucket(self.bkt_sz_mb, offsets=offsets)

        # Build the bucket in reverse order
        for para in reversed(list(self.module.parameters())):
            # Initialize gradient vector only for `requires_grad == True` parameters
            if para.requires_grad:
                # Initialize gradient tensor
                para.grad = torch.zeros_like(para, dtype=self.dtype, device=self.device)

                # Record the order of bucketed parameters
                self._reversed_paras.append(para)
            
                # Register the para to the bucket
                temp_bucket.add_tensor(para)
                self.para_2_bucket[para] = temp_bucket # Record the para to bucket mapping for quick lookup

                # Yield & New a bucket if is full
                if temp_bucket.is_full():
                    # Add the filled bucket to the buckets list
                    self.buckets.append(temp_bucket) 
                    # Init a new bucket
                    offsets += temp_bucket.numel # Update the offsets for the new bucket
                    temp_bucket = Bucket(self.bkt_sz_mb, offsets=offsets)
            
        # Add the last bucket if it has any parameter
        if len(temp_bucket.para_tensor_dict.keys()) > 0:
            self.buckets.append(temp_bucket)

        # Initialize a global flattened grads vector
        self.temp_flatten_grad = torch.zeros(offsets, dtype=self.dtype, device=self.device)
        log_dist(f"Rank {dist.get_rank()}, Finished building buckets. Total bucketss: {len(self.buckets)}")
        
    def forward(self, *inputs, **kwargs):
        """
        A standard self.module forward function.

        TODO: Implement parallelized forward. (Tensor parallelism, pipeline parallelism, etc.)
        """
        return self.module(*inputs, **kwargs)

    def _sync_grad(self, input_bucket:Bucket):
        """
        Sync the gradient of parameters rigstered in the `input_bucket` by:
            1. Yield the partial flattened gradient vector of the bucket from `input_bucket.yeild`
            2. Replace the full flattened gradient vector with the yielded bucket gradient vector
            3. Inplace all-reduce the correpsonding slice of the flattened gradient vector across ranks.

        NOTE: The `input_bucket` must be ready, ie. all paras have received their gradients.
        """
        # Assert the bucket is ready for synchronization
        assert input_bucket.grad_ready(), "The gradient is not ready"        
        
        # Get corresponding boundary in the flattened gradient vector for this bucket
        s,e = input_bucket.bound
        # Yield a flattened gradient vector from the bucket, normalize it, and replace the part in the full model's flat grad 
        self.temp_flatten_grad[s:e] = input_bucket.yeild_flatten_grad() / dist.get_world_size() 
        log_dist(f"Syncing bucket bound [{s}, {e}]")
        # Inplace all-reduce the full model's flat grad 
        work = dist.all_reduce(self.temp_flatten_grad[s:e], op=dist.ReduceOp.SUM, async_op=True)
        self.comm_work_handles.append(work)

    def _hook_register_grad_ready(self, para):
        """
        A post-accumulate grad hook function to be registered on each parameter.

        When recieved the bwd gradient, Effects: 
            1. Modify the bucket, update the bucket's para as received gradient.
            2. If the bucket is ready after this, trigger the synchronization of the bucket para's gradients.
        """
        log_dist(f"Parameter of size {para.numel()} ready/accumulated.")
        bucket = self.para_2_bucket[para]
        bucket.record_updates(para)
        if bucket.grad_ready():
            log_dist("Bucket full & ready. Triggering sync.")
            self._sync_grad(bucket)

    def _add_grad_ready_hooks(self):
        """
        Register the post-accumulate grad hook on each parameter, enable the overlapping sync and comp.
        """
        # Iterate and add post-hooks
        for para in self.module.parameters():
            if para.requires_grad:
                # Hook Fn 1: Record the para as received gradient in its bucket
                #            and sync the gradient if the bucket is ready
                hook = para.register_post_accumulate_grad_hook(self._hook_register_grad_ready)
                self.hooks.append(hook)

    def finish_gradient_synchromnization(self):
        """
        A barrier Fn to wait for synchronization to finish.

        Procedures:
            1. Wait for all communication work to finish (if any).
        """
        # If no communication work was queued this step, do not overwrite
        # parameter gradients with the (possibly stale/zero) flattened buffer.
        # if len(self.comm_work_handles) == 0:
        #     log_dist("No queued all-reduce work handles in this step; skipping unflatten.")
        #     for bucket in self.buckets:
        #         bucket._updated_grad_para_set.clear()
        #     return

        # Wait for all comm to finish
        for handle in self.comm_work_handles:
            handle.wait()
        self.comm_work_handles.clear()

        # Unflatten the all-reduced flattened gradient vector back to the original parameters.
        self.unflatten_grads()

        # Clear the bucket records
        for bucket in self.buckets:
            bucket._updated_grad_para_set.clear()

    def unflatten_grads(self):
        """
        Replace the grad attribute of each parameter with the unflattened gradient 
        from the full local flattened gradient vector after synchronization.
        """
        original_grad = [p.grad for p in self._reversed_paras]
        vector_to_parameters(self.temp_flatten_grad, original_grad)