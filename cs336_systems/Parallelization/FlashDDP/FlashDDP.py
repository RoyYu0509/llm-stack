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

        - _full_para_set: the set of bucketed parameters registered in the bucket (for quick lookup)
        - _updated_grad_para_set: the set of parameters that have received gradients (for checking if the bucket is ready for synchronization)
        
        - para_list: the list of bucketed parameters (for correct flattening/unflattening order)
    """
    def __init__(self, bucket_size_mb: float, offsets: int):
        self.offsets = offsets
        self.numel = 0
        self.bkt_size_mb = bucket_size_mb
        self._initialize_bkt_size_mb = 0.0
        self._initialize_full = False
        self._full_para_set = set()
        self._updated_grad_para_set = set()
        self.para_list = []
        self.grad_buffer = None
        # Index of this bucket in DDPOverlapBucketed.buckets (set during _build_bucket)
        self.idx: int = -1
        # Set to True by the hook once all grads have arrived; consumed by the scheduler
        self.ready: bool = False
    
    def add_tensor(self, para):
        """
        Register para into the bucket:
            1. Update current bucket size in MB
            2. Add para into the bucket's parameter set (for quick lookup)
            3. Add para into the bucket's parameter list (for correct flattening order)
            4. Update self.numel the number of parameter elements in the bucket (for global tensor slicing)
        
        Update the full flag if the bucket size exceeds.
        """

        self._initialize_bkt_size_mb += _get_tensor_size(para)
        self._full_para_set.add(para)
        self.para_list.append(para)
        self.numel += para.numel()
        # Check if the bucket is full
        if self._initialize_bkt_size_mb >= self.bkt_size_mb:
            # Set the bucket flag to full
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
        Record the parameter `para` as having received its gradient, by adding it to the updated_grad_para_set.
        """
        self._updated_grad_para_set.add(para)

    def clear_grad_buffer(self):
        """
        Clear the gradient buffer of this bucket to 0.
        """
        if self.grad_buffer is not None:
            self.grad_buffer.zero_()

    def _prepare_grad_buffer(self, dtype: torch.dtype, device: torch.device):
        """
        Allocate a contiguous flat grad_buffer tensor for this bucket's parameters.

        Linking each parameter's .grad attribute as a sliced view into it.

        Autograd will then accumulate para.grad directly into the corresponding slice of the bucket's grad_buffer.
        """
        # Prepare contiguous grad buffer
        self.grad_buffer = torch.zeros(self.numel, dtype=dtype, device=device)
        offsets = 0
        # Iterate over registered parameters & link their para.grad as views into the grad_buffer
        for para in self.para_list:
            n = para.numel()
            # Link the para.grad as a view, so that the actual bwd grad flows into the buffer slice
            para.grad = self.grad_buffer[offsets : offsets + n].view_as(para)
            offsets += n
    
    def finalize(self, dtype: torch.dtype, device: torch.device):
        """
        Finalize the bucket after initialization by pre-allocate a grad buffer for the registered parameters.

        Links each parameter's .grad attribute as a sliced view into the bucket's grad_buffer, so that the 
        actual bwd grad flows into the buffer slice, which will be synchronized in-place during training.
        """
        self._prepare_grad_buffer(dtype, device)




class DDPOverlapBucketed(nn.Module):
    """
    An optimized DDP training wrapper with Bucketing and Overlapping Communication and Computation. 

    Precondition: 
        1. Processes must have been initialized with `nccl` backend before creating a wrapper instance.
        2. In outer training loop, optimizer.zero_grad(set_to_none=False) MUST be used. 
           Otherwise, the view links between param.grad and the bucket grad_buffers will be broken by set param.grad = None,
           breaks the in-place accumulation and synchronization in the next iteration, causing incorrect training.
    
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
        - _reversed_paras: kept for reference but not required for unflattening (grads are views into bucket grad_buffers)
    
    Optimization:
        Before: 
            1. Create many bucket-level flat gradient buffers, AND one model-level flat gradient buffer. 
            2. Each synchronization:
                1). Create a dummy flatten the model-level gradients zero tensor.
                2). Copy the bucket-level flatten gradients into the slices of the model-level flatten gradient buffer.
                3). All-reduce the model level gradient tensor slice [start:end].
                4). Unflatten the model-level flatten gradient buffer and replace the model parameters' .grad with the unflattened views.

        Now:
            1. We don't need a whole model-level flat gradient buffer for synchronization. Instead, we set link grad to bucket-level grad_buffers,
            and directly synchronize the bucket-level gradients using the grad_buffers, which is linked directly to the parameter gradients.

            Therefore, the gradients sync into parameters' .grad views in-place with no extra copy, without need to flatten(.) & unflatten(.) at model level.
            
            Each sync:
                1). All-reduce the bucket-level flatten gradient buffer directly, which is already linked to the parameter gradients as views.
                    No unflatten step needed: each param's .grad is already a view into its bucket's grad_buffer, which has been all-reduced in-place.

    We can do this optimization because 
        ```
        g = grad_buffer[offset:offset+n].view_as(param)
        param.grad = g
        ```

    NOTE: optimizer.zero_grad(set_to_none=False) MUST be used. set_to_none=True sets param.grad = None,
    which breaks the param.grad → grad_buffer view links so the next backward accumulates into a fresh
    independent tensor, making all subsequent all_reduce calls operate on stale/zero buffers.
    """
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float = 5.0):
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

        # A local full parameters list in the same order as the buckets.
        self._reversed_paras: List[torch.nn.Parameter] = []
        # Points to the next bucket index that must be all-reduced next.
        # Ensures all ranks issue collectives in the same order (0, 1, 2, …),
        # preventing NCCL deadlocks from out-of-order ready-time firing.
        self.next_bucket_to_reduce: int = 0

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
            # Only bucket the parameters that require gradients
            if para.requires_grad:
                # Record the order of bucketed parameters
                self._reversed_paras.append(para)

                # Register the para to the bucket
                temp_bucket.add_tensor(para)
                self.para_2_bucket[para] = temp_bucket  # Record the para to bucket mapping for quick lookup

                # Finalize & replace the bucket if it is full
                if temp_bucket.is_full():
                    # Allocate grad_buffer and assign param grad views for this bucket
                    temp_bucket.finalize(self.dtype, self.device)
                    # Record the bucket index
                    temp_bucket.idx = len(self.buckets)
                    # Add to the bucket list
                    self.buckets.append(temp_bucket)
                    # Init a new bucket
                    offsets += temp_bucket.numel  # Update the offsets for the new bucket
                    temp_bucket = Bucket(self.bkt_sz_mb, offsets=offsets)

        # Finalize and add the last bucket if it has any parameter
        if len(temp_bucket._full_para_set) > 0:
            # SAME AS ABOVE
            temp_bucket.finalize(self.dtype, self.device)
            temp_bucket.idx = len(self.buckets)
            self.buckets.append(temp_bucket)

        log_dist(f"Rank {dist.get_rank()}, Finished building buckets. Total buckets: {len(self.buckets)}")
        
    def forward(self, *inputs, **kwargs):
        """
        A standard self.module forward function.

        TODO: Implement parallelized forward. (Tensor parallelism, pipeline parallelism, etc.)
        """
        return self.module(*inputs, **kwargs)

    def _sync_grad_bucket(self, input_bucket: Bucket):
        """
        Sync the gradients of parameters registered in `input_bucket`.

        Because each parameter's .grad is a view into input_bucket.grad_buffer,
        the buffer already holds the locally accumulated gradients with no extra
        copy.  We divide in-place (pre-scale before the sum all-reduce to get
        the mean) and issue an async all-reduce directly on grad_buffer.

        NOTE: The `input_bucket` must be ready, i.e. all paras have received their gradients.
        """
        assert input_bucket.grad_ready(), "The gradient is not ready"

        log_dist(f"Syncing bucket at offset {input_bucket.offsets}, numel {input_bucket.numel}")
        # Average bucket's gradient tensor in-place before All-Reduce
        input_bucket.grad_buffer.div_(dist.get_world_size())
        # All-reduce the bucket's buffer, which is already linked to the parameter gradients through views(.) method.
        work = dist.all_reduce(input_bucket.grad_buffer, op=dist.ReduceOp.SUM, async_op=True)
        self.comm_work_handles.append(work)

    def _schedule_bucket_sync(self):
        """
        Scheduler: launch all-reduce for buckets that are ready AND in-order based on their bucket index.

        Preventing Bucket[1] para's sync hook fn triggered before Bucket[0] is triggered, No out-of-order error from NCCL.
        """
        while (
            self.next_bucket_to_reduce < len(self.buckets) and 
            self.buckets[self.next_bucket_to_reduce].ready
        ):
            self._sync_grad_bucket(self.buckets[self.next_bucket_to_reduce])
            self.next_bucket_to_reduce += 1

    def _hook_register_grad_ready(self, para):
        """
        A post-accumulate grad hook function to be registered on each parameter.

        When recieved the bwd gradient, Effects: 
            1. Record the parameter as having received its gradient in its bucket.
            2. If the bucket is fully ready, mark it ready and invoke the scheduler,
               which launches all_reduce only when it is this bucket's turn.
        """
        log_dist(f"Parameter of size {para.numel()} ready/accumulated.")
        bucket = self.para_2_bucket[para]
        bucket.record_updates(para)
        if bucket.grad_ready():
            log_dist(f"Bucket {bucket.idx} full & ready. Notifying scheduler.")
            bucket.ready = True
            self._schedule_bucket_sync()

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
        # Wait for all comm to finish
        for handle in self.comm_work_handles:
            handle.wait()
        assert self.next_bucket_to_reduce == len(self.buckets), "Not all buckets have been synchronized"
        
        self.comm_work_handles.clear()

        # Reset the bucket states for the next iteration
        self.next_bucket_to_reduce = 0
        for bucket in self.buckets:
            # Bucket set to not ready
            bucket.ready = False
            # Remove the para_grad_has_updated records 
            bucket._updated_grad_para_set.clear()

        