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
import os
import argparse


DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

ATTN_KERNELS = [
    ("CompTorch", vectorized_attn_torch_fn),
    ("Naive Attention", scaled_dot_product_attention),
    ("MyTriton", flash_attention_my_triton),
]



def DDP_lm_trainer():
    """Training TransofrmerLM with DDP and different attention kernels"""