import argparse
import os
import time
from typing import Callable

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

from cs336_systems.Parallelization.DDP.stream_dataset import TokenStreamDataset
from cs336_systems.Parallelization.FlashDDP.FlashDDP import DDPOverlapBucketed

def set_dist_env(rank: int, world_size: int, backend: str = "gloo") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)



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
    bucket_size_mb = 1,
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
    model = parallel_wrapper(model, bucket_size_mb=bucket_size_mb)
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
            model.finish_gradient_synchromnization()

            # # Inspect gradients
            # if print_every is not None and (i + 1) % print_every == 0:
            #     j = 0
            #     for name, param in model.module.named_parameters():
            #         if param.grad is not None:
            #             print(f"Rank {rank}, Parameter {name}, Grad Sample: {param.grad.view(-1)[:5]}")
            #             j += 1
            #         if j == 5:  # Print at most 5 parameters' gradients
            #             break

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
    dist.destroy_process_group()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training TransformerLM with manual DDP-style gradient sync and lazy data loading"
    )

    assert torch.cuda.is_available(), "CUDA is required for this script."

    # Data and Training Hyperparameters
    parser.add_argument("--EPOCHES", type=int, default=5)
    parser.add_argument("--WARMUP_EPOCHS", type=int, default=2)
    parser.add_argument("--EVAL_INTERVAL", type=int, default=100)
    parser.add_argument("--TR_BAT_SIZE", type=int, default=8)
    parser.add_argument("--VAL_BAT_SIZE", type=int, default=8)
    parser.add_argument("--TRAIN_PATH", type=str, required=True)
    parser.add_argument("--VAL_PATH", type=str, required=True)
    parser.add_argument("--DTYPE", type=str, default="float32", choices=list(DTYPE_DICT.keys()))
    parser.add_argument("--BUCKET_SIZE_MB", type=int, default=1)
    
    # Model Hyperparameters
    parser.add_argument("--CONTEXT_LENGTH", type=int, default=256)
    parser.add_argument("--PRINT_EVERY", type=int, default=1)
    parser.add_argument("--VOCAB_SIZE", type=int, default=10000)
    parser.add_argument("--ROPE_THETA", type=float, default=10_000.0)
    parser.add_argument("--NUM_LAYERS", type=int, default=16)
    parser.add_argument("--D_MODEL", type=int, default=256)
    parser.add_argument("--NUM_HEADS", type=int, default=8)
    parser.add_argument("--D_FF", type=int, default=256*4)
    parser.add_argument( "--ATTN_KERNEL", type=str, default="Naive Attention",
        choices=[name for name, _ in ATTN_KERNELS],
    )
    parser.add_argument("--AUTOCAST", action="store_true", default=False)

    # Optimizer Hyperparameters
    parser.add_argument("--LR", type=float, default=3e-4)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01)
    parser.add_argument("--BETA1", type=float, default=0.9)
    parser.add_argument("--BETA2", type=float, default=0.999)
    parser.add_argument("--ADAM_EPS", type=float, default=1e-8)

    args = parser.parse_args()

    # Set the Pre-DDP configs
    dtype = DTYPE_DICT[args.DTYPE]
    attention_fn = dict(ATTN_KERNELS)[args.ATTN_KERNEL]
    backend = "nccl"
    world_size = torch.cuda.device_count()

    # Load the same initial model 
    model_kwargs = dict(
        vocab_size=args.VOCAB_SIZE,
        context_length=args.CONTEXT_LENGTH,
        num_layers=args.NUM_LAYERS,
        d_model=args.D_MODEL,
        heads_num=args.NUM_HEADS,
        d_ff=args.D_FF,
        theta=args.ROPE_THETA,
        device="cpu",
        dtype=dtype,
        attention_fn=attention_fn,
    )
    model = TransformerLM(**model_kwargs)

    # Same optim configs
    optim_kwargs = dict(
        lr=args.LR,
        weight_decay=args.WEIGHT_DECAY,
        betas=(args.BETA1, args.BETA2),
        eps=args.ADAM_EPS,
    )

    # Load Data Set
    val_dataset = TokenStreamDataset(args.VAL_PATH, args.CONTEXT_LENGTH)
    tr_dataset = TokenStreamDataset(args.TRAIN_PATH, args.CONTEXT_LENGTH)

    # Load the same model to all ranks and start DDP training
    mp.spawn(
        fn=parallel_train,
        args=(
            DDPOverlapBucketed,
            world_size,
            tr_dataset,
            val_dataset,
            model,
            optim_kwargs,
            cross_entropy,
            args.EPOCHES,
            args.EVAL_INTERVAL,
            args.TR_BAT_SIZE,
            args.VAL_BAT_SIZE,
            backend,
            args.PRINT_EVERY,
            args.WARMUP_EPOCHS,
            args.BUCKET_SIZE_MB,
        ),
        nprocs=world_size,
        join=True,
    )

