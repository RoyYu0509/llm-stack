import argparse
import glob
import itertools
import math
import os
import time
from datetime import timedelta
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import tqdm
import wandb

from cs336_basics.train.checkpointing import save_checkpoint_and_log, save_checkpoint, load_checkpoint
from cs336_basics.lm import TransformerLM
from cs336_basics.train.loss import cross_entropy
from cs336_basics.train.optimizer import AdamW, grad_clip, lr_scheduler
from cs336_basics.transfromer.scaled_dot_prod_attention import (
    flash_attention_my_triton,
    scaled_dot_product_attention,
    vectorized_attention_torch,
    ATTENTION_KERNEL_REGISTRY,
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

def set_dist_env(
    rank: int, world_size: int, backend: str = "gloo", timeout_minutes: int = 60
) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    # NCCL's default collective timeout is 10 minutes. Any rank-0-only work that sits
    # between two collectives (a multi-GB checkpoint write, W&B teardown) stalls the
    # other ranks inside dist.barrier() and, past the timeout, the watchdog SIGABRTs
    # the whole job -- which is exactly how the first long calibration run died.
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
    )


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prune_old_checkpoints(checkpoint_dir: str, tag: str, keep_last: int) -> None:
    """Delete all but the `keep_last` newest ``checkpoint_{tag}_*.pt`` files.

    Each checkpoint here is model + AdamW state (~2.3GB at 190M params) and the
    rented instance only has ~10GB free, so an unattended multi-hour run that
    checkpoints periodically will fill the disk and die unless old ones are pruned.
    Pruning is per-tag so that "step" rotation never eats the final "epoch" file.
    """
    existing = sorted(glob.glob(os.path.join(checkpoint_dir, f"checkpoint_{tag}_*.pt")))
    for stale in existing[:-keep_last]:
        try:
            os.remove(stale)
            print(f"[Checkpoint] pruned old checkpoint {stale}")
        except OSError as e:
            print(f"[Checkpoint] WARNING: could not prune {stale}: {e}")


def save_ddp_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: str,
    rank: int,
    wandb_run = None,
    tag: str = "epoch",
    keep_last: int = 2,
    global_step: int | None = None,
    epoch_index: int | None = None,
) -> None:
    """
    Save a DDP training checkpoint. Only rank 0 performs the actual save to
    avoid file-system races. All ranks synchronize via a barrier afterwards
    so that no rank races ahead before the file is fully written.

    The checkpoint stores:
        - model state dict (unwrapped from DDPOverlapBucketed via .module)
        - optimizer state dict
        - epoch number (the epoch that just completed)

    Args:
        model:  The DDPOverlapBucketed wrapper (or any wrapper with .module).
        optimizer: The optimizer being used for training.
        epoch: The 1-based epoch number that just finished.
        checkpoint_dir: Directory in which to write checkpoint files.
        rank: Current process rank.
    """
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{tag}_{epoch:04d}.pt")
        raw_model = model.module if hasattr(model, "module") else model
        # Unwrap torch.compile too, so checkpoints are written with plain module keys and
        # stay loadable by anything that builds a bare TransformerLM (the sampler, the
        # serving adapter). Otherwise every key carries an "_orig_mod." prefix.
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        # Deliberately save_checkpoint (local only), NOT save_checkpoint_and_log:
        # uploading each ~2.3GB checkpoint as a W&B artifact parks rank 0 inside a
        # blocking upload while rank 1 sits in the barrier below, which is what blew
        # the NCCL watchdog and SIGABRT'd the 2026-07-27 calibration run. Checkpoints
        # are rsync'd off the instance instead.
        # Written directly rather than via save_checkpoint(): that helper's schema has a
        # single "iter" field, but this function is called with a *step* count when
        # tag="step" and an *epoch* count when tag="epoch". One ambiguous field cannot
        # drive a correct resume -- see load_ddp_checkpoint for what that cost. "iter" is
        # still written so old checkpoints and cs336_basics' load_checkpoint keep working.
        torch.save(
            {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": epoch,
                "global_step": global_step,
                "epoch": epoch_index,
            },
            ckpt_path,
        )
        print(f"[Checkpoint] Rank {rank} saved checkpoint to {ckpt_path} "
              f"(global_step={global_step}, epoch={epoch_index})")
        if keep_last is not None and keep_last > 0:
            _prune_old_checkpoints(checkpoint_dir, tag, keep_last)
    dist.barrier()  # Ensure all ranks wait until checkpoint is saved before proceeding

def load_ddp_checkpoint(
    resume_from: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """
    Load a previously saved DDP checkpoint and restore model / optimizer
    states. Every rank loads the same file independently (the model weights
    are identical across ranks after broadcast, so this is safe).

    Args:
        resume_from: Path to the checkpoint ``.pt`` file.
        model: The DDPOverlapBucketed wrapper (or any wrapper with .module).
        optimizer: The optimizer to restore state into.
        device: The device to map tensors onto (``cuda:<rank>``).

    Returns:
        The epoch number stored in the checkpoint (training resumes from
        ``epoch + 1``).
    """
    checkpoint = torch.load(resume_from, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    # Checkpoints written by a compile=True run carry torch.compile's "_orig_mod." key
    # prefix (OptimizedModule wraps the real module). Strip it so such a checkpoint loads
    # into either a compiled or an uncompiled model; a no-op on uncompiled checkpoints.
    state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
    raw_model.load_state_dict(state)
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Prefer the unambiguous fields. Older checkpoints only have "iter", which was written
    # with a *step* count for tag="step" files and an *epoch* count for tag="epoch" files;
    # feeding either into `range(start_epoch, epochs)` produced an EMPTY range
    # (range(15000, 100) and range(100, 100) alike), so resuming silently trained for zero
    # steps and exited looking like a successful run. global_step was never restored at
    # all, so even a non-empty range would have restarted the LR warmup from scratch.
    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")
    if epoch is None or global_step is None:
        raise RuntimeError(
            f"{resume_from} predates the checkpoint schema fix and records only "
            f"iter={checkpoint.get('iter')}, which is ambiguous between a step count and "
            f"an epoch count -- resuming from it cannot be done correctly. Use --init_from "
            f"to start a fresh run from its weights instead."
        )
    print(f"[Checkpoint] Loaded {resume_from}: resuming at epoch {epoch}, global_step {global_step}")
    return epoch, global_step


def parallel_train(
    rank: int,
    parallel_wrapper: torch.nn.Module,
    world_size: int,
    trDataset: Dataset,
    valDataset: Dataset,
    model_kwargs: dict,
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
    checkpoint_dir = None,
    checkpoint_interval = None,
    resume_from = None,
    compile: bool = False,
    seed: int = 0,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    max_iters: int | None = None,
    checkpoint_every_n_steps: int | None = None,
    eval_every_n_steps: int | None = None,
    log_every_n_steps: int | None = None,
    grad_clip_max_norm: float | None = None,
    val_bat_num: int | None = None,
    warmup_iters: int | None = None,
    init_from: str | None = None,
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
        - epochs: total number of training epochs (upper bound; training also stops early if max_iters is hit)
        - eval_interval: how many epochs to wait before running evaluation on the validation set
        - tr_batch_size: batch size for training
        - val_batch_size: batch size for validation
        - backend: the distributed backend to use (e.g. "nccl" for GPU training)
        - print_every: Mianly for inspecting gradient and parameter values. If None, only print at eval intervals.
        - checkpoint_dir: directory to save training checkpoints (None disables saving).
        - checkpoint_interval: save a checkpoint every N epochs.
        - resume_from: path to a .pt checkpoint file to resume training from.
        - compile: if True, torch.compile the model for kernel fusion.
        - seed: random seed for reproducibility.
        - wandb_project: Weights & Biases project name (only rank 0 logs). None disables W&B.
        - wandb_run_name: optional W&B run name override.
        - max_iters: optional total (per-rank) training-step budget across all epochs combined.
          None means run every batch of every epoch with no step cap.
        - checkpoint_every_n_steps: optional step-granularity safety checkpoint, independent of
          checkpoint_interval's epoch granularity. Useful when max_iters caps training well
          short of one full epoch, so at least a few intermediate checkpoints exist for a
          long-running job. None disables step-based checkpointing.
        - eval_every_n_steps: optional step-granularity validation pass, independent of
          eval_interval's epoch granularity. Same motivation as checkpoint_every_n_steps.
        - log_every_n_steps: optional step-granularity train-loss W&B log point, so a run that
          never completes a full epoch still produces a real loss curve, not a single point.
        - grad_clip_max_norm: optional global-norm gradient clipping threshold, applied after
          finish_gradient_synchromnization() (so every rank clips the same, already-synced
          gradients) and before optimizer.step(). None disables clipping.
        - val_bat_num: caps each evaluation pass to this many batches instead of iterating the
          entire (sliding-window, so potentially huge) val_loader. Matters a lot once
          eval_every_n_steps triggers eval frequently -- an unbounded full pass over a
          multi-hundred-thousand-batch val_loader can single-handedly take hours. None means
          no cap (iterate the whole val_loader, matching the original behavior).
        - warmup_iters: if set (together with max_iters), applies cosine-with-linear-warmup
          LR scheduling (reusing cs336_basics.train.optimizer.lr_scheduler) instead of the
          fixed LR this loop previously used unconditionally. This loop has no LR schedule
          at all otherwise, which is a real risk for a from-scratch model at this scale --
          jumping straight to the full LR at step 0 is a well-known cause of early training
          divergence (NaN loss), which is exactly what was observed in this project's own
          calibration run without warmup. None keeps the old fixed-LR behavior.
    """
    # Seed for reproducibility
    torch.manual_seed(seed + rank)

    set_dist_env(rank, world_size, backend=backend)

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        # Ampere (3090) TF32 matmuls: ~2x the fp32 matmul throughput at 10-bit mantissa
        # with fp32 accumulation. The linear layers dominate this model's FLOPs, so this
        # is the single cheapest way to buy more training tokens inside a fixed wall-clock
        # budget; it is the standard setting for fp32 training on Ampere.
        # CS336_DISABLE_TF32=1 turns it off, so the speedup can be measured rather than
        # asserted (see the TF32 arm of the benchmark suite).
        use_tf32 = os.environ.get("CS336_DISABLE_TF32", "0") != "1"
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        if rank == 0:
            print(f"TF32 matmul/cudnn: {'enabled' if use_tf32 else 'DISABLED'}")
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
    model = TransformerLM(**model_kwargs)
    model = model.to(device)

    # init_from vs resume_from: resume_from continues an interrupted run (weights +
    # optimizer moments + step counter). init_from starts a NEW run from pretrained
    # weights only -- fresh AdamW state and a fresh LR schedule -- which is what
    # fine-tuning wants: the pretraining run ended with its cosine decayed to the
    # minimum and its moments tuned to that regime, and inheriting either would fight
    # the fine-tune's own (much smaller) schedule.
    if init_from is not None:
        ckpt = torch.load(init_from, map_location=device)
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        print(f"Rank {rank} initialized weights from {init_from} "
              f"(pretraining iter {ckpt.get('iter')}); optimizer state NOT restored.")

    print(f"Rank {rank} broadcasting initial model parameters...")
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model.train()
    model = parallel_wrapper(model, bucket_size_mb=bucket_size_mb)
    # Compute model size
    para_num_billion = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Rank {rank} model initialized with {para_num_billion:.2f}B parameters.")

    # Compile the model if flagged
    if compile:
        print(f"Rank {rank} compiling model with torch.compile (inductor backend)...")
        model.module = torch.compile(model.module, backend="inductor")

    # Initialize W&B logging on rank 0 only
    wandb_run = None
    if wandb_project is not None and rank == 0:
        run_name = wandb_run_name or f"flashddp-{para_num_billion:.2f}B_LM-bkt{bucket_size_mb}MB"
        wandb_run = wandb.init(project=wandb_project, name=run_name, config={
            "world_size": world_size,
            "epochs": epochs,
            "tr_batch_size": tr_batch_size,
            "val_batch_size": val_batch_size,
            "bucket_size_mb": bucket_size_mb,
            "compile": compile,
            "seed": seed,
            **optimizer_args,
            **model_kwargs,
        })

    # Timing logs
    total_avg_comm_time = 0.0
    total_avg_epoch_time = 0.0

    # Move model to device (cuda:rank)
    print(f"Rank {rank} initializing optimizer...")

    # Init optimizer
    optimizer = AdamW(model.parameters(), **optimizer_args)

    peak_lr = optimizer_args.get("lr", optimizer.param_groups[0]["lr"])
    use_lr_schedule = warmup_iters is not None and max_iters is not None

    # Resume from checkpoint if provided
    start_epoch = 0
    resume_global_step = 0
    if resume_from is not None:
        start_epoch, resume_global_step = load_ddp_checkpoint(resume_from, model, optimizer, device)
        print(f"Rank {rank} resuming training from epoch {start_epoch + 1}, "
              f"global_step {resume_global_step} (LR schedule continues from there)")

    def _run_eval(step_label: int) -> float:
        """Run a (optionally capped, via val_bat_num) validation pass on every rank
        (all_reduce is collective) and, on rank 0 only, print + log to W&B. Restores
        model.train() before returning."""
        print(f"Rank {rank} starting evaluation on validation set (step {step_label})...")
        model.eval()
        val_loss = 0.0
        n_batches = 0
        val_iter = iter(val_loader)
        if val_bat_num is not None:
            val_iter = itertools.islice(val_iter, val_bat_num)
        with torch.no_grad():
            for val_bat_X, val_bat_y_ref in tqdm.tqdm(
                val_iter, desc=f"Rank {rank} Evaluating (step {step_label})", unit="batch",
                total=val_bat_num,
            ):
                val_bat_X = val_bat_X.to(device, non_blocking=True)
                val_bat_y_ref = val_bat_y_ref.to(device, non_blocking=True)
                val_bat_y_pred = model(val_bat_X)
                val_loss += loss_fn(val_bat_y_pred, val_bat_y_ref).item()
                n_batches += 1

        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / (n_batches * world_size)
        print(f"Rank {rank} | step {step_label:06d} | Val loss: {avg_val_loss:.4f}")

        if wandb_run is not None:
            wandb_run.log({"val/loss": avg_val_loss}, step=step_label)

        model.train()
        return avg_val_loss

    # Continue the step counter across a resume, so the LR schedule picks up where it left
    # off instead of re-running warmup on an already-trained model.
    global_step = resume_global_step
    reached_max_iters = False
    # Defined up front because the final checkpoint save below reads it: if the epoch range
    # is empty (already at `epochs`), the loop variable would otherwise never be bound.
    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        # Set up DDP sampler, ensuring no data leaks across ranks
        print(f"Rank {rank} starting epoch {epoch + 1}/{epochs}")
        tr_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        acc_loss = 0.0

        # # Start timing
        # _synchronize_if_cuda(device)
        # t0 = time.perf_counter()

        # Train on local batch
        i = 0
        for bat_X, bat_y_ref in tqdm.tqdm(tr_loader, desc=f"Rank {rank} Epoch {epoch + 1}", unit="batch"):
            # Move to device
            bat_X = bat_X.to(device, non_blocking=True)
            bat_y_ref = bat_y_ref.to(device, non_blocking=True)

            # preserve the param.grad -> grad_buffer view links established by DDPOverlapBucketed. 
            optimizer.zero_grad(set_to_none=False)
            bat_y_pred = model(bat_X)
            loss = loss_fn(bat_y_pred, bat_y_ref)
            # Backward is synchronous, it return after all para.grad are updated; thus, async ops are queued.
            loss.backward()
            
            # wait for grad tensor sync
            model.finish_gradient_synchromnization()

            if grad_clip_max_norm is not None:
                grad_clip(list(model.parameters()), grad_clip_max_norm)

            if use_lr_schedule:
                current_lr = lr_scheduler(
                    it=global_step,
                    max_learning_rate=peak_lr,
                    min_learning_rate=peak_lr * 0.1,
                    warmup_iters=warmup_iters,
                    cosine_cycle_aiters=max_iters,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

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
            loss_value = loss.item()
            acc_loss += loss_value

            i += 1
            global_step += 1

            # Fail fast on divergence. Without this the job happily burns its whole
            # wall-clock budget multiplying NaNs, and nothing in stdout says so until the
            # next eval -- which can be thousands of steps away.
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"Rank {rank}: training DIVERGED -- non-finite loss ({loss_value}) at "
                    f"global_step {global_step} (lr={current_lr if use_lr_schedule else peak_lr:.3e}). "
                    f"Aborting instead of burning the rest of the budget."
                )

            if log_every_n_steps is not None and global_step % log_every_n_steps == 0:
                # Also to stdout, not just W&B: the run log is the only thing available when
                # reconnecting over SSH, and previously it showed nothing but a tqdm bar.
                lr_str = f" | lr {current_lr:.3e}" if use_lr_schedule else ""
                print(f"Rank {rank} | step {global_step:06d} | train loss {loss_value:.4f}{lr_str}", flush=True)
                if wandb_run is not None:
                    log_payload = {"train/loss": loss_value, "epoch": epoch + 1}
                    if use_lr_schedule:
                        log_payload["train/lr"] = current_lr
                    wandb_run.log(log_payload, step=global_step)

            if (
                checkpoint_dir is not None
                and checkpoint_every_n_steps is not None
                and global_step % checkpoint_every_n_steps == 0
            ):
                save_ddp_checkpoint(model, optimizer, global_step, checkpoint_dir, rank, wandb_run,
                                    tag="step", global_step=global_step, epoch_index=epoch)

            if eval_every_n_steps is not None and global_step % eval_every_n_steps == 0:
                _run_eval(global_step)

            if max_iters is not None and global_step >= max_iters:
                reached_max_iters = True
                break

        # W&B: log average training loss for the epoch (best-effort step marker; may repeat
        # global_step if the epoch ended exactly on a log_every_n_steps boundary, which W&B
        # tolerates as a duplicate point rather than an error).
        avg_train_loss = acc_loss / max(i, 1)
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch + 1, "train/epoch_avg_loss": avg_train_loss}, step=global_step)

        # # Log Timing Info
        # _synchronize_if_cuda(device)
        # t3 = time.perf_counter()
        # epoch_time = t3 - t0

        # if time_warmup_ep is not None and epoch < time_warmup_ep:
        #     timing = torch.tensor([epoch_time], device=device)
        #     dist.all_reduce(timing, op=dist.ReduceOp.SUM)
        #     timing /= world_size

        #     avg_epoch_time = timing[0].item()

        #     total_avg_epoch_time += avg_epoch_time

        # Evaluation on validation set every eval_interval epochs
        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            avg_val_loss = _run_eval(global_step)
            print(f"Rank {rank} | Epoch {epoch + 1:04d} | Local loss sum: {acc_loss:.4f} | Val loss: {avg_val_loss:.4f}")
            print(f"Inspect parameters sample (rank {rank}): {model.parameters().__next__()[0, :5].tolist()}")

        # ---- Checkpoint Save ----
        if (
            checkpoint_dir is not None
            and checkpoint_interval is not None
            and (epoch + 1) % checkpoint_interval == 0
        ):
            save_ddp_checkpoint(
                model,
                optimizer,
                epoch + 1,  # Store the epoch that just finished
                checkpoint_dir,
                rank,
                wandb_run,
                global_step=global_step,
                epoch_index=epoch + 1,
            )

        # ---- Epoch End ----
        if reached_max_iters:
            print(f"Rank {rank} reached max_iters={max_iters} at epoch {epoch + 1}, stopping early.")
            break

    # One last validation pass so the run always ends with a final val number, even when
    # it stopped on max_iters rather than on an eval_every_n_steps boundary.
    if eval_every_n_steps is not None or eval_interval > 0:
        final_val_loss = _run_eval(global_step)
        print(f"Rank {rank} | FINAL | global_step {global_step} | Val loss: {final_val_loss:.4f}")

    # Save a final checkpoint at the end of training
    if checkpoint_dir is not None:
        save_ddp_checkpoint(model, optimizer, epochs, checkpoint_dir, rank, wandb_run,
                            global_step=global_step, epoch_index=epoch + 1)

    # Finished Training, print final timing results averaged across ranks.
    if rank == 0:
        print("\n=== Final Timing Results (avg across ranks) ===")
        print(f"Average communication time per epoch: {total_avg_comm_time / epochs:.4f} seconds")
        print(f"Average total time per epoch: {total_avg_epoch_time / epochs:.4f} seconds")

    # Barrier BEFORE the W&B teardown, not after: wandb_run.finish() only exists on rank 0
    # and can block for a long time flushing, so leaving a collective behind it means the
    # other ranks burn their NCCL timeout waiting on rank 0's network I/O.
    dist.barrier()
    if wandb_run is not None:
        wandb_run.finish()

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

    # Checkpointing
    parser.add_argument("--CHECKPOINT_DIR", type=str, default=None,
                        help="Directory to save training checkpoints. If None, no checkpoints are saved.")
    parser.add_argument("--CHECKPOINT_INTERVAL", type=int, default=None,
                        help="Save a checkpoint every N epochs. Requires --CHECKPOINT_DIR.")
    parser.add_argument("--RESUME_FROM", type=str, default=None,
                        help="Path to a .pt checkpoint file to resume training from.")
    parser.add_argument("--COMPILE", action="store_true", default=False,
                        help="Compile the model with torch.compile for kernel fusion.")
    parser.add_argument("--SEED", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--WANDB_PROJECT", type=str, default=None,
                        help="Weights & Biases project name. None disables W&B logging.")
    parser.add_argument("--WANDB_RUN_NAME", type=str, default=None,
                        help="Optional W&B run name override.")
    
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
            args.CHECKPOINT_DIR,
            args.CHECKPOINT_INTERVAL,
            args.RESUME_FROM,
            args.COMPILE,
            args.SEED,
            args.WANDB_PROJECT,
            args.WANDB_RUN_NAME,
        ),
        nprocs=world_size,
        join=True,
    )
