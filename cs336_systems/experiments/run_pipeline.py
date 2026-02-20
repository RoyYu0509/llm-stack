#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end CUDA-only LM pipeline.

Stages:
  1. (Optional) Download TinyStories dataset
  2. Train BPE tokenizer
  3. Tokenize text → .npy
  4. Train Transformer LM with selectable attention kernel & DDP wrapper

Usage:
  # Single-GPU, default attention:
  uv run python cs336_systems/experiments/run_pipeline.py --config cs336_systems/experiments/default_pipeline_config.json

  # Override kernel + DDP from CLI:
  uv run python cs336_systems/experiments/run_pipeline.py \
      --config cs336_systems/experiments/default_pipeline_config.json \
      --attention_kernel flash_attention_triton \
      --ddp_wrapper flashddp

  # Skip dataset stages if data already exists:
  uv run python cs336_systems/experiments/run_pipeline.py \
      --config cs336_systems/experiments/default_pipeline_config.json \
      --skip_data
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("Cannot locate repo root (pyproject.toml not found).")


REPO = _repo_root()

VALID_KERNELS = [
    "scaled_dot_prod_attention",
    "vectorized_torch",
    "flash_attention_triton",
]

VALID_DDP = ["none", "naive", "flashddp"]

VALID_DATASET_MODES = ["tinystories", "local_text", "local_npy", "auto"]


def _resolve(rel_path: str) -> Path:
    """Resolve a path relative to repo root."""
    return (REPO / rel_path).resolve()


def _ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.  No CUDA device detected.")


# ---------------------------------------------------------------------------
# Stage 1: Dataset download (TinyStories)
# ---------------------------------------------------------------------------
def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest* using wget or urllib."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest} already exists.")
        return
    print(f"  Downloading {url} → {dest} ...")
    try:
        subprocess.check_call(["wget", "-q", "-O", str(dest), url])
    except FileNotFoundError:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))


def stage_download(cfg: dict) -> None:
    mode = cfg.get("dataset_mode", "tinystories")
    if mode not in ("tinystories", "auto"):
        print(f"[stage:download] dataset_mode={mode}, skipping download.")
        return
    train_url = cfg["tinystories_train_url"]
    valid_url = cfg["tinystories_valid_url"]
    _download_file(train_url, _resolve(cfg["raw_train_text"]))
    _download_file(valid_url, _resolve(cfg["raw_valid_text"]))


# ---------------------------------------------------------------------------
# Stage 2: Tokenizer training
# ---------------------------------------------------------------------------
def stage_train_tokenizer(cfg: dict) -> None:
    vocab_path = _resolve(cfg["vocab_path"])
    merges_path = _resolve(cfg["merges_path"])
    if vocab_path.exists() and merges_path.exists():
        print("[stage:tokenizer] vocab & merges already exist, skipping.")
        return

    from cs336_basics.build_tokenizer import train_tokenizer

    raw_text = _resolve(cfg["raw_train_text"])
    vocab_size = cfg.get("tokenizer_vocab_size", 10_000)
    special_tokens = cfg.get("tokenizer_special_tokens", ["<|endoftext|>"])
    num_processes = cfg.get("tokenizer_num_processes", 8)

    print(f"[stage:tokenizer] Training BPE tokenizer (vocab_size={vocab_size}) ...")
    vocab, merges = train_tokenizer(
        input_path=str(raw_text),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes,
    )
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"  Saved vocab → {vocab_path}")
    print(f"  Saved merges → {merges_path}")


# ---------------------------------------------------------------------------
# Stage 3: Tokenize text → .npy
# ---------------------------------------------------------------------------
def stage_build_dataset(cfg: dict) -> None:
    mode = cfg.get("dataset_mode", "tinystories")
    if mode == "local_npy":
        print("[stage:dataset] dataset_mode=local_npy, skipping tokenization.")
        return

    train_npy = _resolve(cfg["tokenized_train"])
    valid_npy = _resolve(cfg["tokenized_valid"])
    if train_npy.exists() and valid_npy.exists():
        print("[stage:dataset] .npy files already exist, skipping.")
        return

    from cs336_basics.build_dataset import _encode_file

    vocab_path = _resolve(cfg["vocab_path"])
    merges_path = _resolve(cfg["merges_path"])
    num_workers = cfg.get("build_dataset_num_workers", 10)

    for raw_key, npy_path in [("raw_train_text", train_npy), ("raw_valid_text", valid_npy)]:
        raw = _resolve(cfg[raw_key])
        if npy_path.exists():
            print(f"  [skip] {npy_path} already exists.")
            continue
        print(f"  Tokenizing {raw} → {npy_path} ...")
        tokens = _encode_file(raw, vocab_path, merges_path, max_size=None, num_workers=num_workers)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), tokens)
        print(f"  {len(tokens):,} tokens written.")


# ---------------------------------------------------------------------------
# Stage 4: Train LM
# ---------------------------------------------------------------------------
def _get_kernel_fn(name: str):
    """Lazy-import to avoid Triton init when not needed."""
    from cs336_basics.transfromer.scaled_dot_prod_attention import ATTENTION_KERNEL_REGISTRY

    if name not in ATTENTION_KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown attention_kernel '{name}'. "
            f"Valid: {list(ATTENTION_KERNEL_REGISTRY.keys())}"
        )
    return ATTENTION_KERNEL_REGISTRY[name]


def stage_train_single_gpu(cfg: dict) -> None:
    """Single-GPU training via lm_trainer.train_lm()."""
    from cs336_basics.lm_trainer import train_lm

    m = cfg["model"]
    t = cfg["training"]
    c = cfg["checkpointing"]
    w = cfg["wandb"]

    kernel_name = cfg.get("attention_kernel", "scaled_dot_prod_attention")
    print(f"[stage:train] attention_kernel={kernel_name}  ddp_wrapper=none")

    # NOTE: train_lm uses the default attention inside TransformerLM.
    # To inject a custom kernel, we monkey-patch the model constructor's default.
    # We pass attention_fn indirectly by overriding the module-level default.
    # For the single-GPU path, we modify lm.py's default at import time.
    from cs336_basics import lm as lm_module

    original_default = lm_module.scaled_dot_product_attention
    kernel_fn = _get_kernel_fn(kernel_name)
    lm_module.scaled_dot_product_attention = kernel_fn

    try:
        result = train_lm(
            TRAIN_PATH=str(_resolve(cfg["tokenized_train"])),
            VAL_PATH=str(_resolve(cfg["tokenized_valid"])),
            VOCAB_PATH=str(_resolve(cfg["vocab_path"])),
            MERGES_PATH=str(_resolve(cfg["merges_path"])),
            TR_BAT_SIZE=t.get("tr_batch_size", 32),
            VAL_SAMP_SIZE=t.get("val_sample_size", 50),
            VAL_BAT_SIZE=t.get("val_batch_size", 32),
            CONTEXT_LENGTH=m.get("context_length", 256),
            EPOCHES=t.get("epochs", 5000),
            VOCAB_SIZE=m.get("vocab_size", 10_000),
            NUM_LAYERS=m.get("num_layers", 4),
            D_MODEL=m.get("d_model", 512),
            NUM_HEADS=m.get("num_heads", 16),
            D_FF=m.get("d_ff", 1344),
            ROPE_THETA=m.get("rope_theta", 10_000.0),
            LR=t.get("lr", 6e-4),
            WEIGHT_DECAY=t.get("weight_decay", 0.01),
            BETA1=t.get("beta1", 0.9),
            BETA2=t.get("beta2", 0.999),
            ADAM_EPS=t.get("adam_eps", 1e-8),
            GRAD_CLIP=t.get("grad_clip", 1.0),
            MAX_ITERS=t.get("max_iters", 5000),
            WARMUP_ITERS=t.get("warmup_iters", 1500),
            DEVICE="cuda",
            DTYPE=t.get("dtype", "float32"),
            COMPILE=t.get("compile", True),
            CHECKPOINT_DIR=c.get("checkpoint_dir", "checkpoints"),
            LOG_INTERVAL=c.get("log_interval", 50),
            EVAL_INTERVAL=c.get("eval_interval", 100),
            SAVE_INTERVAL=c.get("save_interval", 200),
            CHECKPOINTING_EVERY=c.get("checkpointing_every"),
            SEED=t.get("seed", 0),
            WANDB_PROJECT=w.get("project", "Train_Transformer_LM"),
            WANDB_RUN_NAME=w.get("run_name"),
        )
        print(f"[stage:train] done — last_val_perplexity={result.get('last_val_perplexity')}")
    finally:
        lm_module.scaled_dot_product_attention = original_default


def stage_train_ddp(cfg: dict, wrapper: str) -> None:
    """Multi-GPU DDP training via mp.spawn."""
    _ensure_cuda()
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("[stage:train] Only 1 GPU detected — falling back to single-GPU training.")
        stage_train_single_gpu(cfg)
        return

    m = cfg["model"]
    t = cfg["training"]
    kernel_name = cfg.get("attention_kernel", "scaled_dot_prod_attention")
    kernel_fn = _get_kernel_fn(kernel_name)

    from cs336_basics.lm import TransformerLM
    from cs336_basics.train.loss import cross_entropy
    from cs336_systems.Parallelization.DDP.stream_dataset import TokenStreamDataset

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(t.get("dtype", "float32"), torch.float32)

    model_kwargs = dict(
        vocab_size=m["vocab_size"],
        context_length=m["context_length"],
        num_layers=m["num_layers"],
        d_model=m["d_model"],
        heads_num=m["num_heads"],
        d_ff=m["d_ff"],
        theta=m.get("rope_theta", 10_000.0),
        device="cpu",
        dtype=dtype,
        attention_fn=kernel_fn,
    )
    model = TransformerLM(**model_kwargs)

    optim_kwargs = dict(
        lr=t.get("lr", 3e-4),
        weight_decay=t.get("weight_decay", 0.01),
        betas=(t.get("beta1", 0.9), t.get("beta2", 0.999)),
        eps=t.get("adam_eps", 1e-8),
    )

    tr_dataset = TokenStreamDataset(str(_resolve(cfg["tokenized_train"])), m["context_length"])
    val_dataset = TokenStreamDataset(str(_resolve(cfg["tokenized_valid"])), m["context_length"])

    epochs = t.get("epochs", 10)
    eval_interval = cfg["checkpointing"].get("eval_interval", 5)
    tr_bs = t.get("tr_batch_size", 8)
    val_bs = t.get("val_batch_size", 8)
    backend = "nccl"

    if wrapper == "naive":
        from cs336_systems.Parallelization.DDP.naiveDDP import naive_LLM_DDP

        print(f"[stage:train] DDP=naive  kernel={kernel_name}  GPUs={world_size}")
        mp.spawn(
            fn=naive_LLM_DDP,
            args=(world_size, tr_dataset, val_dataset, model, optim_kwargs,
                  cross_entropy, epochs, eval_interval, tr_bs, val_bs, backend),
            nprocs=world_size, join=True,
        )
    elif wrapper == "flashddp":
        from cs336_systems.Parallelization.FlashDDP.FlashDDP import DDPOverlapBucketed
        from cs336_systems.Parallelization.FlashDDP.FlashDDP_runner import parallel_train

        bucket_mb = cfg.get("bucket_size_mb", 1)
        print(f"[stage:train] DDP=flashddp  kernel={kernel_name}  GPUs={world_size}  bucket={bucket_mb}MB")
        mp.spawn(
            fn=parallel_train,
            args=(DDPOverlapBucketed, world_size, tr_dataset, val_dataset, model,
                  optim_kwargs, cross_entropy, epochs, eval_interval, tr_bs, val_bs,
                  backend, None, None, bucket_mb),
            nprocs=world_size, join=True,
        )
    else:
        raise ValueError(f"Unknown ddp_wrapper '{wrapper}'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end CUDA-only LM pipeline: download → tokenize → train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Path to JSON pipeline config.")
    p.add_argument("--attention_kernel", type=str, default=None, choices=VALID_KERNELS,
                   help="Override config attention_kernel.")
    p.add_argument("--ddp_wrapper", type=str, default=None, choices=VALID_DDP,
                   help="Override config ddp_wrapper.")
    p.add_argument("--checkpointing_every", type=int, default=None,
                   help="Override config checkpointing_every.")
    p.add_argument("--skip_data", action="store_true",
                   help="Skip download/tokenizer/dataset stages (assume data exists).")
    p.add_argument("--override", nargs="*", default=[],
                   help="Dot-separated key=value overrides, e.g. training.epochs=1000")
    return p


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply dot-separated key=value overrides to the config dict."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        key, value = ov.split("=", 1)
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Auto-cast to int/float/bool/null
        for caster in (int, float):
            try:
                value = caster(value)
                break
            except (ValueError, TypeError):
                continue
        else:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() == "null" or value.lower() == "none":
                value = None
        d[parts[-1]] = value


def main() -> None:
    _ensure_cuda()
    args = build_parser().parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # CLI overrides take precedence over config file
    if args.attention_kernel:
        cfg["attention_kernel"] = args.attention_kernel
    if args.ddp_wrapper:
        cfg["ddp_wrapper"] = args.ddp_wrapper
    if args.checkpointing_every is not None:
        cfg.setdefault("checkpointing", {})["checkpointing_every"] = args.checkpointing_every
    _apply_overrides(cfg, args.override)

    kernel = cfg.get("attention_kernel", "scaled_dot_prod_attention")
    ddp = cfg.get("ddp_wrapper", "none")

    print("=" * 60)
    print(f"  Pipeline: kernel={kernel}  ddp={ddp}")
    print("=" * 60)

    # Stage 1-3: data prep
    if not args.skip_data:
        stage_download(cfg)
        stage_train_tokenizer(cfg)
        stage_build_dataset(cfg)
    else:
        print("[pipeline] --skip_data: skipping data preparation stages.")

    # Stage 4: training
    if ddp == "none":
        stage_train_single_gpu(cfg)
    else:
        stage_train_ddp(cfg, ddp)

    print("\n[pipeline] All stages complete.")


if __name__ == "__main__":
    main()
