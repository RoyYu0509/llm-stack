import argparse
import os
from collections.abc import Sequence

import numpy as np
import torch
import wandb
from tqdm import tqdm

from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer
from cs336_basics.lm import TransformerLM
from cs336_basics.train.checkpointing import save_checkpoint_and_log
from cs336_basics.train.data_loader import data_loading
from cs336_basics.train.loss import cross_entropy, perplexity
from cs336_basics.train.optimizer import AdamW, grad_clip, lr_scheduler
from cs336_basics.transfromer.scaled_dot_prod_attention import scaled_dot_product_attention

DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
}


def build_trainer_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used by the original trainer script."""
    parser = argparse.ArgumentParser(description="Training LLM")

    # Logging
    parser.add_argument("--WANDB_PROJECT", type=str, default=None, help="Weights & Biases project (optional).")
    parser.add_argument("--WANDB_RUN_NAME", type=str, default=None, help="Weights & Biases run name.")
    # Data / experiment setup.
    parser.add_argument("--TRAIN_PATH", type=str, required=True, help="Path to tokenized training data file.")
    parser.add_argument("--VAL_PATH", type=str, required=True, help="Path to tokenized validation data file.")
    parser.add_argument("--VOCAB_PATH", type=str, required=True, help="Pickled tokenizer vocab data file.")
    parser.add_argument("--MERGES_PATH", type=str, required=True, help="Pickled tokenizer merges data file.")
    parser.add_argument("--TR_BAT_SIZE", type=int, default=32, help="Sequences per optimization step.")
    parser.add_argument("--VAL_BAT_NUM", type=int, default=100, help="Number of validation batches sampled per evaluation.")
    parser.add_argument("--VAL_BAT_SIZE", type=int, default=32, help="Sequences per optimization step.")
    parser.add_argument("--CONTEXT_LENGTH", type=int, default=256, help="Tokens per training sequence.")
    parser.add_argument("--EPOCHES", type=int, default=500, help="Number of training epoches.")

    # Model hyperparameters.
    parser.add_argument("--VOCAB_SIZE", type=int, default=10_000, help="Tokenizer vocabulary size.")
    parser.add_argument("--NUM_LAYERS", type=int, default=4, help="Transformer block count.")
    parser.add_argument("--D_MODEL", type=int, default=512, help="Transformer embedding dimension.")
    parser.add_argument("--NUM_HEADS", type=int, default=16, help="Attention head count.")
    parser.add_argument("--D_FF", type=int, default=1_344, help="Point-wise FFN hidden size.")
    parser.add_argument("--ROPE_THETA", type=float, default=10_000.0, help="RoPE theta parameter.")

    # Optimization settings.
    parser.add_argument("--LR", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--BETA1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--BETA2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--ADAM_EPS", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--GRAD_CLIP", type=float, default=1.0, help="Global gradient norm clip value.")
    parser.add_argument("--MAX_ITERS", type=int, default=10_000, help="Number of optimizer steps.")
    parser.add_argument("--WARMUP_ITERS", type=int, default=2_000, help="Linear warmup steps.")

    # Device.
    parser.add_argument("--DEVICE", type=str, default="cpu", help="Torch device string, e.g., 'cuda', 'cpu', 'mps'.")
    parser.add_argument("--DTYPE", type=str, default="float32", help="Torch dtype string, e.g., 'float32', 'bfloat16'.")
    parser.add_argument("--COMPILE", action="store_true", help="Compile the model to enable kernel fusion.")

    # Checkpointing
    parser.add_argument("--CHECKPOINT_DIR", type=str, default="checkpoints", help="Where to store checkpoints.")
    parser.add_argument("--RESUME_FROM", type=str, default=None, help="Checkpoint file to resume from.")
    parser.add_argument("--LOG_INTERVAL", type=int, default=50, help="Steps between training log prints.")
    parser.add_argument("--EVAL_INTERVAL", type=int, default=500, help="Steps between validation runs.")
    parser.add_argument("--SAVE_INTERVAL", type=int, default=1_000, help="Steps between checkpoint saves.")
    parser.add_argument("--CHECKPOINTING_EVERY", type=int, default=None,
                        help="If set, overrides SAVE_INTERVAL as the checkpoint cadence. "
                             "<=0 disables periodic checkpoints (only final checkpoint is saved).")
    parser.add_argument("--SEED", type=int, default=0, help="Random seed.")
    return parser


def parse_trainer_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse trainer args from a token list (or sys.argv if argv is None)."""
    return build_trainer_parser().parse_args(argv)


def _load_np_tokens(path: str, device: str) -> torch.Tensor:
    """Load token ids from .npy and keep transfer-friendly layout for CUDA."""
    arr = np.load(path, mmap_mode="r")
    tensor = torch.from_numpy(arr).long()
    # Pin only for CUDA to overlap H2D copies; MPS/CPU do not support pin_memory the same way.
    if device.startswith("cuda"):
        tensor = tensor.pin_memory()
    return tensor


def local_train(
    TRAIN_PATH: str,
    VAL_PATH: str,
    VOCAB_PATH: str,
    MERGES_PATH: str,
    TR_BAT_SIZE: int = 32,
    VAL_BAT_NUM: int = 100,
    VAL_BAT_SIZE: int = 32,
    CONTEXT_LENGTH: int = 256,
    EPOCHES: int = 500,
    VOCAB_SIZE: int = 10_000,
    NUM_LAYERS: int = 4,
    D_MODEL: int = 512,
    NUM_HEADS: int = 16,
    D_FF: int = 1_344,
    ROPE_THETA: float = 10_000.0,
    LR: float = 3e-4,
    WEIGHT_DECAY: float = 0.01,
    BETA1: float = 0.9,
    BETA2: float = 0.999,
    ADAM_EPS: float = 1e-8,
    GRAD_CLIP: float = 1.0,
    MAX_ITERS: int = 10_000,
    WARMUP_ITERS: int = 2_000,
    DEVICE: str = "cpu",
    DTYPE: str = "float32",
    COMPILE: bool = False,
    CHECKPOINT_DIR: str = "checkpoints",
    RESUME_FROM: str | None = None,
    LOG_INTERVAL: int = 50,
    EVAL_INTERVAL: int = 500,
    SAVE_INTERVAL: int = 1_000,
    CHECKPOINTING_EVERY: int | None = None,
    SEED: int = 0,
    WANDB_PROJECT: str | None = None,
    WANDB_RUN_NAME: str | None = None,
    ATTENTION_FN=scaled_dot_product_attention,
):
    """
    Train a Transformer language model end-to-end.

    This is a Python API version of the old CLI trainer. Parameter names are
    intentionally ALL_CAPS to match previous `--FLAG` names one-to-one.

    Args:
        TRAIN_PATH: `.npy` file of training token ids.
        VAL_PATH: `.npy` file of validation token ids.
        VOCAB_PATH: tokenizer vocab file path.
        MERGES_PATH: tokenizer merges file path.
        TR_BAT_SIZE: training batch size.
        VAL_SAMP_SIZE: number of sampled validation batches per eval pass.
        VAL_BAT_SIZE: validation batch size.
        CONTEXT_LENGTH: tokens per sequence.
        EPOCHES: total training iterations.
        VOCAB_SIZE: model vocabulary size.
        NUM_LAYERS: number of transformer blocks.
        D_MODEL: model hidden dimension.
        NUM_HEADS: number of attention heads.
        D_FF: FFN hidden size.
        ROPE_THETA: RoPE theta value.
        LR: max learning rate for schedule.
        WEIGHT_DECAY: AdamW weight decay.
        BETA1: AdamW beta1.
        BETA2: AdamW beta2.
        ADAM_EPS: AdamW epsilon.
        GRAD_CLIP: global gradient norm clipping threshold.
        MAX_ITERS: cosine scheduler cycle length.
        WARMUP_ITERS: linear warmup length.
        DEVICE: target torch device string.
        DTYPE: dtype key in `DTYPE_DICT`.
        COMPILE: whether to `torch.compile` the model.
        CHECKPOINT_DIR: parent directory for checkpoints.
        RESUME_FROM: reserved for checkpoint resume (currently informational only).
        LOG_INTERVAL: steps between train-loss prints.
        EVAL_INTERVAL: steps between validation runs.
        SAVE_INTERVAL: steps between checkpoint saves.
        CHECKPOINTING_EVERY: if set, overrides SAVE_INTERVAL as the cadence.
            <=0 disables periodic checkpoints (only final checkpoint is saved).
        SEED: random seed.
        WANDB_PROJECT: Weights & Biases project name (required).
        WANDB_RUN_NAME: optional run name override.

    Returns:
        dict with trained model, optimizer, wandb run object, and final metrics.
    """
    if DTYPE not in DTYPE_DICT:
        supported = ", ".join(sorted(DTYPE_DICT))
        raise ValueError(f"Unsupported DTYPE '{DTYPE}'. Supported values: {supported}")

    if not WANDB_PROJECT:
        raise RuntimeError("Provide a WanDB Project to log the results.")

    torch.manual_seed(SEED)
    betas = (BETA1, BETA2)
    torch_dtype = DTYPE_DICT[DTYPE]

    # Resolve checkpoint cadence: CHECKPOINTING_EVERY overrides SAVE_INTERVAL.
    _effective_save_interval = CHECKPOINTING_EVERY if CHECKPOINTING_EVERY is not None else SAVE_INTERVAL
    # Use the legacy naming scheme so old experiments/checkpoint folders look familiar.
    run_name = WANDB_RUN_NAME or f"lr-{LR}-beta1-{BETA1}-beta2-{BETA2}"
    run_dir = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Start a W&B run first so metrics/artifacts are attached to one run context.
    run = wandb.init(project=WANDB_PROJECT, name=run_name)

    # 1) Build model from hyperparameters.
    lm_model = TransformerLM(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        NUM_LAYERS,
        D_MODEL,
        NUM_HEADS,
        D_FF,
        ROPE_THETA,
        device=DEVICE,
        dtype=torch_dtype,
        attention_fn=ATTENTION_FN,
    )

    # 2) Optionally compile for backend-specific speedups.
    if COMPILE:
        if DEVICE.startswith("cuda") and torch.cuda.is_available():
            backend = "inductor"
        elif DEVICE.startswith("mps") and torch.backends.mps.is_available():
            backend = "aot_eager"
        else:
            backend = "eager"

        try:
            lm_model = torch.compile(lm_model, mode="reduce-overhead", backend=backend)
            print(f"Compiled model with backend='{backend}' for kernel fusion.")
        except Exception as compile_err:
            print(f"torch.compile failed ({compile_err}); continuing without compilation.")

    opt = AdamW(
        lm_model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=betas,
        eps=ADAM_EPS,
    )
    # 3) Validate tokenizer assets exist/load (matches old script side effect).
    Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])

    if RESUME_FROM:
        raise NotImplementedError("Checkpoint resume is not yet implemented in the API trainer.")

    # 4) Load token arrays once, then sample random subsequences each iteration.
    train_data = _load_np_tokens(TRAIN_PATH, DEVICE)
    valid_data = _load_np_tokens(VAL_PATH, DEVICE)
    # Reuse positional offsets across batches to avoid per-step tensor allocation.
    offsets = torch.arange(CONTEXT_LENGTH, dtype=torch.long, device=train_data.device)

    checkpoint_num = 1
    last_train_loss = None
    last_val_perplexity = None

    for iteration in tqdm(range(EPOCHES), desc="Training", unit="iter"):
        # 5) Sample a random train batch.
        inputs, targets = data_loading(train_data, TR_BAT_SIZE, CONTEXT_LENGTH, DEVICE, offsets)
        opt.zero_grad()

        # 6) Forward + loss + backward + clipping + optimizer step.
        prediction = lm_model.forward(inputs)
        tr_loss = cross_entropy(prediction, targets)
        tr_loss.backward()
        clipped_grad_l2 = grad_clip(lm_model.parameters(), GRAD_CLIP)
        opt.step()

        # 7) Update LR using warmup then cosine decay (same policy as CLI trainer).
        lr = lr_scheduler(
            it=iteration,
            max_learning_rate=LR,
            min_learning_rate=LR * 0.2,
            warmup_iters=WARMUP_ITERS,
            cosine_cycle_aiters=MAX_ITERS,
        )
        for group in opt.param_groups:
            group["lr"] = lr

        if iteration % LOG_INTERVAL == 0:
            print(f"Iter:{iteration} | Training Loss: {tr_loss}")

        # 8) Run periodic validation and log to W&B. (sample VAL_BAT_NUM batches)
        if iteration % EVAL_INTERVAL == 0:
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(VAL_BAT_NUM):
                    inputs, targets = data_loading(valid_data, VAL_BAT_SIZE, CONTEXT_LENGTH, DEVICE, offsets)
                    prediction = lm_model.forward(inputs)
                    val_loss += perplexity(prediction, targets)

                val_perplexity = val_loss / VAL_BAT_NUM
                last_train_loss = tr_loss
                last_val_perplexity = val_perplexity

                print(
                    f"Iter:{iteration} | Training Loss: {tr_loss} | "
                    f"Validation Loss: {val_perplexity}"
                )
                wandb.log(
                    {
                        # Keep metric keys unchanged so existing dashboards still match.
                        "train_loss (CE)": tr_loss,
                        "validation_loss (Perplexity)": val_perplexity,
                        "gradient": clipped_grad_l2,
                        "iter": iteration,
                    }
                )

        # 9) Save checkpoints periodically (and always at the final iteration).
        _is_periodic = _effective_save_interval > 0 and iteration != 0 and iteration % _effective_save_interval == 0
        _is_final = iteration == EPOCHES - 1
        if _is_periodic or _is_final:
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(VAL_BAT_NUM):
                    inputs, targets = data_loading(valid_data, VAL_BAT_SIZE, CONTEXT_LENGTH, DEVICE, offsets)
                    prediction = lm_model.forward(inputs)
                    val_loss += perplexity(prediction, targets)
                val_perplexity = val_loss / VAL_BAT_NUM

                print("Saving Checkpoint....")
                print(
                    f"Checkpoint {checkpoint_num}: Training Loss: {tr_loss} | "
                    f"Validation Loss: {val_perplexity}"
                )

                local_checkpoint_path = os.path.join(
                    run_dir,
                    f"iter_{iteration}-loss_{val_perplexity}.pt",
                )
                save_checkpoint_and_log(lm_model, opt, iteration, local_checkpoint_path, run)
                checkpoint_num += 1
                last_train_loss = tr_loss
                last_val_perplexity = val_perplexity

    return {
        "model": lm_model,
        "optimizer": opt,
        "wandb_run": run,
        "last_iter": EPOCHES - 1,
        "last_train_loss": last_train_loss,
        "last_val_perplexity": last_val_perplexity,
    }


def train_lm_from_args(argv: Sequence[str] | None = None):
    """
    Parse CLI-style args and forward them into `local_train(...)`.

    Useful when you still want list-of-args behavior, but no standalone CLI file.
    """
    args = parse_trainer_args(argv)
    print("==== Trainer arguments ====")
    for name in sorted(vars(args)):
        print(f"{name}: {getattr(args, name)}")
    print("===========================")
    return local_train(**vars(args))
