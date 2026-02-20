import argparse
import wandb
import os
import torch
DTYPE_DICT={
    "float32": torch.float32,
    "float16": torch.float16
}

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
parser.add_argument("--VAL_SAMP_SIZE", type=int, default=100, help="Sequences per optimization step.")
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



args = parser.parse_args()

TRAIN_PATH = args.TRAIN_PATH
VAL_PATH = args.VAL_PATH
VOCAB_PATH = args.VOCAB_PATH
MERGES_PATH = args.MERGES_PATH
TR_BAT_SIZE = args.TR_BAT_SIZE

VAL_SAMP_SIZE = args.VAL_SAMP_SIZE
VAL_BAT_SIZE = args.VAL_BAT_SIZE
EPOCHES = args.EPOCHES

CONTEXT_LENGTH = args.CONTEXT_LENGTH
VOCAB_SIZE = args.VOCAB_SIZE
NUM_LAYERS = args.NUM_LAYERS
D_MODEL = args.D_MODEL
NUM_HEADS = args.NUM_HEADS
D_FF = args.D_FF
ROPE_THETA = args.ROPE_THETA

LR = args.LR
WEIGHT_DECAY = args.WEIGHT_DECAY
BETAS = (args.BETA1, args.BETA2)
ADAM_EPS = args.ADAM_EPS
GRAD_CLIP = args.GRAD_CLIP
MAX_ITERS = args.MAX_ITERS
WARMUP_ITERS = args.WARMUP_ITERS

DEVICE = args.DEVICE
DTYPE = DTYPE_DICT[args.DTYPE]
COMPILE = args.COMPILE

CHECKPOINT_DIR = args.CHECKPOINT_DIR
RESUME_FROM = args.RESUME_FROM
LOG_INTERVAL = args.LOG_INTERVAL
EVAL_INTERVAL = args.EVAL_INTERVAL
SAVE_INTERVAL = args.SAVE_INTERVAL
CHECKPOINTING_EVERY = args.CHECKPOINTING_EVERY
# Resolution: CHECKPOINTING_EVERY overrides SAVE_INTERVAL when set.
_effective_save_interval = CHECKPOINTING_EVERY if CHECKPOINTING_EVERY is not None else SAVE_INTERVAL
SEED = args.SEED

WANDB_PROJECT = args.WANDB_PROJECT
WANDB_RUN_NAME = args.WANDB_RUN_NAME

print("==== Trainer arguments ====")
for name in sorted(vars(args)):
    print(f"{name}: {getattr(args, name)}")
print("===========================")

# Prepared the logging 
os.makedirs(os.path.join(CHECKPOINT_DIR, f"lr-{LR}-beta1-{BETAS[0]}-beta2-{BETAS[1]}"), exist_ok=True)
if not WANDB_PROJECT:
    raise RuntimeError("Provide a WanDB Project to log the results.")
run = wandb.init(project= WANDB_PROJECT, name=f"lr-{LR}-beta1-{BETAS[0]}-beta2-{BETAS[1]}")


import argparse
from cs336_basics.lm import TransformerLM
from cs336_basics.train.optimizer import AdamW, grad_clip
from cs336_basics.train.checkpointing import load_checkpoint, save_checkpoint, save_checkpoint_and_log
from cs336_basics.train.data_loader import data_loading
from cs336_basics.train.loss import cross_entropy, perplexity
from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
from cs336_basics.train.optimizer import lr_scheduler    

# Initialize Model
lm_model = TransformerLM(VOCAB_SIZE, CONTEXT_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, ROPE_THETA,
                         device=DEVICE, dtype=DTYPE)

# If Compile the PyTorch Code
if COMPILE:
    # Pick a backend based on the selected device.
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        backend = "inductor"
    elif DEVICE.startswith("mps") and torch.backends.mps.is_available():
        # aot_eager is currently the most stable backend for MPS.
        backend = "aot_eager"
    else:
        backend = "eager"

    try:
        lm_model = torch.compile(lm_model, mode="reduce-overhead", backend=backend)
        print(f"Compiled model with backend='{backend}' for kernel fusion.")
    except Exception as compile_err:
        print(f"torch.compile failed ({compile_err}); continuing without compilation.")

# Init Optimizer
opt = AdamW(lm_model.parameters(), LR, WEIGHT_DECAY, BETAS)

# Use load the tokenizer if the data is raw text, 
# But here we directly load the tokenized data, so we can skip this step.
# toeknizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])

# Helper data loader
def _load_np_tokens(path, device):
    arr = np.load(path, mmap_mode="r")
    tensor = torch.from_numpy(arr).long()
    # Pin only for CUDA to overlap H2D copies; MPS/CPU do not support pin_memory the same way.
    if device.startswith("cuda"):
        tensor = tensor.pin_memory()
    return tensor

# Prepare a data loader
train_data = _load_np_tokens(TRAIN_PATH, DEVICE)
valid_data = _load_np_tokens(VAL_PATH, DEVICE)
offsets = torch.arange(CONTEXT_LENGTH, dtype=torch.long, device=train_data.device)

checkpoint_num = 1
# Training Loop
for iter in tqdm(range(EPOCHES), desc="Training", unit="iter"):
    # Randomly sample batch from the training data
    inputs, targets = data_loading(train_data, batch_size=TR_BAT_SIZE, context_length=CONTEXT_LENGTH, device=DEVICE, offsets=offsets)
    # Reset the gradients for all learnable parameters.
    opt.zero_grad() 
    
    prediction = lm_model.forward(inputs)
    tr_loss = cross_entropy(prediction, targets)
    tr_loss.backward() # The returned loss is a Tensor, with a computational graph attached, so that we can bp
    cliped_gra_l2 = grad_clip(lm_model.parameters(), GRAD_CLIP) # Clip gradient
    opt.step() # After bp, all parameters' tensors have collect grad values


    # adjust learning rate
    lr = lr_scheduler(
        it=iter,
        max_learning_rate=LR,
        min_learning_rate=LR * 0.2,
        warmup_iters=WARMUP_ITERS,
        cosine_cycle_aiters=MAX_ITERS,
    )
    for group in opt.param_groups:
        group["lr"] = lr
    if iter % EVAL_INTERVAL == 0:
        with torch.no_grad():
            # Compute the Validation Loss (Perplexity)
            val_loss = 0
            for sample in range(VAL_SAMP_SIZE):
                # Randomly sample batch from the validation data
                inputs, targets = data_loading(valid_data, batch_size=VAL_BAT_SIZE, context_length=CONTEXT_LENGTH, device=DEVICE, offsets=offsets)
                prediction = lm_model.forward(inputs)
                val_loss += perplexity(prediction, targets)
            print(f"Iter:{iter} | Training Loss: {tr_loss} | Validation Loss: {val_loss/VAL_SAMP_SIZE}")
            wandb.log({
            "train_loss (CE)": tr_loss,
            "validation_loss (Perplexity)": val_loss/VAL_SAMP_SIZE,
            "gradient": cliped_gra_l2, # Save the gradient progerss (under gradient clipping)
            "iter": iter
            })

    if (iter != 0 and iter % SAVE_INTERVAL == 0) or iter == EPOCHES-1:
        # Use _effective_save_interval for periodic saves; always save final.
        _is_periodic = _effective_save_interval > 0 and iter != 0 and iter % _effective_save_interval == 0
        _is_final = iter == EPOCHES - 1
        if _is_periodic or _is_final:
            with torch.no_grad():
                # Compute the Validation Loss (Perplexity)
                val_loss = 0
                for sample in range(VAL_SAMP_SIZE):
                    # Sample validation batch
                    inputs, targets = data_loading(valid_data, batch_size=VAL_BAT_SIZE, context_length=CONTEXT_LENGTH, device=DEVICE, offsets=offsets)
                    prediction = lm_model.forward(inputs)
                    val_loss += perplexity(prediction, targets)
                print(f"Saving Checkpoint....")
                print(f"Checkpoint {checkpoint_num}: Training Loss: {tr_loss} | Validation Loss: {val_loss/VAL_SAMP_SIZE}")

                # Log into WanDB
                local_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"lr-{LR}-beta1-{BETAS[0]}-beta2-{BETAS[1]}/iter_{iter}-loss_{val_loss/VAL_SAMP_SIZE}.pt")
                save_checkpoint_and_log(lm_model, opt, iter, local_checkpoint_path, run)
                checkpoint_num += 1
