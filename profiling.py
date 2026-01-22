import timeit
import argparse
import wandb
import os
import torch
import pandas as pd
import os
import torch.cuda.nvtx as nvtx

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.9"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.8"

DTYPE_DICT={
    "float32": torch.float32,
    "float16": torch.float16
}

parser = argparse.ArgumentParser(description="Training LLM")

# Profiling Arguments
config_df = pd.DataFrame({
    "SIZE": ["xl"],
    "D_MODEL": [1280],
    "NUM_HEADS": [20],      # head dim ~32–64
    "D_FF": [5120], # ≈4×d_model
    "NUM_LAYERS": [36],
})


# Data / experiment setup.
parser.add_argument("--WARM_UP_ITER", type=int, required=True, help="Path to tokenized training data file.")
parser.add_argument("--PROFILE_ITER", type=int, required=True, help="Path to tokenized training data file.")
parser.add_argument("--TRAIN_PATH", type=str, required=True, help="Path to tokenized training data file.")
parser.add_argument("--VAL_PATH", type=str, required=True, help="Path to tokenized validation data file.")
parser.add_argument("--VOCAB_PATH", type=str, required=True, help="Pickled tokenizer vocab data file.")
parser.add_argument("--MERGES_PATH", type=str, required=True, help="Pickled tokenizer merges data file.")
parser.add_argument("--TR_BAT_SIZE", type=int, default=4, help="Sequences per optimization step.")
parser.add_argument("--VAL_SAMP_SIZE", type=int, default=100, help="Sequences per optimization step.")
parser.add_argument("--VAL_BAT_SIZE", type=int, default=32, help="Sequences per optimization step.")
parser.add_argument("--CONTEXT_LENGTH", type=int, default=256, help="Tokens per training sequence.")
parser.add_argument("--EPOCHES", type=int, default=500, help="Number of training epoches.")

# Model hyperparameters.
parser.add_argument("--VOCAB_SIZE", type=int, required=True, help="Pickled tokenizer vocab data file.")
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

# Training autocast dtype
parser.add_argument("--CAST_DTYPE", type=str, default="float32", help="Torch autocast dtype string, e.g., 'float32', 'bfloat16'.")

# Profiling settings
parser.add_argument("--MEMORY_PROFILE", type=bool, default=False, help="Whether to perform memory profiling during training.")


args = parser.parse_args()

WARM_UP_ITER = args.WARM_UP_ITER
PROFILE_ITER = args.PROFILE_ITER

TRAIN_PATH = args.TRAIN_PATH
VAL_PATH = args.VAL_PATH
VOCAB_PATH = args.VOCAB_PATH
MERGES_PATH = args.MERGES_PATH
TR_BAT_SIZE = args.TR_BAT_SIZE

VAL_SAMP_SIZE = args.VAL_SAMP_SIZE
VAL_BAT_SIZE = args.VAL_BAT_SIZE
EPOCHES = WARM_UP_ITER + PROFILE_ITER

CONTEXT_LENGTH = args.CONTEXT_LENGTH
VOCAB_SIZE = args.VOCAB_SIZE
ROPE_THETA = args.ROPE_THETA

LR = args.LR
WEIGHT_DECAY = args.WEIGHT_DECAY
BETAS = (args.BETA1, args.BETA2)
ADAM_EPS = args.ADAM_EPS
GRAD_CLIP = args.GRAD_CLIP
MAX_ITERS = args.MAX_ITERS
WARMUP_ITERS = args.WARMUP_ITERS
# Add a safe fall back
DEVICE = args.DEVICE
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    print("CUDA is not available; switching to CPU.")
    DEVICE = "mps"
DTYPE = DTYPE_DICT[args.DTYPE]
COMPILE = args.COMPILE

SEED = 0

DTYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "tf32": torch.float32,
}
CAST_DTYPE = DTYPE_DICT[args.CAST_DTYPE]

MEMORY_PROFILE = args.MEMORY_PROFILE


def timing_wrapper(function_obj:object, inputs: dict=None, contianer:dict=None):
    """
    Define a function wrapper to wrap around the function, 
    so that we can get the function's return val.
    """
    contianer["value"] = function_obj(**inputs)


######################################################################
######################################################################
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

if MEMORY_PROFILE:
    torch.cuda.memory._record_memory_history(max_entries=25_000)

profiling_result = pd.DataFrame(
    columns=["size", "forward_pass_time", "backward_pass_time"]
)

def _sync_device():
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif DEVICE.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

# Profile all the model configs
for _, config in config_df.iterrows():
    # Get the model configuration
    SIZE = config["SIZE"]
    NUM_LAYERS = config["NUM_LAYERS"]
    D_MODEL = config["D_MODEL"]
    NUM_HEADS = config["NUM_HEADS"]
    D_FF = config["D_FF"]

    # Initialize Modules
    lm_model = TransformerLM(VOCAB_SIZE, CONTEXT_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, ROPE_THETA,
                            device=DEVICE, dtype=DTYPE)
    """ DON'T COMPILE
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
    """
    opt = AdamW(lm_model.parameters(), LR, WEIGHT_DECAY, BETAS)
    toeknizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])

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
    
    # Training Loop
    forward_pass_times = []
    backward_pass_times = []

    # Warm-up iterations without timing
    nvtx.range_push("Warm-up")
    for iter in tqdm(range(WARM_UP_ITER), desc="Training", unit="iter"):
        inputs, targets = data_loading(train_data, TR_BAT_SIZE, CONTEXT_LENGTH, DEVICE, offsets)
        
        # Reset the gradients for all learnable parameters.
        opt.zero_grad() 
        
        #Forward
        prediction = lm_model.forward(x=inputs)
        tr_loss = cross_entropy(prediction, targets)
        tr_loss.backward()
        cliped_gra_l2 = grad_clip(lm_model.parameters(), GRAD_CLIP) # Clip gradient

        opt.step()

    nvtx.range_pop()

    # Profiling iterations with timing
    nvtx.range_push("Profiling")
    for iter in tqdm(range(PROFILE_ITER), desc="Training", unit="iter"):
        # Data Loading
        inputs, targets = data_loading(train_data, TR_BAT_SIZE, CONTEXT_LENGTH, DEVICE, offsets)

        # Reset the gradients for all learnable parameters.
        opt.zero_grad() 

        #Forward
        value = {}
        _sync_device()
        forward_start = timeit.default_timer()
        
        # Record the following chunks
        with nvtx.range("forward_pass"):
            if CAST_DTYPE != torch.float32:
                with torch.autocast(device_type=DEVICE, dtype=CAST_DTYPE):
                    prediction = lm_model.forward(x=inputs)
            else:
                prediction = lm_model.forward(x=inputs)

        _sync_device()
        forward_pass_time = timeit.default_timer() - forward_start
        tr_loss = cross_entropy(prediction, targets)

        # Backward
        _sync_device()
        backward_start = timeit.default_timer()
        # Record the following chunks
        with nvtx.range("backward_pass"):
            tr_loss.backward()

        _sync_device()
        backward_pass_time = timeit.default_timer() - backward_start
        cliped_gra_l2 = grad_clip(lm_model.parameters(), GRAD_CLIP) # Clip gradient
        
        opt.step() 

        # After bp, all parameters' tensors have collect grad values
        forward_pass_times.append(forward_pass_time)
        backward_pass_times.append(backward_pass_time)

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
    nvtx.range_pop()  # pop the profiling range
    
    # Store the profiling results
    profiling_result.loc[len(profiling_result)] = [SIZE, 
                                                   sum(forward_pass_times)/len(forward_pass_times), 
                                                   sum(backward_pass_times)/len(backward_pass_times)
                                                   ]

    # Clear Cache
    del lm_model, opt, train_data, valid_data, offsets
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.cuda.empty_cache()

    

# Display the profiling summary instead of calling the nonexistent DataFrame.view().
print(profiling_result)

if MEMORY_PROFILE:
    torch.cuda.memory._dump_snapshot(f"memory_profile_context_len{CONTEXT_LENGTH}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
