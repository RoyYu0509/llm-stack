import pandas as pd
import torch
import cs336_basics
from cs336_basics.transfromer.scaled_dot_prod_attention import scaled_dot_product_attention
import timeit
from cs336_basics.transfromer.multiheads_attention import MultiHeadsAttention
import argparse
from cs336_basics.train.optimizer import AdamW
import torch.cuda.nvtx as nvtx
import os
import tqdm
import matplotlib.pyplot as plt
from cs336_basics.transfromer.scaled_dot_prod_attention import softmax, scaled_dot_product_attention, flash_attention_my_triton, flash_attention_torch


parser = argparse.ArgumentParser(description="Benchmarking Attention Mechanism")
parser.add_argument("--DTYPE", type=str, default="float32", help="Data type for the tensors")
parser.add_argument("--PROFILE_FORWARD_MEMORY", type=bool, default=False, help="Whether to perform memory profiling during forward pass.")
parser.add_argument("--PROFILE_BACKWARD_MEMORY", type=bool, default=False, help="Whether to perform memory profiling during backward pass.")
parser.add_argument("--COMPILED", action="store_true", help="Whether to use compiled attention module")
parser.add_argument("--MyTritonAttn", action="store_true", help="Whether to use MyTriton attention module")
parser.add_argument("--RefTritonAttn", action="store_true", help="Whether to use Reference Triton module")
parser.add_argument("--VecTorchAttn", action="store_true", help="Whether to use Vectorized Torch attention module")

args = parser.parse_args()
DTYPE = getattr(torch, args.DTYPE)
PROFILE_FORWARD_MEMORY = args.PROFILE_FORWARD_MEMORY
PROFILE_BACKWARD_MEMORY = args.PROFILE_BACKWARD_MEMORY
COMPILED = args.COMPILED
COMPILED_STR = "True" if COMPILED else "False"
MyTritonAttn = args.MyTritonAttn
RefTritonAttn = args.RefTritonAttn
VecTorchAttn = args.VecTorchAttn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def _sync_device():
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

def attn_kernel_selector():
    """Select attention function based on command-line arguments."""
    if MyTritonAttn:
        return flash_attention_my_triton, "MyTriton"
    elif VecTorchAttn:
        return flash_attention_torch, "VecTorch"
    else:
        return scaled_dot_product_attention, "Baseline"


def benchmarking_naive_attention(
        atten_fn:callable,
        configs:list,  # List of (heads, d_model) tuples
        profiling_dir:str,
        context_length:int=128, batch_size:int=16, 
        device:torch.device=torch.device("cuda"), dtype:torch.dtype=DTYPE,
    ):
    """
    Benchmarking the attention mechanism.
    """
    # Record
    df = pd.DataFrame({
        "heads_num": [],
        "d_model": [],
        "forward_time": [],
        "backward_time": [],
    })
    for head, d_model in configs:
        print(f"Benchmarking MultiHead Attention: heads={head}, d_model={d_model}")
        mha = MultiHeadsAttention(d_model, head, device=device, dtype=dtype, attention_fn=atten_fn)
        if COMPILED:
            mha = torch.compile(mha)
        mha.to(device=device, dtype=dtype)
        opt = AdamW(mha.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        forward_times = 0
        backward_times = 0
        
        # Warm-up
        nvtx.range_push("Warm-up")
        for _ in range(10):  # Warm-up
            opt.zero_grad()  # Clear gradients to prevent accumulation
            x = torch.randn((batch_size, context_length, d_model), device=device, dtype=dtype, requires_grad=True)
            y = mha._multiHead(x, token_positions=torch.arange(context_length, device=device, dtype=torch.long))
            y.sum().backward()
        opt.zero_grad()  # Clear warmup gradients before benchmarking
        torch.cuda.empty_cache()  # Free warmup memory
        nvtx.range_pop()

        # Benchmarking
        nvtx.range_push("Benchmarking")
        tqdm_iter = 100
        for _ in tqdm.tqdm(range(tqdm_iter), desc=f"heads={head}, d_model={d_model}"):
            opt.zero_grad()
            # Create random QKV matrix 
            shape = (batch_size, context_length, d_model)
            x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

            
            if PROFILE_FORWARD_MEMORY:
                torch.cuda.memory._record_memory_history(max_entries=25_000)
            
            # forward
            _sync_device()
            start_time = timeit.default_timer()
            with nvtx.range("forward_pass"):
                y = mha._multiHead(x, token_positions=torch.arange(context_length, device=device, dtype=torch.long))
            _sync_device()  # Sync BEFORE stopping timer to wait for GPU
            forward_time = timeit.default_timer() - start_time

            if PROFILE_FORWARD_MEMORY:
                torch.cuda.memory._dump_snapshot(os.path.join(profiling_dir, f"head{head}_dmodel{d_model}_mem_f.pickle"))
                torch.cuda.memory._record_memory_history(enabled=None)


            if PROFILE_BACKWARD_MEMORY:
                torch.cuda.memory._record_memory_history(max_entries=25_000)
            # backward
            _sync_device()
            start_time = timeit.default_timer()
            with nvtx.range("backward_pass"):
                y.sum().backward()
            _sync_device()  # Sync BEFORE stopping timer to wait for GPU
            backward_time = timeit.default_timer() - start_time
            if PROFILE_BACKWARD_MEMORY:
                torch.cuda.memory._dump_snapshot(os.path.join(profiling_dir, f"head{head}_dmodel{d_model}_mem_b.pickle"))
                torch.cuda.memory._record_memory_history(enabled=None)

            # Do not update parameters
            # opt.step()
            forward_times += forward_time
            backward_times += backward_time
            
            # Clean up to prevent memory accumulation
            del x, y
            if _ % 10 == 0:  # Periodic cleanup every 10 iterations
                torch.cuda.empty_cache()
        nvtx.range_pop()

        # Record
        df = pd.concat([df, pd.DataFrame({
            "heads_num": [head],
            "d_model": [d_model],
            "forward_time": [forward_times],
            "backward_time": [backward_times],
        })], ignore_index=True)
        
        del mha
        torch.cuda.empty_cache()
    return df
            
            
def main():
    configs = [
        (8, 512),   # d_head=64
        (8, 1024),  # d_head=128
        (8, 2048),  # d_head=256
        (16, 512),  # d_head=32
        (16, 1024), # d_head=64
        (16, 2048), # d_head=128
        (32, 1024), # d_head=32
        (32, 2048), # d_head=64
    ]
    
    # Select attention function
    atten_fn, attn_name = attn_kernel_selector()
    print(f"Starting benchmarking with {attn_name} attention (compiled={COMPILED})...")
    
    # Create directory for profiling results relative to this script
    _here = os.path.dirname(os.path.abspath(__file__))
    global _profiling_dir
    _profiling_dir = os.path.join(_here, f"profile_{attn_name}_compiled_{COMPILED_STR}")
    os.makedirs(_profiling_dir, exist_ok=True)

    df = benchmarking_naive_attention(
        atten_fn=atten_fn,
        configs=configs,
        context_length=256,
        batch_size=16,
        device=device,
        dtype=DTYPE,
        profiling_dir=_profiling_dir
    )

    # Plot 1: heads_num vs runtime (averaged over d_models)
    heads_avg = df.groupby("heads_num")[["forward_time", "backward_time"]].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(heads_avg["heads_num"], heads_avg["forward_time"], marker='o', label="Forward Time (avg)")
    plt.plot(heads_avg["heads_num"], heads_avg["backward_time"], marker='s', label="Backward Time (avg)")
    plt.xlabel("Number of Heads")
    plt.ylabel("Time (seconds)")
    plt.title(f"Runtime vs Number of Heads ({attn_name}, compiled={COMPILED})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(_profiling_dir, "heads_num_vs_runtime.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: d_models vs runtime (averaged over heads_num)
    dmodel_avg = df.groupby("d_model")[["forward_time", "backward_time"]].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(dmodel_avg["d_model"], dmodel_avg["forward_time"], marker='o', label="Forward Time (avg)")
    plt.plot(dmodel_avg["d_model"], dmodel_avg["backward_time"], marker='s', label="Backward Time (avg)")
    plt.xlabel("d_model")
    plt.ylabel("Time (seconds)")
    plt.title(f"Runtime vs d_model ({attn_name}, compiled={COMPILED})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(_profiling_dir, "dmodel_vs_runtime.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    df.to_csv(os.path.join(_profiling_dir, "benchmark_results.csv"), index=False)
    
    print(f"Results saved to {_profiling_dir}")


if __name__ == "__main__":
    main()  


