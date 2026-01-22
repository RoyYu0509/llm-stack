import pandas as pd
import torch
import cs336_basics
import itertools
from cs336_basics.transfromer.scaled_dot_prod_attention import scaled_dot_product_attention
import timeit
from cs336_basics.transfromer.multiheads_attention import MultiHeadsAttention
import argparse
from cs336_basics.train.optimizer import AdamW
import torch.cuda.nvtx as nvtx
import os
import tqdm

# Create directory for profiling results
os.makedirs("./profiling_attention", exist_ok=True)

parser = argparse.ArgumentParser(description="Benchmarking Attention Mechanism")
parser.add_argument("--DTYPE", type=str, default="float32", help="Data type for the tensors")
parser.add_argument("--PROFILE_FORWARD_MEMORY", type=bool, default=False, help="Whether to perform memory profiling during forward pass.")
parser.add_argument("--PROFILE_BACKWARD_MEMORY", type=bool, default=False, help="Whether to perform memory profiling during backward pass.")
args = parser.parse_args()
DTYPE = getattr(torch, args.DTYPE)
PROFILE_FORWARD_MEMORY = args.PROFILE_FORWARD_MEMORY
PROFILE_BACKWARD_MEMORY = args.PROFILE_BACKWARD_MEMORY


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def _sync_device():
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmarking_naive_attention(
        heads_num:list, d_models:list, 
        context_length:int=256, batch_size:int=16, 
        device:torch.device=torch.device("cuda"), dtype:torch.dtype=DTYPE
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
    for head, d_model in itertools.product(heads_num, d_models):
        print(f"Benchmarking MultiHead Attention: heads={head}, d_model={d_model}")
        mha = MultiHeadsAttention(d_model, head, device=device, dtype=dtype)
        opt = AdamW(mha.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        forward_times = 0
        backward_times = 0
        
        # Warm-up
        nvtx.range_push("Warm-up")
        for _ in range(10):  # Warm-up
            x = torch.randn((batch_size, context_length, d_model), device=device, dtype=dtype, requires_grad=True)
            y = mha._multiHead(x, token_positions=torch.arange(context_length, device=device, dtype=torch.long))
            y.sum().backward()
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
            forward_time = timeit.default_timer() - start_time
            _sync_device()

            if PROFILE_FORWARD_MEMORY:
                torch.cuda.memory._dump_snapshot(f"./profiling_attention/head{head}_dmodel{d_model}_mem_f.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)


            if PROFILE_BACKWARD_MEMORY:
                torch.cuda.memory._record_memory_history(max_entries=25_000)
            # backward
            _sync_device()
            start_time = timeit.default_timer()
            with nvtx.range("backward_pass"):
                y.sum().backward()
            backward_time = timeit.default_timer() - start_time
            _sync_device()
            if PROFILE_BACKWARD_MEMORY:
                torch.cuda.memory._dump_snapshot(f"./profiling_attention/head{head}_dmodel{d_model}_mem_b.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)

            # Do not update parameters
            # opt.step()
            forward_times += forward_time
            backward_times += backward_time
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
    heads_num = [32, 64]
    d_models = [1024, 4096, 8192]
    print("Starting...")

    df = benchmarking_naive_attention(
        heads_num=heads_num,
        d_models=d_models,
        context_length=256,
        batch_size=64,
        device=device,
        dtype=DTYPE
    )

    # Save to csv
    df.to_csv("./profiling_attention/benchmark_attention.csv", index=False)
    print("Benchmarking results saved to benchmark_attention.csv")        


if __name__ == "__main__":
    main()  


