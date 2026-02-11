import os
import time
from typing import List
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--GPU', action='store_true', help='use GPU if available')
parser.add_argument('--tensor_mb', type=int, help='number of MB of a data tensor', default=3)
parser.add_argument('--cpu_nproc', type=int, help='number of CPU processes to spawn', default=1)
args = parser.parse_args()

data_mb = 5
BACKEND = "nccl" if args.GPU and torch.cuda.is_available() else "gloo"
TENSOR_DIM = data_mb * 256 * 1024  # 4 bytes * 256 * 1024 = 1 MB
CPU_NPROC = args.cpu_nproc

def set_dist_env(rank, world_size):
    # Set the master node address and port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Initialize the the proc - `rank` into the group
    init_process_group(
        backend=BACKEND, 
        rank=rank, 
        world_size=world_size
    )

def time_tensor_all_reduce(
        rank: int, world_size: int, device: str,
        input_size: int, 
        warm_up_iter: int, timed_iter: int
    ) -> None:
    """1. Setting up process group"""
    set_dist_env(rank, world_size)
    DEVICE = f"cuda:{rank}" if BACKEND == "nccl" else "cpu"

    """2. Start benchmarking"""
    #   Initialize random data tensor on each process.
    input_tensor = torch.randint(0, 10, (input_size,), device=DEVICE, dtype=torch.float32)
    data = input_tensor.clone()

    #   Warm-up iterations
    for _ in range(warm_up_iter):
        torch.distributed.all_reduce(data, async_op=False)

    print(f"==========      Rank {rank} completed warmup.     ===========")
    if DEVICE == f"cuda:{rank}":
            torch.cuda.synchronize()
    
    print(f"=========== Rank {rank} starting timed iterations ===========")
    #   Timed iterations
    time_results = []
    for _ in range(timed_iter):
        start_time = time.perf_counter()
        torch.distributed.all_reduce(data, async_op=False)
        # Sync
        if DEVICE == f"cuda:{rank}":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        # Validate the result by comparing 0 and last rank
        if rank == 0 or rank == world_size - 1:
            print(f"{rank} first 5: {[int(i) for i in data[:5].to('cpu').tolist()]}")
        time_results.append(end_time - start_time)
    
    #   All-reduce the timing results across all processes
    time_tensor = torch.tensor(time_results, device=DEVICE)
    torch.distributed.all_reduce(time_tensor, async_op=False)
    if rank == 0:
        avg_time = sum(time_tensor.cpu().tolist())/ (world_size * timed_iter)
        print(f"Size {input_size/256/1024} MB -> Average times: {avg_time}")

    """3. Clean up process group"""
    destroy_process_group()



if __name__ == "__main__":
    time_results = []
    N_PROC = (
        torch.cuda.device_count() if BACKEND == "nccl" # Return the number of GPUs available
        else args.cpu_nproc # Use the custom CPU proc number
    )
    DEVICE = "cuda" if BACKEND == "nccl" else "cpu"
    # Spawn Processes
    """
    Initialize `nprocs` processes, rank is assigned automatically by `mp.spawn`.

        - All proc running: `fn=time_tensor_all_reduce`.
        - proc's RANK is pass automatically by `mp.spawn` to `fn` as the first argument.
        - Inside the `time_tensor_all_reduce` method, we initialize the distributed env using the enforced passed `RANK` and `args`
    """
    mp.spawn(
        fn=time_tensor_all_reduce,
        args=(N_PROC, DEVICE, TENSOR_DIM, 5, 10),
        nprocs=N_PROC, # Number of processes to spawn
        join=True,
    )
