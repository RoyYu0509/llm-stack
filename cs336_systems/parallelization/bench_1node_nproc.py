import os
import time
from typing import List
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nproc', type=int, default=4, help='number of processes to spawn')
parser.add_argument('--GPU', action='store_true', help='use GPU if available')
parser.add_argument('--tensor_mb', type=int, help='number of MB of a data tensor', default=3)
args = parser.parse_args()
BACKEND = "nccl" if args.GPU and torch.cuda.is_available() else "gloo"
TENSOR_DIM = args.tensor_mb * 256 * 1024  # 4 bytes * 256 * 1024 = 1 MB
N_PROC = args.nproc


def setup(rank: int, world_size: int) -> None:
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)


def time_tensor_all_reduce(
        rank: int, world_size: int, 
        input_size: int, 
        warm_up_iter: int, timed_iter: int
    ) -> None:
    # Initialize the distributed environment.
    setup(rank, world_size)
    
    # Initialize random data tensor on each process.
    DEVICE = f"cuda:{rank}" if BACKEND == "nccl" else "cpu"
    input_tensor = torch.randint(0, 10, (input_size,), device=DEVICE, dtype=torch.float32)
    data = input_tensor.clone()

    # Warm-up iterations
    for _ in range(warm_up_iter):
        dist.all_reduce(data, async_op=False)
        print(f"rank {rank} data (after all-reduce): {data}")

    print(f"=========== Rank {rank} completed warm-up. ===========")
    if DEVICE == f"cuda:{rank}":
            torch.cuda.synchronize()
    
    print(f"=========== Rank {rank} starting timed iterations ===========")
    # Timed iterations
    time_results = []
    for _ in range(timed_iter):
        start_time = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        # Sync
        if DEVICE == f"cuda:{rank}":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        # Validate the result by comparing 0 and last rank
        if rank == 0 or rank == world_size - 1:
            print(f"{rank} first 5: {[int(i) for i in data[:5].to('cpu').tolist()]}")
        time_results.append(end_time - start_time)
    
    # All-reduce the timing results across all processes
    time_tensor = torch.tensor(time_results, device=DEVICE)
    dist.all_reduce(time_tensor, async_op=False)
    if rank == 0:
        avg_time = sum(time_tensor.cpu().tolist())/ (world_size * timed_iter)
        print(f"Size {input_size/256/1024} MB -> Average times: {avg_time}")


if __name__ == "__main__":
    time_results = []
    mp.spawn(
        fn=time_tensor_all_reduce,
        args=(N_PROC, TENSOR_DIM, 5, 10),
        nprocs=N_PROC,
        join=True,
    )
    # All-reduce the timing results across all processes

