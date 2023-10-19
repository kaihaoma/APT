import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import numpy as np


def run_unidirection(rank, world_size, backend, data_size):
    def run_once():
        if rank == 0:
            tensor = torch.randn(data_size, dtype=torch.float32, device="cuda:0")

            t1_time = time.time()
            dist.send(tensor=tensor, dst=1)
            # Wait for all data to be sent
            torch.cuda.synchronize(device="cuda:0")
            return time.time() - t1_time
        else:
            tensor = torch.randn(data_size, dtype=torch.float32, device="cuda:1")

            t1_time = time.time()
            dist.recv(tensor=tensor, src=0)
            torch.cuda.synchronize(device="cuda:1")
            return time.time() - t1_time

    # execute a few rounds of warmup
    warmup_time = 0.0
    for _ in range(2):
        warmup_time += run_once()
    # measure runtime
    benchmark_time = []
    for _ in range(10):
        benchmark_time.append(run_once())

    print(
        f"Rank: {rank} | Backend: {backend} | Data Vol.: {(data_size * 4) / 1000} KB | Warmup: {(warmup_time):.3f} s | Max: {np.max(benchmark_time):.5f} s | Min: {np.min(benchmark_time):.5f} s | Avg: {np.mean(benchmark_time):.5f} s"
    )


def run_bidirection(rank, world_size, backend, data_size):
    def run_once():
        input = torch.randn(data_size, dtype=torch.float32, device=f"cuda:{rank}")
        output = torch.randn(data_size, dtype=torch.float32, device=f"cuda:{rank}")

        t1_time = time.time()
        dist.all_to_all_single(output, input)
        # Wait for all data to be sent
        torch.cuda.synchronize(device=f"cuda:{rank}")
        return time.time() - t1_time

    # execute a few rounds of warmup
    warmup_time = 0.0
    for _ in range(2):
        warmup_time += run_once()
    # measure runtime
    benchmark_time = []
    for _ in range(10):
        benchmark_time.append(run_once())

    print(
        f"Rank: {rank} | Backend: {backend} | Data Vol.: {(data_size * 4) / 1000} KB | Warmup: {(warmup_time):.3f} s | Max: {np.max(benchmark_time):.5f} s | Min: {np.min(benchmark_time):.5f} s | Avg: {np.mean(benchmark_time):.5f} s"
    )


def init_process(rank, size, fn, backend, data_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, backend, data_size)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    world_size = 2
    func = run_unidirection
    
    print(f"World size: {world_size}")

    for data_size in [1000000, 10000000, 100000000, 500000000, 1000000000]:
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=init_process, args=(rank, world_size, func, "nccl", data_size))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
