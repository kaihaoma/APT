import npc
import torch
import torch.multiprocessing as mp
import atexit
import torch.distributed as dist
from queue import Queue

def kill_proc(p):
    try:
        p.terminate()
    except Exception:
        pass

def setup(rank, world_size):
    print(f'Start rank {rank}, world_size {world_size}.')
    torch.cuda.set_device(0)
    master_addr = "localhost"
    master_port = '12306'
    init_method = 'tcp://{master_addr}:{master_port}'.format(
        master_addr=master_addr, master_port=master_port)
    dist.init_process_group('nccl', init_method=init_method,
                            rank=rank, world_size=world_size)

def run(rank, world_size, shared_queue):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    npc.init(rank, world_size, shared_queue)

    device = torch.device(f'cuda:{rank}')
    a = torch.ones([3], dtype=torch.int64).to(device)

    a = npc.allreduce(a)
    print(a)

if __name__ == '__main__':
    nproc = 2
    processes = []
    q = mp.Queue()
    for i in range(nproc):
        p = mp.Process(target=run, args=(i, nproc, q))
        atexit.register(kill_proc, p)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()