import npc
import torch
import torch.multiprocessing as mp
import atexit
import torch.distributed as dist
import utils


def kill_proc(p):
    try:
        p.terminate()
    except Exception:
        pass


def run(rank, world_size, shared_queue):
    world_size = 2
    utils.setup(rank, world_size)

    npc.init(rank=rank, world_size=world_size, shared_queue=shared_queue, init_mp=True)
    device = torch.device(f'cuda:{rank}')
    a = torch.ones([3], dtype=torch.int64).to(device)

    a = npc.allreduce(a)
    print(a)


if __name__ == '__main__':
    nproc = 2
    processes = []
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    mp.spawn(run,
             args=(nproc, q,),
             nprocs=nproc,
             join=True)
