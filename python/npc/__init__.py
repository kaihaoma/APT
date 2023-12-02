from .ops import *
from .sampler import *
from .sageconv import *
from .gatconv import *
from .graphconv import *
import os
import time


def _load_npc_library():
    package_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    so_path = os.path.join(package_path, "libnpc.so")
    try:
        torch.classes.load_library(so_path)
        print(f"[Note]load so from {so_path}")
    except Exception:
        raise ImportError("Cannot load NPC C++ library")


def _init(rank, local_rank, world_size, shared_queue):
    if rank == 0:
        nccl_unique_id = torch.ops.npc.nccl_get_unique_id()
        for i in range(world_size):
            shared_queue.put(nccl_unique_id)
    dist.barrier()
    nccl_unique_id = shared_queue.get()

    torch.ops.npc.init(rank, local_rank, world_size, nccl_unique_id)


def _init_broadcast(rank, local_rank, world_size, node_size, device, num_nccl_comms):
    print(f"[Note]No#[{rank}\t {local_rank}\t {world_size}] device:{device}")
    nccl_unique_id_list = []
    for i in range(num_nccl_comms):
        nccl_unique_id = torch.ops.npc.nccl_get_unique_id().to(device)
        dist.broadcast(nccl_unique_id, 0)
        nccl_unique_id = nccl_unique_id.to("cpu")
        nccl_unique_id_list.append(nccl_unique_id)

    nccl_unique_id_list = torch.vstack(nccl_unique_id_list)
    torch.ops.npc.init(
        rank,
        local_rank,
        world_size,
        nccl_unique_id_list,
        node_size,
    )


def init(rank, local_rank, world_size, node_size, num_nccl_comms=2, device=None, init_mp=True):
    _load_npc_library()
    if init_mp:
        # _init(rank, world_size, shared_queue)
        _init_broadcast(rank, local_rank, world_size, node_size, device, num_nccl_comms)
