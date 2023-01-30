from .ops import *
import os

def _load_npc_library():
    package_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    so_path = os.path.join(package_path, 'libnpc.so')
    try:
        torch.classes.load_library(so_path)
    except Exception:
        raise ImportError("Cannot load NPC C++ library")

def _init(rank, world_size, shared_queue):
    if rank == 0:
        nccl_unique_id = torch.ops.npc.nccl_get_unique_id()
        for i in range(world_size):
            shared_queue.put(nccl_unique_id)
    nccl_unique_id = shared_queue.get()
    torch.ops.npc.init(rank, world_size, nccl_unique_id)

def init(rank, world_size, shared_queue):
    _load_npc_library()
    _init(rank, world_size, shared_queue)

