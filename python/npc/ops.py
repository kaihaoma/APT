
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

def allreduce(a: torch.Tensor):
    return torch.ops.npc.allreduce(a)
