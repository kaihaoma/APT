
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

def allreduce(a: torch.Tensor):
    return torch.ops.npc.allreduce(a)

def test(a: torch.Tensor, idx: int, val: float):
    return torch.ops.npc.test(a, idx, val)

def cache_feats(node_feats: torch.Tensor, sorted_idx: torch.Tensor, num_cached_nodes: int):
    return torch.ops.npc.cache_feats(node_feats, sorted_idx, num_cached_nodes)
