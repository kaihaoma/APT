import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import *


def adapt(args, model: nn.Module, strategy: str, rank: int):
    device = next(model.parameters()).device
    # model adaptor
    if model.__class__.__name__ == "SAGE":
        if strategy == "DP":
            adapted_model = DGLSAGE(args=args, activation=model.activation)
        elif strategy == "NP":
            adapted_model = NPCSAGE(args=args, activation=model.activation)
        elif strategy == "SP":
            adapted_model = SPSAGE(args=args, activation=model.activation)
        elif strategy == "MP":
            adapted_model = MPSAGE(args=args, activation=model.activation)
            fc_self = torch.empty(args.num_hidden, args.input_dim, device=device)
            fc_neigh = torch.empty(args.num_hidden, args.input_dim, device=device)
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_uniform_(fc_self, gain=gain)
            nn.init.xavier_uniform_(fc_neigh, gain=gain)
            dist.broadcast(fc_self, 0)
            dist.broadcast(fc_neigh, 0)
            adapted_model.mp_layers.fc_self.weight.data = (
                fc_self[:, args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
                .clone()
                .detach()
                .cpu()
            )
            adapted_model.mp_layers.fc_neigh.weight.data = (
                fc_neigh[:, args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
                .clone()
                .detach()
                .cpu()
            )
        else:
            raise ValueError(f"Invalid parallism strategy: {strategy}")
    elif model.__class__.__name__ == "GAT":
        heads = [args.num_heads] * len(args.fan_out)
        if strategy == "DP":
            adapted_model = DGLGAT(args=args, heads=heads, activation=model.activation)
        elif strategy == "NP":
            adapted_model = NPCGAT(args=args, heads=heads, activation=model.activation)
        elif strategy == "SP":
            adapted_model = SPGAT(args=args, heads=heads, activation=model.activation)
        elif strategy == "MP":
            adapted_model = MPGAT(args=args, heads=heads, activation=F.relu)
            fc = torch.empty(args.num_hidden * heads[0], args.input_dim, device=device)
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_uniform_(fc, gain=gain)
            dist.broadcast(fc, 0)
            adapted_model.fc.weight.data = (
                fc[:, args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
                .clone()
                .detach()
                .cpu()
            )
        else:
            raise ValueError(f"Invalid system: {strategy}")
    elif model.__class__.__name__ == "GCN":
        if strategy == "DP":
            adapted_model = DGLGCN(args=args, activation=model.activation)
        elif strategy == "NP":
            adapted_model = NPCGCN(args=args, activation=model.activation)
        elif strategy == "SP":
            adapted_model = SPGCN(args=args, activation=model.activation)
        elif strategy == "MP":
            adapted_model = MPGCN(args=args, activation=model.activation)
            weight = torch.empty(args.input_dim, args.num_hidden, device=device)
            nn.init.xavier_uniform_(weight)
            dist.broadcast(weight, 0)
            adapted_model.mp_layers.weight.data = (
                weight[args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
                .clone()
                .detach()
                .cpu()
            )
        else:
            raise ValueError(f"Invalid parallism strategy: {strategy}")
    else:
        raise ValueError(f"Invalid model type: {model.__class__.__name__}")
    adapted_model = adapted_model.to(device)
    if args.world_size > 1 and strategy != "MP":
        adapted_model = DDP(
            adapted_model,
            device_ids=[device],
            output_device=device,
        )
    return adapted_model
