import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..ops import NPFeatureShuffle, MPFeatureShuffle
from .sageconv import *


class NPCSAGE(nn.Module):
    def __init__(
        self,
        args,
        activation=torch.relu,
    ):
        super().__init__()

        self.init(
            args.fan_out,
            args.input_dim,
            args.num_hidden,
            args.num_classes,
            activation,
            args.dropout,
        )

    def init(
        self,
        fan_out,
        in_feats,
        n_hidden,
        n_classes,
        activation,
        dropout,
    ):
        self.fan_out = fan_out
        self.n_layers = len(fan_out)
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
            for i in range(1, self.n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    # blocks: sampled blocks

    # x: input features
    def forward(self, loading_result):
        (
            blocks,
            input_feats,
            fsi,
        ) = loading_result
        # event = torch.cuda.Event(enable_timing=True)
        h = input_feats
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l == 0:
                # event.record()
                h = h[fsi.inverse_idx]
                h = NPFeatureShuffle.apply(fsi, h)
            if l != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h  # event


class DGLSAGE(nn.Module):
    def __init__(self, args, activation=torch.relu):
        super().__init__()
        self.init(
            args.fan_out,
            args.input_dim,
            args.num_hidden,
            args.num_classes,
            activation,
            args.dropout,
        )

    def init(
        self,
        fan_out,
        in_feats,
        n_hidden,
        n_classes,
        activation,
        dropout,
    ):
        print(
            f"[Note]DGL SAGE: fanout: {fan_out}\t in: {in_feats}, hid: {n_hidden}, out: {n_classes}"
        )
        self.fan_out = fan_out
        self.n_layers = len(fan_out)
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
            for i in range(1, self.n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, sampling_result):
        (
            blocks,
            x,
        ) = sampling_result
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class SPSAGE(nn.Module):
    def __init__(self, args, activation=torch.relu):
        super().__init__()
        self.init(
            args.fan_out,
            args.input_dim,
            args.num_hidden,
            args.num_classes,
            activation,
            args.dropout,
            args.shuffle_with_dst,
        )

    def init(
        self,
        fan_out,
        in_feats,
        n_hidden,
        n_classes,
        activation,
        dropout,
        shuffle_with_dst,
    ):
        self.fan_out = fan_out
        self.n_layers = len(fan_out)
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            self.layers.append(
                SPSAGEConv(
                    in_feats,
                    n_hidden,
                    "mean",
                    shuffle_with_dst=shuffle_with_dst,
                )
            )
            for i in range(1, self.n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, sampling_result):
        (
            blocks,
            input_feats,
            fsi,
        ) = sampling_result

        h = input_feats
        # layer 0
        h = self.layers[0](blocks[:2], h, fsi)
        h = self.activation(h)
        h = self.dropout(h)
        # layer 1~n-1
        for l, (layer, block) in enumerate(zip(self.layers[1:], blocks[2:])):
            h = layer(block, h)
            if l != self.n_layers - 2:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class SimpleConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# MODEL PARA
# wrap Data Parallel
class MPDDP(nn.Module):
    def __init__(self, args, activation=torch.relu):
        super().__init__()
        self.init(
            args.fan_out,
            args.input_dim,
            args.num_hidden,
            args.num_classes,
            activation,
            args.dropout,
        )

    def init(
        self,
        fan_out,
        in_feats,
        n_hidden,
        n_classes,
        activation,
        dropout,
    ):
        self.fan_out = fan_out
        self.n_layers = len(fan_out)
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.bias = nn.Parameter(torch.Tensor(in_feats))
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
            for i in range(1, self.n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, sampling_result):
        (
            blocks,
            x,
        ) = sampling_result
        h = x
        h = h + self.bias
        h = self.activation(h)
        h = self.dropout(h)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class MPSAGE(nn.Module):
    def __init__(self, args, activation=torch.relu):
        super().__init__()
        self.init(
            args.fan_out,
            args.input_dim,
            args.mp_input_dim_list,
            args.num_hidden,
            args.num_classes,
            args.rank,
            args.world_size,
            args.device,
            activation,
            args.dropout,
        )

    def init(
        self,
        fan_out,
        in_feats,
        in_feats_list,
        n_hidden,
        n_classes,
        rank,
        n_workers,
        device,
        activation,
        dropout,
    ):
        self.fan_out = fan_out
        self.n_layers = len(fan_out)
        self.in_feats = in_feats
        self.in_feats_list = in_feats_list
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.rank = rank
        self.n_workers = n_workers
        self.device = device

        if self.n_layers > 1:
            # first mp layer
            self.mp_layers = MPSAGEConv(
                self.in_feats_list[self.rank],
                self.n_hidden,
                aggregator_type="mean",
                bias=False,
            ).to(self.device)
            # ddp
            ddp_config = SimpleConfig(
                fan_out=fan_out[:-1],
                input_dim=n_hidden,
                num_hidden=n_hidden,
                num_classes=n_classes,
                dropout=dropout,
            )
            self.ddp_modules = DDP(
                MPDDP(ddp_config, activation).to(self.device),
                device_ids=[self.device],
                output_device=self.device,
            )

        else:
            raise NotImplementedError

    def forward(self, sampling_result):
        (
            blocks,
            x,
            fsi,
        ) = sampling_result

        h = x
        # fir mp layer
        h = self.mp_layers(blocks[0], h)
        # custom shuffle
        h = MPFeatureShuffle.apply(fsi, h)

        h = self.ddp_modules((blocks[1:], h))
        return h
