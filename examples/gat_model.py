import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import npc
import time
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class NPCGAT(nn.Module):
    def __init__(self, args, heads, activation=F.elu):
        super().__init__()

        self.init(
            args.fan_out,
            args.input_dim,
            args.num_hidden,
            args.num_classes,
            heads,
            activation,
            args.dropout,
        )

    def init(self, fan_out, in_feats, n_hidden, n_classes, heads, activation, dropout):
        self.fan_out = fan_out
        self.n_layers = len(fan_out)
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            self.layers.append(
                dglnn.GATConv(
                    in_feats,
                    n_hidden,
                    heads[0],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=activation,
                )
            )
            for i in range(1, self.n_layers - 1):
                self.layers.append(
                    dglnn.GATConv(
                        n_hidden * heads[i],
                        n_hidden,
                        heads[i + 1],
                        feat_drop=dropout,
                        attn_drop=dropout,
                        activation=activation,
                    )
                )
            self.layers.append(
                dglnn.GATConv(
                    n_hidden * heads[-2],
                    n_classes,
                    heads[-1],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=None,
                )
            )
        else:
            self.layers.append(
                dglnn.GATConv(
                    in_feats,
                    n_classes,
                    heads[0],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=None,
                )
            )
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
        h = input_feats
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l == 0:
                h = npc.NPFeatureShuffle.apply(fsi, h)
            if l == self.n_layers - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h  # event


class DGLGAT(nn.Module):
    def __init__(self, args, heads, activation=F.elu):
        super().__init__()
        self.init(
            args.fan_out,
            args.input_dim,
            args.num_hidden,
            args.num_classes,
            heads,
            activation,
            args.dropout,
        )

    def init(self, fan_out, in_feats, n_hidden, n_classes, heads, activation, dropout):
        print(f"[Note]DGL SAGE: fanout: {fan_out}\t in: {in_feats}, hid: {n_hidden}, out: {n_classes}")
        self.fan_out = fan_out
        self.n_layers = len(fan_out)
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            self.layers.append(
                dglnn.GATConv(
                    in_feats,
                    n_hidden,
                    heads[0],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=activation,
                )
            )
            for i in range(1, self.n_layers - 1):
                self.layers.append(
                    dglnn.GATConv(
                        n_hidden * heads[i],
                        n_hidden,
                        heads[i + 1],
                        feat_drop=dropout,
                        attn_drop=dropout,
                        activation=activation,
                    )
                )
            self.layers.append(
                dglnn.GATConv(
                    n_hidden * heads[-2],
                    n_classes,
                    heads[-1],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=None,
                )
            )
        else:
            self.layers.append(
                dglnn.GATConv(
                    in_feats,
                    n_classes,
                    heads[0],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=None,
                )
            )
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
            if l == self.n_layers - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


class SPGAT(nn.Module):
    def __init__(self, args, heads, activation=F.elu):
        super().__init__()
        self.init(
            args.fan_out,
            args.input_dim,
            args.num_hidden,
            args.num_classes,
            heads,
            activation,
            args.dropout,
        )

    def init(self, fan_out, in_feats, n_hidden, n_classes, heads, activation, dropout):
        self.fan_out = fan_out
        self.n_layers = len(fan_out)
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            self.layers.append(
                npc.SPGATConv(
                    in_feats,
                    n_hidden,
                    heads[0],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=activation,
                )
            )
            for i in range(1, self.n_layers - 1):
                self.layers.append(
                    dglnn.GATConv(
                        n_hidden * heads[i],
                        n_hidden,
                        heads[i + 1],
                        feat_drop=dropout,
                        attn_drop=dropout,
                        activation=activation,
                    )
                )
            self.layers.append(
                dglnn.GATConv(
                    n_hidden * heads[-2],
                    n_classes,
                    heads[-1],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=None,
                )
            )
        else:
            self.layers.append(
                dglnn.GATConv(
                    in_feats,
                    n_classes,
                    heads[0],
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=None,
                )
            )
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
        # layer 1~n-1
        for l, (layer, block) in enumerate(zip(self.layers[1:], blocks[2:])):
            h = layer(block, h)
            if l == self.n_layers - 2:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


class SimpleConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# MODEL PARA
# wrap Data Parallel
class MPGAT(nn.Module):
    def __init__(self, args, heads, activation=F.elu):
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
            heads,
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
        heads,
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

        self.layers = nn.ModuleList()

        if self.n_layers > 1:
            # first mp layer
            self.mp_layers = npc.MPGATConv(
                in_feats,
                n_hidden,
                heads[0],
                feat_drop=dropout,
                attn_drop=dropout,
                activation=activation,
            ).to(self.device)
            # ddp
            ddp_config = SimpleConfig(
                fan_out=fan_out[:-1],
                input_dim=n_hidden * heads[0],
                num_hidden=n_hidden,
                num_classes=n_classes,
                dropout=dropout,
            )
            self.ddp_modules = DDP(
                DGLGAT(ddp_config, heads[1:], F.elu).to(self.device),
                device_ids=[self.device],
                output_device=self.device,
            )

        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, sampling_result):
        (
            blocks,
            x,
            fsi,
        ) = sampling_result

        h = x
        # fir mp layer
        h = self.mp_layers(blocks[0], h, fsi)
        h = h.flatten(1)

        h = self.ddp_modules((blocks[1:], h))
        h = h.mean(1)
        return h
