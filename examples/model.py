import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F


class SAGE(nn.Module):
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


class GAT(nn.Module):
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
                        n_hidden * heads[i - 1],
                        n_hidden,
                        heads[i],
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


class GCN(nn.Module):
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
            f"[Note]DGL GCN: fanout: {fan_out}\t in: {in_feats}, hid: {n_hidden}, out: {n_classes}"
        )
        self.fan_out = fan_out
        self.n_layers = len(fan_out)
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            self.layers.append(dglnn.GraphConv(in_feats, n_hidden, norm="none"))
            for i in range(1, self.n_layers - 1):
                self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
            self.layers.append(dglnn.GraphConv(n_hidden, n_classes))
        else:
            self.layers.append(dglnn.GraphConv(in_feats, n_classes))
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
