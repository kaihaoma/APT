import torch
import torch.nn as nn
import dgl.nn as dglnn
import npc
import torch.nn.functional as F


class NPCSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    # blocks: sampled blocks

    # x: input features
    def forward(self, loading_result):
        blocks, input_feats, fsi, = loading_result
        h = input_feats
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l == 0:
                h = npc.FeatureShuffle.apply(fsi, h)
            if l != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class DGLSAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
            for i in range(1, num_layers - 1):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, sampling_result):
        blocks, x, = sampling_result
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
