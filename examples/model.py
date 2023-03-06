import torch
import torch.distributed as dist
import torch.nn as nn
import dgl.nn as dglnn
from typing import List
import npc


class SAGE(nn.Module):
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
    def forward(self, blocks, input_feats, fsi):
        h = input_feats

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l == 0:
                h = npc.FeatureShuffle.apply(fsi, h)
            if l != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
