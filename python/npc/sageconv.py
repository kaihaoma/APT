import dgl
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
import dgl.function as fn

from .ops import SPFeatureShuffle


class SPSAGEConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        shuffle_with_dst=False,
        remote_stream=None,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(SPSAGEConv, self).__init__()

        self._in_src_feats = in_feats
        self._in_dst_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.shuffle_with_dst = shuffle_with_dst
        self._remote_stream = remote_stream
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, blocks, feat, fsi):
        # block2 fwd (VirtualNode, ori_neighbor)
        graph = blocks[0]

        if self.shuffle_with_dst:
            num_send_dst, num_recv_dst = fsi.num_dst
        else:
            num_dst = fsi.num_dst
        feat_all = self.feat_drop(feat)

        if self.shuffle_with_dst:
            h_dst = self.fc_self(feat_all[:num_recv_dst])
            feat_src = feat_all[num_recv_dst:]
        else:
            h_dst = self.fc_self(feat_all[:num_dst])
            feat_src = feat_all[num_dst:]

        with graph.local_scope():
            # Message Passing
            graph.srcdata["h"] = self.fc_neigh(feat_src)
            # print(f"[Note]graph:{graph}, graph_dev:{graph.device}")
            graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "neigh"))
            h_vir = graph.dstdata["neigh"]

        if self.shuffle_with_dst:
            shuffle_feat = SPFeatureShuffle.apply(fsi, torch.cat([h_dst, h_vir], dim=0))
        else:
            shuffle_feat = SPFeatureShuffle.apply(fsi, h_vir)

        # block1 fwd, (ori_node, VirtualNode)
        graph = blocks[1]
        with graph.local_scope():
            if self.shuffle_with_dst:
                graph.srcdata["h"] = shuffle_feat[num_send_dst:]
            else:
                graph.srcdata["h"] = shuffle_feat
            # Message Passing
            graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]

            if self.shuffle_with_dst:
                h_self = shuffle_feat[:num_send_dst]
            else:
                h_self = h_dst
            rst = h_self + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class Aggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph: Tuple[torch.Tensor], X: torch.Tensor, X_offset: torch.Tensor, out_offset: torch.Tensor) -> torch.Tensor:
        ctx.graph = graph
        all_coo_row, all_coo_col, coo_offset = graph
        out = torch.ops.npc.spmm_copy_u_sum(all_coo_row, all_coo_col, X, coo_offset, X_offset, out_offset)
        ctx.save_for_backward(X_offset, out_offset)
        return out

    @staticmethod
    def backward(ctx, dZ: torch.Tensor) -> torch.Tensor:
        all_coo_row, all_coo_col, coo_offset = ctx.graph
        X_offset, out_offset = ctx.saved_tensors
        dX = torch.ops.npc.spmm_copy_u_sum(all_coo_col, all_coo_row, dZ, coo_offset, out_offset, X_offset)
        return None, dX, None, None


class MPSAGEConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(MPSAGEConv, self).__init__()
        valid_aggre_types = {"mean"}
        if aggregator_type not in valid_aggre_types:
            raise KeyError("Invalid aggregator_type. Must be one of {}. " "But got {!r} instead.".format(valid_aggre_types, aggregator_type))

        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        (
            all_coo_row,
            all_coo_col,
            recv_frontier_size,
            recv_coo_size,
            recv_seed_size,
        ) = graph
        fanout = all_coo_row.numel() // recv_seed_size.sum().item()
        frontier_offset = torch.cat([torch.tensor([0], device=recv_frontier_size.device), torch.cumsum(recv_frontier_size, dim=0)])
        coo_offset = torch.cat([torch.tensor([0], device=recv_coo_size.device), torch.cumsum(recv_coo_size, dim=0)])
        seed_offset = torch.cat([torch.tensor([0], device=recv_seed_size.device), torch.cumsum(recv_seed_size, dim=0)])
        feat_src = self.feat_drop(feat)
        feat_dst = torch.cat([feat_src[frontier_offset[i] : frontier_offset[i] + recv_seed_size[i]] for i in range(recv_seed_size.numel())])

        h_self = feat_dst

        # Determine whether to apply linear transformation before message passing A(XW)
        lin_before_mp = self._in_src_feats > self._out_feats

        # Message Passing
        if self._aggre_type == "mean":
            h_inputs = self.fc_neigh(feat_src) if lin_before_mp else feat_src
            h_neigh = Aggregate.apply((all_coo_row, all_coo_col, coo_offset), h_inputs, frontier_offset, seed_offset)
            h_neigh = h_neigh / fanout
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)
        else:
            raise KeyError("Aggregator type {} not recognized.".format(self._aggre_type))

        rst = self.fc_self(h_self) + h_neigh

        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst
