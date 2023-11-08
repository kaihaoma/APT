import dgl
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
import dgl.function as fn
import numpy as np

from .ops import SPFeatureShuffle


class SPGraphConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(SPGraphConv, self).__init__()

        if norm not in ("none", "both", "right", "left"):
            raise KeyError('Invalid norm value. Must be either "none", "both", "right" or "left".' ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, blocks, feat, fsi):
        # block2 fwd (VirtualNode, ori_neighbor)
        graph = blocks[0]
        num_recv_dst = fsi.num_recv_dst
        with graph.local_scope():
            feat_dst = feat[:num_recv_dst]
            feat_src = feat[num_recv_dst:]
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm
            # mult W first to reduce the feature size for aggregation.
            h_dst = torch.matmul(feat_dst, self.weight)
            feat_src = torch.matmul(feat_src, self.weight)
            graph.srcdata["h"] = feat_src
            graph.update_all(fn.copy_u("h", "m"), fn.sum(msg="m", out="h"))
            h_vir = graph.dstdata["h"]

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                h_vir = h_vir * norm

        # shuffle_vir, shuffle_dst = npc.SPFeatureShuffle.apply(fsi, h_vir, h_dst)
        shuffle_feat = SPFeatureShuffle.apply(fsi, torch.cat([h_dst, h_vir], dim=0))
        # block1 fwd, (ori_node, VirtualNode)
        graph = blocks[1]
        num_send_dst = fsi.num_send_dst
        with graph.local_scope():
            graph.srcdata["h"] = shuffle_feat[num_send_dst:]
            feat_dst = shuffle_feat[:num_send_dst]
            # Message Passing
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            rst = graph.dstdata["h"]

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)
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


class MPGraphConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(MPGraphConv, self).__init__()
        if norm not in ("none", "both", "right", "left"):
            raise KeyError('Invalid norm value. Must be either "none", "both", "right" or "left".' ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat):
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
        if isinstance(feat, tuple):
            feat_src = feat[0]
            # feat_dst = feat[1]
        else:
            feat_src = feat
            # feat_dst = torch.cat([feat_src[frontier_offset[i] : frontier_offset[i] + recv_seed_size[i]] for i in range(recv_seed_size.numel())])

        if self._norm in ["left", "both"]:
            if self._norm == "both":
                norm = np.power(fanout, -0.5)
            else:
                norm = 1.0 / fanout
            feat_src = feat_src * norm

        feat_src = torch.matmul(feat_src, self.weight)
        rst = Aggregate.apply((all_coo_row, all_coo_col, coo_offset), feat_src, frontier_offset, seed_offset)

        if self._norm in ["right", "both"]:
            if self._norm == "both":
                norm = np.power(fanout, -0.5)
            else:
                norm = 1.0 / fanout
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)
        return rst
