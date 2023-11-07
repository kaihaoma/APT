import dgl
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
import dgl.function as fn
from dgl.ops import edge_softmax
import numpy as np

from .ops import SPFeatureShuffle, MPFeatureShuffle, MPFeatureShuffleInfo


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class SPGATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(SPGATConv, self).__init__()

        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=bias)
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, blocks, feat, fsi):
        # block2 fwd (VirtualNode, ori_neighbor)
        graph = blocks[0]
        num_recv_dst = fsi.num_recv_dst

        with graph.local_scope():
            src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
            h_src = self.feat_drop(feat)
            h_dst = h_src[:num_recv_dst]
            feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
            feat_dst = feat_src[:num_recv_dst]
            dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            # TODO: MP all reduce
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            # TODO: SP all reduce
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            return rst


# class ScatterUAddV(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, graph: Tuple[torch.Tensor], X: torch.Tensor, Y: torch.Tensor, X_offset: torch.Tensor, Y_offset: torch.Tensor) -> torch.Tensor:
#         ctx.graph = graph
#         all_coo_row, all_coo_col, coo_offset = graph
#         out = torch.ops.npc.sddmm_u_add_v(all_coo_row, all_coo_col, X, Y, coo_offset, X_offset, Y_offset)
#         ctx.save_for_backward(X_offset, Y_offset)
#         return out

#     @staticmethod
#     def backward(ctx, dZ: torch.Tensor) -> torch.Tensor:
#         all_coo_row, all_coo_col, coo_offset = ctx.graph
#         X_offset, Y_offset = ctx.saved_tensors
#         dX = torch.ops.npc.spmm_copy_e_sum(all_coo_col, all_coo_row, dZ, coo_offset, X_offset)
#         dY = torch.ops.npc.spmm_copy_e_sum(all_coo_row, all_coo_col, dZ, coo_offset, Y_offset)
#         return None, dX, dY, None, None


# class AggregateUMulESum(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, graph: Tuple[torch.Tensor], X: torch.Tensor, E: torch.Tensor, X_offset: torch.Tensor, out_offset: torch.Tensor) -> torch.Tensor:
#         ctx.graph = graph
#         all_coo_row, all_coo_col, coo_offset = graph
#         out = torch.ops.npc.spmm_u_mul_e_sum(all_coo_row, all_coo_col, X, E, coo_offset, X_offset, out_offset)
#         ctx.save_for_backward(X, E, X_offset, out_offset)
#         return out

#     @staticmethod
#     def backward(ctx, dZ: torch.Tensor) -> torch.Tensor:
#         all_coo_row, all_coo_col, coo_offset = ctx.graph
#         X, E, X_offset, out_offset = ctx.saved_tensors
#         dX = torch.ops.npc.spmm_u_mul_e_sum(all_coo_col, all_coo_row, dZ, E, coo_offset, out_offset, X_offset)
#         dE = torch.ops.npc.sddmm_u_mul_v(all_coo_row, all_coo_col, X, dZ, coo_offset, X_offset, out_offset)
#         return None, dX, dE, None, None


class MPGATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(MPGATConv, self).__init__()

        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            raise NotImplementedError
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, graph_tuple, feat, fsi: MPFeatureShuffleInfo, edge_weight=None):
        (
            all_coo_row,
            all_coo_col,
            send_frontier_size,
            recv_frontier_size,
            recv_coo_size,
            recv_seed_size,
            graph,
        ) = graph_tuple
        src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
        h_src = self.feat_drop(feat)
        feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)

        fsi.feat_dim = self._num_heads * self._out_feats
        fsi.send_size = send_frontier_size.to("cpu")
        fsi.recv_size = recv_frontier_size.to("cpu")
        feat_src = MPFeatureShuffle.apply(fsi, feat_src)
        feat_dst = feat_src[: graph.number_of_dst_nodes()]
        dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

        with graph.local_scope():
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(1, self._num_heads, 1).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                raise NotImplementedError
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst

        # (
        #     all_coo_row,
        #     all_coo_col,
        #     recv_frontier_size,
        #     recv_coo_size,
        #     recv_seed_size,
        # ) = graph
        # # fanout = all_coo_row.numel() // recv_seed_size.sum().item()
        # frontier_offset = torch.cat([torch.tensor([0], device=recv_frontier_size.device), torch.cumsum(recv_frontier_size, dim=0)])
        # coo_offset = torch.cat([torch.tensor([0], device=recv_coo_size.device), torch.cumsum(recv_coo_size, dim=0)])
        # seed_offset = torch.cat([torch.tensor([0], device=recv_seed_size.device), torch.cumsum(recv_seed_size, dim=0)])

        # src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
        # dst_prefix_shape = (seed_offset[-1].item(),) + dst_prefix_shape[1:]
        # h_src = self.feat_drop(feat)
        # h_dst = torch.cat([h_src[frontier_offset[i] : frontier_offset[i] + recv_seed_size[i]] for i in range(recv_seed_size.numel())])
        # feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
        # feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)

        # el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        # er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        # h_e = ScatterUAddV.apply((all_coo_row, all_coo_col, coo_offset), el, er, frontier_offset, seed_offset)
        # e = self.leaky_relu(h_e)
        # # compute softmax
        # exp_e = xxx
        # # TODO: MP all reduce
        # a_e = self.attn_drop(edge_softmax(graph, e))
        # # message passing
        # rst = AggregateUMulESum.apply((all_coo_row, all_coo_col, coo_offset), feat_src, a_e, frontier_offset, seed_offset)
        # # residual
        # if self.res_fc is not None:
        #     # Use -1 rather than self._num_heads to handle broadcasting
        #     resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
        #     rst = rst + resval
        # # bias
        # if self.has_explicit_bias:
        #     rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
        # # activation
        # if self.activation:
        #     rst = self.activation(rst)
        # return rst
