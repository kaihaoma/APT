import dgl
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
import dgl.function as fn
from dgl._sparse_ops import _gsddmm, _gspmm
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


class DistributedEdgeSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, score, fsi):
        """Forward function.

        Pseudo-code:

        .. code:: python

            score = dgl.EData(g, score)
            score_max = score.dst_max()  # of type dgl.NData
            score = score - score_max  # edge_sub_dst, ret dgl.EData
            score_sum = score.dst_sum()  # of type dgl.NData
            out = score / score_sum    # edge_div_dst, ret dgl.EData
            return out.data
        """
        score_max = _gspmm(gidx, "copy_rhs", "max", None, score)[0]
        score = torch.exp(_gsddmm(gidx, "sub", score, score_max, "e", "v"))
        score_sum = _gspmm(gidx, "copy_rhs", "sum", None, score)[0]
        out = _gsddmm(gidx, "div", score, score_sum, "e", "v")
        ctx.backward_cache = gidx
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Backward function.

        Pseudo-code:

        .. code:: python

            g, out = ctx.backward_cache
            grad_out = dgl.EData(g, grad_out)
            out = dgl.EData(g, out)
            sds = out * grad_out  # type dgl.EData
            sds_sum = sds.dst_sum()  # type dgl.NData
            grad_score = sds - out * sds_sum  # multiple expressions
            return grad_score.data
        """
        gidx = ctx.backward_cache
        (out,) = ctx.saved_tensors
        sds = out * grad_out
        accum = _gspmm(gidx, "copy_rhs", "sum", None, sds)[0]
        grad_score = sds - _gsddmm(gidx, "mul", out, accum, "e", "v")
        return None, grad_score, None


def distribued_edge_softmax(graph, logits, fsi):
    return DistributedEdgeSoftmax.apply(graph._graph, logits, fsi)


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

    def forward(self, blocks, feat, fsi):
        # block2 fwd (VirtualNode, ori_neighbor)
        graph = blocks[0]
        num_dst = fsi.num_dst
        with graph.local_scope():
            src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
            h_src = self.feat_drop(feat)
            h_dst = h_src[:num_dst]
            feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
            feat_dst = feat_src[:num_dst]
            dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(distribued_edge_softmax(graph, e, fsi))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            h_vir = graph.dstdata["ft"]

        fsi.feat_dim = self._num_heads * self._out_feats
        shuffle_feat = SPFeatureShuffle.apply(fsi, h_vir)

        # block1 fwd, (ori_node, VirtualNode)
        graph = blocks[1]
        with graph.local_scope():
            graph.srcdata["h"] = shuffle_feat
            # Message Passing
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "neigh"))
            rst = graph.dstdata["neigh"]
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
