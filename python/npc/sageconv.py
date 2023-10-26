import dgl
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple


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
        return None, dX


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
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise KeyError("Invalid aggregator_type. Must be one of {}. " "But got {!r} instead.".format(valid_aggre_types, aggregator_type))

        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
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
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
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
