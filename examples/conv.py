import torch
from torch import nn
from torch.nn import functional as F
import npc
import dgl.function as fn


class EXPSAGEConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        remote_stream=None,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(EXPSAGEConv, self).__init__()

        self._in_src_feats = in_feats
        self._in_dst_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
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
        num_dst = fsi.num_dst
        with graph.local_scope():
            feat_dst = feat[:num_dst]
            h_dst = self.fc_self(feat_dst)

            feat_src = self.feat_drop(feat[num_dst:])
            # Message Passing
            graph.srcdata["h"] = self.fc_neigh(feat_src)
            graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "neigh"))
            h_vir = graph.dstdata["neigh"]

        shuffle_feat = npc.SPFeatureShuffle.apply(fsi, h_vir)
        # block1 fwd, (ori_node, VirtualNode)
        graph = blocks[1]

        with graph.local_scope():
            graph.srcdata["h"] = shuffle_feat
            # Message Passing
            graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]

            h_self = h_dst
            rst = h_self + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst

    def ref_forward(
        self,
        graph,
        feat,
        fsi=None,
    ):
        with graph.local_scope():
            feat_src = self.feat_drop(feat)
            # if graph.is_block:
            feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = self.fc_neigh(feat_src)
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                h_self = feat_dst
                rst = self.fc_self(h_self) + h_neigh
            else:
                raise NotImplementedError

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
