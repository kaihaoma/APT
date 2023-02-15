
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import cast, Iterator, List, Optional, Tuple
import argparse
import dgl


def allreduce(a: torch.Tensor) -> torch.Tensor:
    return torch.ops.npc.allreduce(a)


def test(a: torch.Tensor, idx: int, val: float) -> None:
    torch.ops.npc.test(a, idx, val)


def cache_feats(node_feats: torch.Tensor, sorted_idx: torch.Tensor, num_cached_nodes: int) -> None:
    torch.ops.npc.cache_feats(node_feats, sorted_idx, num_cached_nodes)


def cache_graphs(num_local_nodes: int, num_graph_nodes: int, num_cached_nodes: int, sorted_idx: torch.Tensor, indptr: torch.Tensor, indices: torch.Tensor) -> None:
    torch.ops.npc.cache_graphs(num_local_nodes, num_graph_nodes, num_cached_nodes, sorted_idx, indptr, indices)


def load_subtensor(node_idx: torch.Tensor) -> torch.Tensor:
    return torch.ops.npc.load_subtensor(node_idx)


def load_partition(args: argparse.ArgumentParser, rank: int, shared_tensor_list: List[torch.Tensor]) -> None:
    # load graph partition
    graph, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)
    global_node_features, global_test_mask = shared_tensor_list
    # [NOTE] graph.ndata
    # 'orig_id', 'part_id', '_ID', 'inner_node'
    # [NOTE] graph.edata
    # '_ID', 'inner_edge','orig_id'
    # [NOTE] node_feats
    # '_N/features', '_N/labels', '_N/train_mask', '_N/test_mask', '_N/val_mask'

    # prepare global_node_features for sharing
    # local node: node belongs to this partition
    # graph node: local node and HALO (remoted 1-hop neighbor) node
    local_node_feats = node_feats['_N/features']
    local_train_mask = node_feats['_N/train_mask']
    inner_node_mask = graph.ndata['inner_node'].bool()
    global_nid = graph.ndata[dgl.NID]
    num_local_nodes, d1 = local_node_feats.shape
    assert d1 == args.input_dim
    num_graph_nodes = graph.num_nodes()

    print(f"[Note]Rk#{rank}: #local: {num_local_nodes}\t #graph:{num_graph_nodes}")

    # prepare graphnodes feat
    local_node_lid = torch.nonzero(graph.ndata['inner_node']).squeeze()
    assert num_local_nodes == local_node_lid.shape[0]
    global_test_mask[global_nid[local_node_lid]] += 1
    global_node_features[global_nid[local_node_lid]] = local_node_feats
    dist.barrier()

    # cache node feats
    in_degrees = graph.in_degrees()
    sorted_deg_idx = torch.argsort(in_degrees, descending=True)
    graph_node_feats = global_node_features[global_nid[sorted_deg_idx]]
    cache_feats(graph_node_feats, sorted_idx=sorted_deg_idx, num_cached_nodes=args.num_cached_nodes)

    # cache graphs
    indptr, indices, edge_ids = graph.adj_sparse('csr')
    local_nodes_sorted_deg_idx = sorted_deg_idx[inner_node_mask[sorted_deg_idx]]
    num_cached_graph_nodes = num_local_nodes

    cache_graphs(num_local_nodes, num_graph_nodes, num_cached_graph_nodes, local_nodes_sorted_deg_idx, indptr, indices)
