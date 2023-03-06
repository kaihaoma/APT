
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Tuple
import argparse
import dgl
import time
import os
from dataclasses import dataclass
import psutil


class PartData(object):
    def __init__(self, graph, min_vids, train_nid, labels):
        super().__init__()
        self.graph = graph
        self.min_vids = min_vids
        self.train_nid = train_nid
        self.labels = labels


def show_process_memory_usage(tag: str) -> None:
    process = psutil.Process(os.getpid())
    print(f"[Note]{tag} memory usage:{process.memory_info().rss / 1024**2}MB")


def allreduce(a: torch.Tensor) -> torch.Tensor:
    return torch.ops.npc.allreduce(a)


def test(a: torch.Tensor, perm: torch.Tensor) -> None:
    torch.ops.npc.test(a, perm)


def cache_feats(node_feats: torch.Tensor, sorted_idx: torch.Tensor, num_cached_nodes: int) -> None:
    torch.ops.npc.cache_feats(node_feats, sorted_idx, num_cached_nodes)


def cache_graphs(num_local_nodes: int, num_graph_nodes: int, num_cached_nodes: int, sorted_idx: torch.Tensor, indptr: torch.Tensor, local_indices: torch.Tensor, global_indices: torch.Tensor) -> None:
    torch.ops.npc.cache_graphs(num_local_nodes, num_graph_nodes, num_cached_nodes,
                               sorted_idx, indptr, local_indices, global_indices)


def load_subtensor(labels: torch.Tensor, seeds: torch.Tensor, input_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return labels[seeds], torch.ops.npc.load_subtensor(input_nodes)


def clear_graph_data(graph):
    for k in list(graph.ndata.keys()):
        graph.ndata.pop(k)
    for k in list(graph.edata.keys()):
        graph.edata.pop(k)


def load_partition(args: argparse.ArgumentParser, rank: int, shared_queue: mp.Queue, shared_tensor_list: List[torch.Tensor]) -> PartData:
    # load graph partition
    # show_process_memory_usage(f"[Note]Rank#{rank} before load partition")
    graph, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(
        args.part_config, rank)
    # extract data and delete graph node&edge data
    # [NOTE] graph.ndata
    # 'part_id', '_ID', 'inner_node'
    # [NOTE] graph.edata
    # '_ID', 'inner_edge'
    # [NOTE] node_feats
    # '_N/feat', '_N/labels', '_N/train_mask', '_N/test_mask', '_N/val_mask'
    local_node_feats = node_feats['_N/feat']
    local_train_mask = node_feats['_N/train_mask'].bool()
    inner_node_mask = graph.ndata['inner_node'].bool()
    local_node_lid = torch.nonzero(graph.ndata['inner_node']).squeeze()
    global_nid = graph.ndata[dgl.NID]
    global_node_map = graph.ndata[dgl.NID]
    clear_graph_data(graph)
    # show_process_memory_usage(f"[Note]Rank#{rank} after load partition")
    graph = dgl.add_self_loop(graph)
    global_node_features, global_test_mask, global_uv, global_graph_indptr, global_graph_indices = shared_tensor_list
    num_edges = graph.num_edges()

    u, v = graph.edges()
    # partition graph transform from local NID to global NID
    uv = torch.vstack((global_node_map[u], global_node_map[v]))
    # show_process_memory_usage(f"[Note]Rank#{rank} Finish stack edges")
    shared_queue.put(uv)
    # show_process_memory_usage(f"[Note]Rank#{rank} Put stack edges")
    dist.barrier()
    world_size = args.world_size
    if rank == 0:
        shared_uv = torch.hstack([shared_queue.get()
                                 for i in range(world_size)])
        d0, num_total_edges, = shared_uv.shape
        for d in range(d0):
            global_uv[d][torch.arange(num_total_edges)] = shared_uv[d]
        shared_graph = dgl.graph((global_uv[0], global_uv[1]))
        shared_graph = dgl.add_self_loop(shared_graph)
        shared_graph = dgl.to_bidirected(shared_graph)
        indptr, indices, edge_ids = shared_graph.adj_sparse('csr')
        del shared_graph
        del shared_uv
        del global_uv
        del edge_ids
       # show_process_memory_usage("Rank0 put whole graph")
        d0_indptr, = indptr.shape
        d0_indices, = indices.shape

        global_graph_indptr[torch.arange(d0_indptr)] = indptr
        global_graph_indices[torch.arange(d0_indices)] = indices
        del d0_indptr
        del d0_indices

    dist.barrier()
    whole_graph = dgl.graph(
        ('csr', (global_graph_indptr, global_graph_indices, [])))
    # whole_indptr, whole_indices, whole_edge_ids = whole_graph.adj_sparse('csr')
    # print(f"[Note]id: {id(whole_indptr)}\t {id(whole_indices)}\t {id(whole_edge_ids)}")
    time.sleep(10)
    dist.barrier()
    # show_process_memory_usage(f"Rank{rank} build csr graph")

    # prepare global_node_features for sharing
    # local node: node belongs to this partition
    # graph node: local node and HALO (remoted 1-hop neighbor) node

    num_local_nodes, d1 = local_node_feats.shape
    assert d1 == args.input_dim
    num_graph_nodes = graph.num_nodes()

    # prepare graphnodes feat

    assert num_local_nodes == local_node_lid.shape[0]
    global_test_mask[global_nid[local_node_lid]] += 1
    global_node_features[global_nid[local_node_lid]] = local_node_feats
    dist.barrier()

    # cache node feats
    in_degrees = graph.in_degrees()
    sorted_deg_idx = torch.argsort(in_degrees, descending=True)
    graph_node_feats = global_node_features[global_nid[sorted_deg_idx]]
    # show_process_memory_usage(f"[Note]Rank#{rank} Before cache feats")
    cache_feats(graph_node_feats, sorted_idx=sorted_deg_idx,
                num_cached_nodes=args.num_cached_nodes)
    # show_process_memory_usage(f"[Note]Rank#{rank} After cache feats")
    # cache graphs
    indptr, indices, edge_ids = graph.adj_sparse('csr')
    local_nodes_sorted_deg_idx = sorted_deg_idx[inner_node_mask[sorted_deg_idx]]
    num_cached_graph_nodes = num_local_nodes

    # two copys of graphs: local indices & global indices
    # transform indices to global indices
    global_indices = global_nid[indices]
    # show_process_memory_usage(f"[Note]Rank#{rank} Before cache graphs")
    cache_graphs(num_local_nodes, num_graph_nodes, num_cached_graph_nodes,
                 local_nodes_sorted_deg_idx, indptr, indices, global_indices)
    # show_process_memory_usage(f"[Note]Rank#{rank} After cache graphs")
    min_vids = torch.LongTensor([0]+list(gpb._max_node_ids))
    train_nid = global_nid[torch.masked_select(
        local_node_lid, local_train_mask)]
    labels = node_feats['_N/labels']

    return PartData(whole_graph, min_vids, train_nid, labels)


class NPCNeighborSampler(object):
    def __init__(self, rank, min_vids, fanouts):
        self.rank = rank
        self.min_vids = min_vids
        self.dgl_fanouts = fanouts[1:]
        self.csp_fanout = fanouts[0]
        self.num_layers = len(fanouts)
        self.dgl_sampler = dgl.dataloading.NeighborSampler(
            fanouts=self.dgl_fanouts, replace=True)

    def sample(self, global_graph, seeds):
        # dgl sampling
        seed_nodes, output_nodes, blocks = self.dgl_sampler.sample(
            global_graph, seeds)
        # CSP sampling
        dist.barrier()
        seeds, neighbors, perm, send_offset, recv_offset = sample_neighbors(
            self.min_vids, seed_nodes, self.csp_fanout)
        replicated_seeds = torch.repeat_interleave(seeds, self.csp_fanout)
        block_g = dgl.graph((neighbors, replicated_seeds))
        block = dgl.to_block(g=block_g, dst_nodes=seeds)
        seeds = block.srcdata[dgl.NID]
        blocks.insert(0, block)
        return seeds, output_nodes, blocks, perm, send_offset, recv_offset


def sample_neighbors(min_vids: torch.Tensor, seeds: torch.Tensor, fanout: int):
    return torch.ops.npc.sample_neighbors(min_vids, seeds, fanout)


@dataclass
class FeatureShuffleInfo(object):
    feat_dim: int
    send_offset: List[int]
    recv_offset: List[int]
    permutation: torch.Tensor


class FeatureShuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fsi: FeatureShuffleInfo, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.fsi = fsi
        shuffle_result = feat_shuffle(input_tensor, fsi.send_offset, fsi.recv_offset,
                                      fsi.permutation, fsi.feat_dim, 1)
        return shuffle_result

    @staticmethod
    def backward(ctx, grad_output_tensor: torch.Tensor) -> torch.Tensor:
        fsi = ctx.fsi
        shuffle_grad = feat_shuffle(grad_output_tensor, fsi.recv_offset,
                                    fsi.send_offset, fsi.permutation, fsi.feat_dim, 0)
        return (None, shuffle_grad)


def feat_shuffle(inputs: torch.Tensor, send_offset: torch.Tensor, recv_offset: torch.Tensor, permutation: torch.Tensor, feat_dim: int, fwd_flag: int):
    return torch.ops.npc.feat_shuffle(inputs, send_offset, recv_offset, permutation, feat_dim, fwd_flag)
