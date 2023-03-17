
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
    def __init__(self, min_vids, train_nid, labels):
        super().__init__()
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


def cache_feats(node_feats: torch.Tensor, sorted_idx: torch.Tensor, cache_ratio: float, num_total_nodes: int) -> None:
    torch.ops.npc.cache_feats(node_feats, sorted_idx, cache_ratio, num_total_nodes)


def cache_graphs(num_local_nodes: int, num_graph_nodes: int, num_cached_nodes: int, sorted_idx: torch.Tensor, indptr: torch.Tensor, local_indices: torch.Tensor, global_indices: torch.Tensor) -> None:
    torch.ops.npc.cache_graphs(num_local_nodes, num_graph_nodes, num_cached_nodes,
                               sorted_idx, indptr, local_indices, global_indices)


def tensor_max_min_info(ts):
    return f"shape:{ts.shape}\t max:{torch.max(ts)}\t min:{torch.min(ts)}"


def load_subtensor(labels: torch.Tensor, seeds: torch.Tensor, input_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return labels[seeds], torch.ops.npc.load_subtensor(input_nodes)


def clear_graph_data(graph):
    for k in list(graph.ndata.keys()):
        graph.ndata.pop(k)
    for k in list(graph.edata.keys()):
        graph.edata.pop(k)


def combine_partition_and_save(args: argparse.ArgumentParser, rank: int, shared_queue: mp.Queue, shared_tensor_list: List[torch.Tensor]) -> None:
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
    local_val_mask = node_feats['_N/val_mask'].bool()
    local_test_mask = node_feats['_N/test_mask'].bool()
    labels = node_feats['_N/labels']
    inner_node_mask = graph.ndata['inner_node'].bool()
    local_node_lid = torch.nonzero(graph.ndata['inner_node']).squeeze()
    print(f"[Note]Rank#{rank}: local_node_id: {local_node_lid}")
    global_nid = graph.ndata[dgl.NID]
    global_node_map = graph.ndata[dgl.NID]
    clear_graph_data(graph)
    # show_process_memory_usage(f"[Note]Rank#{rank} after load partition")
    graph = dgl.add_self_loop(graph)
    # global_node_features, global_test_mask, global_uv, global_graph_indptr, global_graph_indices = shared_tensor_list
    global_uv, global_node_features, global_checker, global_labels, global_train_mask, global_val_mask, global_test_mask = shared_tensor_list
    num_edges = graph.num_edges()

    u, v = graph.edges()
    # partition graph transform from local NID to global NID
    uv = torch.vstack((global_node_map[u], global_node_map[v]))
    # show_process_memory_usage(f"[Note]Rank#{rank} Finish stack edges")
    shared_queue.put(uv)
    global_checker[global_nid[local_node_lid]] += 1
    global_labels[global_nid[local_node_lid]] = labels
    global_node_features[global_nid[local_node_lid]] = local_node_feats
    global_train_mask[global_nid[local_node_lid]] = local_train_mask
    global_val_mask[global_nid[local_node_lid]] = local_val_mask
    global_test_mask[global_nid[local_node_lid]] = local_test_mask
    dist.barrier()
    print(
        f"[Note]global_checker: max:{torch.max(global_checker)}\t min:{torch.min(global_checker)}, shape:{global_checker.shape}")

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
        ndata_keys = shared_graph.ndata.keys()
        print(f"[Note]Before ndata:{ndata_keys}")
        shared_graph.ndata['_N/feat'] = global_node_features
        shared_graph.ndata['_N/labels'] = global_labels
        shared_graph.ndata['_N/train_mask'] = global_train_mask
        shared_graph.ndata['_N/val_mask'] = global_val_mask
        shared_graph.ndata['_N/test_mask'] = global_test_mask
        ndata_keys = shared_graph.ndata.keys()
        print(f"[Note]After ndata:{ndata_keys}")
        save_path = "./npc_dataset/ogbn-productsM4.bin"
        print(f"[Note]Save graph{shared_graph} to dir: {save_path}")
        dgl.data.utils.save_graphs(save_path, [shared_graph])


def load_partition(args: argparse.ArgumentParser, rank: int, graph, shared_tensor_list: List[torch.Tensor]) -> PartData:

    min_vids_list = args.min_vids
    global_node_feats, global_labels, global_train_mask, global_val_mask, global_test_mask, = shared_tensor_list
    # global_node_feats, = shared_tensor_list
    num_total_nodes = graph.num_nodes()
    num_local_nodes = min_vids_list[rank+1] - min_vids_list[rank]
    local_nodes_id = torch.arange(num_local_nodes)
    global_nodes_id = local_nodes_id + min_vids_list[rank]
    # local range
    graph_node_feat_keys = graph.ndata.keys()
    # cache node feats
    # get in_degrees
    u, v = graph.in_edges(global_nodes_id)

    nid, counts = torch.unique(u, return_counts=True)
    sort_idx = nid[torch.argsort(counts, descending=True)]
    num_graph_nodes, = sort_idx.shape
    graph_node_feats = global_node_feats[sort_idx].detach().clone()

    dist.barrier()
    cache_feats(graph_node_feats, sorted_idx=sort_idx,
                cache_ratio=args.cache_ratio, num_total_nodes=num_total_nodes)

    # global_train_mask = graph.ndata['_N/train_mask'].bool()
    # global_labels = graph.ndata['_N/labels']
    if args.rebalance_train_nid:
        all_train_nid = torch.masked_select(torch.arange(num_total_nodes), global_train_mask)
        num_all_train_nids, = all_train_nid.shape
        num_train_nids_per_rank = num_all_train_nids // args.world_size
        global_train_nid = all_train_nid[rank * num_train_nids_per_rank: (rank+1) * num_train_nids_per_rank]
    else:
        local_train_mask = global_train_mask[min_vids_list[rank]:min_vids_list[rank+1]]
        global_train_nid = torch.masked_select(global_nodes_id, local_train_mask)

    min_vids = torch.LongTensor(min_vids_list)
    return PartData(min_vids, global_train_nid, global_labels)


class NPCNeighborSampler(object):
    def __init__(self, rank, min_vids, fanouts):
        self.rank = rank
        self.min_vids = min_vids
        self.fir_fanouts = fanouts[1:]
        self.las_fanouts = fanouts[:1]
        self.num_layers = len(fanouts)
        self.fir_sampler = dgl.dataloading.NeighborSampler(
            fanouts=self.fir_fanouts, replace=True)
        self.las_sampler = dgl.dataloading.NeighborSampler(
            fanouts=self.las_fanouts, replace=True)

    def sample(self, global_graph, seeds):
        # first (k-1) layer sampling
        seed_nodes, output_nodes, blocks = self.fir_sampler.sample(
            global_graph, seeds)
        # shuffle & re-shuffle for last layer
        shuffle_seed_nodes, permutation, send_offset, recv_offset = shuffle_seeds(self.min_vids, seed_nodes)
        # last layer sampling
        las_seed_nodes, las_output_nodes, last_blocks = self.las_sampler.sample(global_graph, shuffle_seed_nodes)
        last_blocks.extend(blocks)
        return las_seed_nodes, output_nodes, last_blocks, permutation, send_offset, recv_offset


def sample_neighbors(min_vids: torch.Tensor, seeds: torch.Tensor, fanout: int):
    return torch.ops.npc.sample_neighbors(min_vids, seeds, fanout)


def shuffle_seeds(min_vids: torch.Tensor, seeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.npc.shuffle_seeds(min_vids, seeds)


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
