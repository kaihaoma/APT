
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
    def __init__(self, min_vids, train_nid, labels, cache_mask):
        super().__init__()
        self.min_vids = min_vids
        self.train_nid = train_nid
        self.labels = labels
        self.cache_mask = cache_mask


GB_TO_BYTES = 1024 * 1024 * 1024


def show_process_memory_usage(tag: str) -> None:
    process = psutil.Process(os.getpid())
    print(f"[Note]{tag} memory usage:{process.memory_info().rss / 1024**2}MB")


def get_tensor_mem_usage_in_gb(ts: torch.Tensor):
    return ts.numel() * ts.element_size() / GB_TO_BYTES


def get_total_mem_usage_in_gb():
    symem = psutil.virtual_memory()
    total = symem[0] / GB_TO_BYTES
    uses = symem[3] / GB_TO_BYTES
    return f"Mem usage: {uses} / {total}GB"


def allreduce(a: torch.Tensor) -> torch.Tensor:
    return torch.ops.npc.allreduce(a)


def test(a: torch.Tensor, perm: torch.Tensor) -> None:
    torch.ops.npc.test(a, perm)


def cache_feats(node_feats: torch.Tensor, sorted_idx: torch.Tensor, cache_ratio: float, num_total_nodes: int) -> None:
    torch.ops.npc.cache_feats(node_feats, sorted_idx,
                              cache_ratio, num_total_nodes)


def cache_feats_shared(global_node_feats: torch.Tensor, cached_feats: torch.Tensor, cached_sort_idx: torch.Tensor) -> None:
    torch.ops.npc.cache_feats_shared(
        global_node_feats, cached_feats, cached_sort_idx)


def cache_graphs(num_local_nodes: int, num_graph_nodes: int, num_cached_nodes: int, sorted_idx: torch.Tensor, indptr: torch.Tensor, local_indices: torch.Tensor, global_indices: torch.Tensor) -> None:
    torch.ops.npc.cache_graphs(num_local_nodes, num_graph_nodes, num_cached_nodes,
                               sorted_idx, indptr, local_indices, global_indices)


def mix_cache_graphs(num_cached_nodes: int, cached_node_idx: torch.Tensor, cached_indptr: torch.Tensor, cached_indices: torch.Tensor, global_indptr: torch.Tensor, global_indices: torch.Tensor):
    torch.ops.npc.mix_cache_graphs(num_cached_nodes, cached_node_idx, cached_indptr,
                                   cached_indices, global_indptr, global_indices)


def tensor_max_min_info(ts: torch.Tensor):
    return f"shape:{ts.shape}\t max:{torch.max(ts)}\t min:{torch.min(ts)}"


def tensor_loc(ts: torch.Tensor):
    return f"pin:{ts.is_pinned()}\t shared:{ts.is_shared()}\t device:{ts.device}"


def load_subtensor(args, sample_result):
    if args.system == "NPC":
        # input_nodes, seeds, blocks, perm, send_offset, recv_offset
        fsi = FeatureShuffleInfo(feat_dim=args.num_hidden,
                                 send_offset=sample_result[4].to("cpu"),
                                 recv_offset=sample_result[5].to("cpu"),
                                 permutation=sample_result[3],)
        return sample_result[2], torch.ops.npc.load_subtensor(sample_result[0]), fsi
    else:
        return sample_result[2], torch.ops.npc.load_subtensor(sample_result[0])


def clear_graph_data(graph):
    for k in list(graph.ndata.keys()):
        graph.ndata.pop(k)
    for k in list(graph.edata.keys()):
        graph.edata.pop(k)


def load_partition(args: argparse.ArgumentParser, rank: int, shared_tensor_list: List[torch.Tensor]) -> PartData:

    min_vids_list = args.min_vids
    global_node_feats, global_labels, global_train_mask, indptr, indices = shared_tensor_list

    num_total_nodes, feat_dim = global_node_feats.shape
    num_local_nodes = min_vids_list[rank+1] - min_vids_list[rank]
    local_nodes_id = torch.arange(num_local_nodes)
    global_nodes_id = local_nodes_id + min_vids_list[rank]
    total_node_id = torch.arange(num_total_nodes)

    if args.cache_memory > 0.:
        # get memory
        sorted_idx = torch.load(os.path.join(
            args.sorted_idx_path, f"{rank}_sorted_idx.pt"))
        num_cache_nodes = int(num_total_nodes * args.feat_cache_ratio)
        sorted_idx = sorted_idx[:num_cache_nodes]
        cache_feat_node_idx = sorted_idx[sorted_idx < num_total_nodes]
        cache_graph_node_idx = sorted_idx[sorted_idx >=
                                          num_total_nodes] - num_total_nodes
        num_cached_feat_nodes = cache_feat_node_idx.numel()
        num_cached_graph_nodes = cache_graph_node_idx.numel()
        print(
            f"[Note]#feats:{num_cached_feat_nodes}\t #graphs:{num_cached_graph_nodes}")
        # cache feat
        cached_feats = global_node_feats[cache_feat_node_idx].to(rank)
        cache_feats_shared(global_node_feats=global_node_feats,
                           cached_feats=cached_feats, cached_sort_idx=cache_feat_node_idx)
        # cache graph
        if num_cached_graph_nodes > 0:
            cache_indptr = torch.hstack(
                [indptr[pt+1] - indptr[pt] for pt in cache_graph_node_idx])
            cache_indices = torch.cat(
                [indices[indptr[pt]:indptr[pt+1]] for pt in cache_graph_node_idx]).to(rank)
            cache_indptr = torch.cat(
                [torch.LongTensor([0]), torch.cumsum(cache_indptr, dim=0)]).to(rank)
        else:
            cache_indptr = torch.empty(0, dtype=torch.long)
            cache_indices = torch.empty(0, dtype=torch.long)

        print(
            f"[Note]#graph cached:{num_cached_graph_nodes}\t indptr:{cache_indptr.shape}\t indices:{cache_indices.shape}")
        dist.barrier()

        mix_cache_graphs(num_cached_nodes=num_cached_graph_nodes, cached_node_idx=cache_graph_node_idx,
                         cached_indptr=cache_indptr, cached_indices=cache_indices, global_indptr=indptr, global_indices=indices)
    else:
        print(f"[Note]Cache node_feats")
        if args.system in ["NPC", "DSP-1hop"]:
            u = indices[indptr[min_vids_list[rank]]: indptr[min_vids_list[rank+1]]]
            nid, counts = torch.unique(u, return_counts=True)
            del u
            sort_idx = nid[torch.argsort(counts, descending=True)]
            del nid
            del counts
            num_graph_nodes = sort_idx.numel()
            num_cached_nodes = min(num_graph_nodes, int(
                args.feat_cache_ratio * num_total_nodes))
            print(
                f"[Note][1hop cache] Cache {args.feat_cache_ratio} of {num_total_nodes} = {num_cached_nodes}")

            cached_sort_idx = sort_idx[:num_cached_nodes]

            cached_feats = global_node_feats[cached_sort_idx].to(rank)
            cache_feats_shared(global_node_feats=global_node_feats,
                               cached_feats=cached_feats, cached_sort_idx=cached_sort_idx)

        elif args.system == "DSP":
            # local cache
            in_degrees = torch.diff(indptr)[global_nodes_id]
            sort_idx = torch.argsort(in_degrees, descending=True)
            del in_degrees
            num_cached_nodes = min(num_local_nodes, int(
                args.feat_cache_ratio * num_total_nodes))
            cached_sort_idx = global_nodes_id[sort_idx[:num_cached_nodes]]
            cached_feats = global_node_feats[cached_sort_idx].to(rank)

            print(
                f"[Note][local cache] Cache {args.feat_cache_ratio} of {num_total_nodes} = {num_cached_nodes}")
            cache_feats_shared(global_node_feats,
                               cached_feats, cached_sort_idx)

        elif args.system == "DGL-global":
            # in_degrees = graph.in_degrees()
            in_degrees = torch.diff(indptr)
            sort_idx = torch.argsort(in_degrees, descending=True)
            num_cached_nodes = min(num_total_nodes, int(
                args.feat_cache_ratio * num_total_nodes))
            cached_sort_idx = total_node_id[sort_idx[:num_cached_nodes]]
            cached_feats = global_node_feats[cached_sort_idx].to(rank)

            print(
                f"[Note][global cache] Cache {args.feat_cache_ratio} of {num_total_nodes} = {num_cached_nodes}")
            cache_feats_shared(global_node_feats,
                               cached_feats, cached_sort_idx)

        else:
            raise NotImplementedError

        print(f"[Note]Rank#{rank} after cache_feats")

        # cache graph topology
        print(f"[Note]System:{args.system} sort_idx_shape:{sort_idx.shape}")
        # if args.graph_cache_ratio > 0.:
        if args.system == "NPC":
            # extract graph
            upper_limit = sort_idx.numel()
            num_cached_graph_nodes = min(upper_limit, int(
                args.graph_cache_ratio * num_total_nodes))
            cache_graph_node_idx = sort_idx[:num_cached_graph_nodes]
            print(
                f"[Note]cache ratio{args.graph_cache_ratio} of {num_total_nodes} = {num_cached_graph_nodes}")

            if num_cached_graph_nodes > 0:

                cache_indptr = torch.hstack(
                    [indptr[pt+1] - indptr[pt] for pt in cache_graph_node_idx])
                cache_indices = torch.cat(
                    [indices[indptr[pt]:indptr[pt+1]] for pt in cache_graph_node_idx]).to(rank)
                cache_indptr = torch.cat(
                    [torch.LongTensor([0]), torch.cumsum(cache_indptr, dim=0)]).to(rank)
            else:
                cache_indptr = torch.empty(0, dtype=torch.long)
                cache_indices = torch.empty(0, dtype=torch.long)

            dist.barrier()
            mix_cache_graphs(num_cached_nodes=num_cached_graph_nodes, cached_node_idx=cache_graph_node_idx,
                             cached_indptr=cache_indptr, cached_indices=cache_indices, global_indptr=indptr, global_indices=indices)

    if args.rebalance_train_nid:
        all_train_nid = torch.masked_select(
            torch.arange(num_total_nodes), global_train_mask)
        num_all_train_nids, = all_train_nid.shape
        num_train_nids_per_rank = num_all_train_nids // args.world_size
        global_train_nid = all_train_nid[rank *
                                         num_train_nids_per_rank: (rank+1) * num_train_nids_per_rank]
    else:
        local_train_mask = global_train_mask[min_vids_list[rank]:min_vids_list[rank+1]]
        global_train_nid = torch.masked_select(
            global_nodes_id, local_train_mask)

    print(f"[Note]Rank#{rank} after rebalance")
    min_vids = torch.LongTensor(min_vids_list)

    return PartData(min_vids, global_train_nid, global_labels, None)


class MixedNeighborSampler(object):
    def __init__(self, rank, min_vids, fanouts):
        self.rank = rank
        self.min_vids = min_vids
        self.fir_fanouts = fanouts[1:]
        self.las_fanouts = fanouts[0]
        self.num_layers = len(fanouts)

    def sample(self, graph, seeds):
        output_nodes = seeds
        blocks = []
        for fanout in reversed(self.fir_fanouts):
            seeds, neighbors = local_sample_one_layer(seeds, fanout)
            replicated_seeds = torch.repeat_interleave(seeds, fanout)
            block_g = dgl.graph((neighbors, replicated_seeds))
            block = dgl.to_block(g=block_g, dst_nodes=seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        # last layer
        seeds, neighbors, perm, send_offset, recv_offset = sample_neighbors(
            self.min_vids, seeds, self.las_fanouts)
        replicated_seeds = torch.repeat_interleave(seeds, self.las_fanouts)
        block_g = dgl.graph((neighbors, replicated_seeds))
        block = dgl.to_block(g=block_g, dst_nodes=seeds)
        seeds = block.srcdata[dgl.NID]
        blocks.insert(0, block)
        return seeds, output_nodes, blocks, perm, send_offset, recv_offset


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
        shuffle_seed_nodes, permutation, send_offset, recv_offset = shuffle_seeds(
            self.min_vids, seed_nodes)
        # last layer sampling
        las_seed_nodes, las_output_nodes, last_blocks = self.las_sampler.sample(
            global_graph, shuffle_seed_nodes)
        last_blocks.extend(blocks)
        return las_seed_nodes, output_nodes, last_blocks, permutation, send_offset, recv_offset


class DGLNeighborSampler(dgl.dataloading.NeighborSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, mask=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(fanouts, edge_dir, prob, mask, replace, prefetch_node_feats,
                         prefetch_labels, prefetch_edge_feats, output_device)

    def sample(self, g, seed_nodes, exclude_eids=None):     # pylint: disable=arguments-differ
        """Sample a list of blocks from the given seed nodes."""
        result = self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
        return result


def sample_neighbors(min_vids: torch.Tensor, seeds: torch.Tensor, fanout: int):
    return torch.ops.npc.sample_neighbors(min_vids, seeds, fanout)


def local_sample_one_layer(seeds: torch.Tensor, fanout: int):
    return torch.ops.npc.local_sample_one_layer(seeds, fanout)


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
