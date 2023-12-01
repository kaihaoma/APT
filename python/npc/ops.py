import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Union, Tuple, Optional
import argparse
import dgl
import time
import os
from dataclasses import dataclass
import threading
import queue
import psutil


GB_TO_BYTES = 1024 * 1024 * 1024
BYTES_PER_ELEMENT = 4


# -- utils --
def show_process_memory_usage(tag: str) -> None:
    process = psutil.Process(os.getpid())
    print(f"[Note]{tag} memory usage:{process.memory_info().rss / 1024**2}MB")


def get_tensor_mem_usage_in_gb(ts: torch.Tensor):
    return ts.numel() * ts.element_size() / GB_TO_BYTES


def get_tensor_info_str(ts):
    return f"shape:{ts.shape}\t max: {torch.max(ts)}\t min: {torch.min(ts)}"


def get_total_mem_usage_in_gb():
    symem = psutil.virtual_memory()
    total = symem[0] / GB_TO_BYTES
    uses = symem[3] / GB_TO_BYTES
    return f"Mem usage: {uses} / {total}GB"


def nccl_get_unique_id() -> torch.Tensor:
    return torch.ops.npc.nccl_get_unique_id()


def allgather(a: torch.Tensor, comm_type: int = 0) -> torch.Tensor:
    return torch.ops.npc.allgather(a, comm_type)


def cache_feats_shared(
    num_total_nodes: int,
    localnode_feats: torch.Tensor,
    cached_feats: torch.Tensor,
    cached_idx: torch.Tensor,
    localnode_idx: Optional[torch.Tensor] = None,
    feat_dim_offset: int = 0,
) -> None:
    torch.ops.npc.cache_feats_shared(
        num_total_nodes,
        localnode_feats,
        cached_feats,
        cached_idx,
        localnode_idx,
        feat_dim_offset,
    )


def mix_cache_graphs(
    num_cached_nodes: int,
    cached_node_idx: torch.Tensor,
    cached_indptr: torch.Tensor,
    cached_indices: torch.Tensor,
    global_indptr: torch.Tensor,
    global_indices: torch.Tensor,
):
    torch.ops.npc.mix_cache_graphs(
        num_cached_nodes,
        cached_node_idx,
        cached_indptr,
        cached_indices,
        global_indptr,
        global_indices,
    )


class PartData(object):
    def __init__(
        self,
        min_vids,
        train_nid,
        labels,
        cache_mask,
        num_cached_feat_nodes,
        num_cached_feat_elements,
        num_cached_graph_nodes,
        num_cached_graph_elements,
    ):
        super().__init__()
        self.min_vids = min_vids
        self.train_nid = train_nid
        self.labels = labels
        self.cache_mask = cache_mask
        self.num_cached_feat_nodes = num_cached_feat_nodes
        self.num_cached_feat_elements = num_cached_feat_elements
        self.num_cached_graph_nodes = num_cached_graph_nodes
        self.num_cached_graph_elements = num_cached_graph_elements


def cache_adj_and_feats(
    args: argparse.Namespace,
    rank: int,
    localnode_feats_idx: int,
    num_total_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Init localnode_feats_idx_mask
    localnode_feats_idx_mask = torch.zeros(num_total_nodes, dtype=torch.bool)
    localnode_feats_idx_mask[localnode_feats_idx] = True

    # [TODO] "costmodel" assume all node feats stored in localnode uva
    if args.cache_mode == "costmodel":
        sorted_idx = torch.load(os.path.join(args.sorted_idx_path, f"{rank}_sorted_idx.pt"))
        sorted_mem_usage = torch.load(args.idx_mem_path)[sorted_idx]
        cumsum_sorted_mem_usage = torch.cumsum(sorted_mem_usage, dim=0)

        num_cache_nodes = max(
            0,
            (torch.max(cumsum_sorted_mem_usage > args.cache_memory, 0)[1].item() - 1),
        )
        print(f"[Note]num_cache_nodes:{num_cache_nodes}\t cache_memory:{args.cache_memory}")

        sorted_idx = sorted_idx[:num_cache_nodes]
        cache_feat_node_idx = sorted_idx[sorted_idx < num_total_nodes]
        cache_graph_node_idx = sorted_idx[sorted_idx >= num_total_nodes] - num_total_nodes
        num_cached_feat_nodes = cache_feat_node_idx.numel()
        num_cached_graph_nodes = cache_graph_node_idx.numel()
    # [TODO] "costmodel" assume all node feats stored in localnode uva
    elif args.cache_mode == "greedy":
        mem_usage = torch.load(args.idx_mem_path)
        data = torch.load(os.path.join(args.greedy_sorted_idx_path, f"rk#{rank}costmodel_sorted_idx.pt"))
        feat_idx = data[0]
        graph_idx = data[1]
        sorted_feat_mem = torch.cumsum(mem_usage[feat_idx], dim=0)
        sorted_graph_mem = torch.cumsum(mem_usage[graph_idx + num_total_nodes], dim=0)
        # --- binary search ---
        lb = 0
        ub = num_total_nodes
        while ub > lb + 1:
            mid = int((ub + lb) // 2)
            total_mem = (sorted_feat_mem[int(mid * args.greedy_feat_ratio)] + sorted_graph_mem[int(mid * (1 - args.greedy_feat_ratio))]).item()

            if total_mem <= args.cache_memory:
                lb = mid
            else:
                ub = mid
        # ---end of binary search ---
        num_cached_feat_nodes = int(lb * args.greedy_feat_ratio)
        num_cached_graph_nodes = int(lb * (1 - args.greedy_feat_ratio))

        cache_feat_node_idx = feat_idx[:num_cached_feat_nodes]
        cache_graph_node_idx = graph_idx[:num_cached_graph_nodes]
        del feat_idx
        del graph_idx
        print(
            f"[Note]greedy: #feats:{num_cached_feat_nodes}\t  mem:{sorted_feat_mem[num_cached_feat_nodes]} #graphs:{num_cached_graph_nodes}\t mem:{sorted_graph_mem[num_cached_graph_nodes]}"
        )
    elif args.cache_mode == "dryrun":
        # temp cache no graph nodes
        print(f"[Note]Rk#{rank} Caching mode:{args.cache_mode} No cache graph nodes")
        num_cached_graph_nodes = 0
        cache_graph_node_idx = torch.tensor([], dtype=torch.long)
        print(f"[Note]Rk#{rank} Caching mode:{args.cache_mode} Cache node_feats")

        print(f"[Note]load dryrun for sys {args.system} from {args.dryrun_file_path}")

        # load dryrun result
        if args.gpu_cache_worker == "single":
            caching_candidate_path = f"{args.dryrun_file_path}/rk#{rank}_epo10.pt"
            node_feats_freqs = torch.load(caching_candidate_path)[1]
            sorted_idx = torch.sort(node_feats_freqs, descending=True)[1]
        else:
            if rank == 0:
                ori_freq_lists = [torch.load(f"{args.dryrun_file_path}/rk#{r}_epo10.pt")[1] for r in range(args.world_size)]
                sum_freq_lists = torch.stack(ori_freq_lists, dim=0).sum(dim=0).to(args.device)
                sorted_idx = torch.sort(sum_freq_lists, descending=True)[1]
            else:
                sorted_idx = torch.empty(num_total_nodes, dtype=torch.long, device=args.device)
            dist.broadcast(sorted_idx, 0)
            sorted_idx = sorted_idx.cpu()

        # if args.system in ["NP"]:
        #     caching_candidate_path = f"{args.dryrun_file_path}/rk#{rank}_epo100.pt"
        #     node_feats_freqs = torch.load(caching_candidate_path)[1]
        #     sorted_idx = torch.sort(node_feats_freqs, descending=True)[1]

        # else:
        #     if rank == 0:
        #         ori_freq_lists = [torch.load(f"{args.dryrun_file_path}/rk#{r}_epo100.pt")[1] for r in range(args.world_size)]
        #         sum_freq_lists = torch.stack(ori_freq_lists, dim=0).sum(dim=0).to(args.device)

        #     if args.system in ["MP", "DP"]:
        #         if rank == 0:
        #             sorted_idx = torch.sort(sum_freq_lists, descending=True)[1]
        #         else:
        #             sorted_idx = torch.empty(num_total_nodes, dtype=torch.long, device=args.device)
        #         dist.broadcast(sorted_idx, 0)
        #         sorted_idx = sorted_idx.cpu()

        #     elif args.system in ["SP"]:
        #         if rank != 0:
        #             sum_freq_lists = torch.empty(num_total_nodes, dtype=torch.long, device=args.device)
        #         dist.broadcast(sum_freq_lists, 0)

        #         node_feats_freqs = torch.zeros(num_total_nodes, dtype=torch.long)
        #         node_feats_freqs[args.min_vids[rank] : args.min_vids[rank + 1]] = sum_freq_lists[args.min_vids[rank] : args.min_vids[rank + 1]]
        #         sorted_idx = torch.sort(node_feats_freqs, descending=True)[1].cpu()
        #         del node_feats_freqs
        #         del sum_freq_lists

        # cache based on sorted_idx
        local_sorted_idx = torch.masked_select(sorted_idx, localnode_feats_idx_mask[sorted_idx])
        base = args.world_size if args.system == "MP" else 1
        num_cached_feat_nodes = min(
            local_sorted_idx.numel(),
            int(args.cache_memory * base / (args.input_dim * BYTES_PER_ELEMENT)),
        )

        cache_feat_node_idx = local_sorted_idx[:num_cached_feat_nodes]

    else:
        num_localnodes = localnode_feats_idx.numel()
        num_cached_feat_nodes = int(args.feat_cache_ratio * num_localnodes)
        rand_cache_feat_node_idx = torch.randperm(num_localnodes)[:num_cached_feat_nodes]
        cache_feat_node_idx = localnode_feats_idx[rand_cache_feat_node_idx]
        num_cached_graph_nodes = 0
        cache_graph_node_idx = torch.tensor([], dtype=torch.long)
        print(f"[Note]Random cache mode, no cache adj. and random cache feats: {num_cached_feat_nodes} of {num_localnodes}")
    return cache_feat_node_idx, cache_graph_node_idx


def load_partition(
    args: argparse.ArgumentParser,
    rank: int,
    device: torch.device,
    shared_tensor_list: List[torch.Tensor],
) -> PartData:
    min_vids_list = args.min_vids
    (
        localnode_feats_idx,
        localnode_feats,
        global_labels,
        global_train_mask,
        indptr,
        indices,
        *rest,
    ) = shared_tensor_list

    # Init cache info
    num_cached_feat_nodes = 0
    num_cached_feat_elements = 0
    num_cached_graph_nodes = 0
    num_cached_graph_elements = 0

    num_localnode_feats, feat_dim = localnode_feats.shape
    num_total_nodes = args.min_vids[-1]
    num_local_nodes = min_vids_list[rank + 1] - min_vids_list[rank]
    print(
        f"[Note]Rk#{rank}\t dev:{device}\t #localnode_feats:{num_localnode_feats}\t #local nodes :{num_local_nodes}\t #total nodes:{num_total_nodes}"
    )
    cache_mask = torch.zeros(
        num_total_nodes,
    ).bool()

    local_nodes_id = torch.arange(num_local_nodes)
    global_nodes_id = local_nodes_id + min_vids_list[rank]
    total_node_id = torch.arange(num_total_nodes)

    (
        cache_feat_node_idx,
        cache_graph_node_idx,
    ) = cache_adj_and_feats(
        args,
        rank,
        localnode_feats_idx,
        num_total_nodes,
    )
    num_cached_feat_nodes = cache_feat_node_idx.numel()
    num_cached_graph_nodes = cache_graph_node_idx.numel()
    print(f"[Note]Rk#{rank} {args.cache_mode}: #feats:{num_cached_feat_nodes}\t #graphs:{num_cached_graph_nodes}\t Mem:{get_total_mem_usage_in_gb()}")
    dist.barrier()
    if args.max_cache_feat_nodes >= 0 and num_cached_feat_nodes > args.max_cache_feat_nodes:
        num_cached_feat_nodes = args.max_cache_feat_nodes
        cache_feat_node_idx = cache_feat_node_idx[:num_cached_feat_nodes]

    if args.max_cache_graph_nodes >= 0 and num_cached_graph_nodes > args.max_cache_graph_nodes:
        num_cached_graph_nodes = args.max_cache_graph_nodes
        cache_graph_node_idx = cache_graph_node_idx[:num_cached_graph_nodes]

    print(f"[Note]Rk#{rank} Force Limit{args.cache_mode}: #feats:{num_cached_feat_nodes}\t #graphs:{num_cached_graph_nodes}\t")
    # cache graph
    if num_cached_graph_nodes > 0:
        cache_indptr = torch.hstack([indptr[pt + 1] - indptr[pt] for pt in cache_graph_node_idx])

        cache_indptr = torch.cat([torch.LongTensor([0]), torch.cumsum(cache_indptr, dim=0)]).to(device)

        cache_indices = torch.cat([indices[indptr[pt] : indptr[pt + 1]] for pt in cache_graph_node_idx]).to(device)

    else:
        cache_indptr = torch.empty(0, dtype=torch.long)
        cache_indices = torch.empty(0, dtype=torch.long)

    print(
        f"[Note]Rk#{rank} #graph cached:{num_cached_graph_nodes}\t indptr:{cache_indptr.shape}\t indices:{cache_indices.shape}\t Mem:{get_total_mem_usage_in_gb()}"
    )
    num_cached_graph_elements = cache_indices.numel()
    dist.barrier()

    mix_cache_graphs(
        num_cached_nodes=num_cached_graph_nodes,
        cached_node_idx=cache_graph_node_idx,
        cached_indptr=cache_indptr,
        cached_indices=cache_indices,
        global_indptr=indptr,
        global_indices=indices,
    )

    # cache feat
    num_cached_feat_elements = num_cached_feat_nodes * args.input_dim
    # map cache_feat_node_idx to pos
    localnode_feat_pos = torch.zeros(num_total_nodes, dtype=torch.long)
    localnode_feat_pos[localnode_feats_idx] = torch.arange(num_localnode_feats)
    cache_feat_node_pos = localnode_feat_pos[cache_feat_node_idx]

    if args.system == "MP":
        print(f"[Note]Rk#{rank} cumsum_feat_dim:{args.cumsum_feat_dim}\t my:{args.cumsum_feat_dim[rank]} - {args.cumsum_feat_dim[rank+1]}")
        # single machine scenario
        if args.num_localnode_feats_in_workers == -1:
            cached_feats = localnode_feats[
                cache_feat_node_pos,
                args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1],
            ].to(device)
            feat_dim_offset = args.cumsum_feat_dim[rank]
        else:
            # multi machine scenario
            localnode_st = args.node_rank * args.nproc_per_node
            feat_dim_offset = args.cumsum_feat_dim[rank] - args.cumsum_feat_dim[localnode_st]
            feat_dim_offset_en = args.cumsum_feat_dim[rank + 1] - args.cumsum_feat_dim[localnode_st]
            cached_feats = localnode_feats[cache_feat_node_pos, feat_dim_offset:feat_dim_offset_en].to(device)

    else:
        cached_feats = localnode_feats[cache_feat_node_pos].to(device)
        feat_dim_offset = 0
    cache_feats_shared(
        num_total_nodes=num_total_nodes,
        localnode_feats=localnode_feats,
        cached_feats=cached_feats,
        cached_idx=cache_feat_node_idx,
        localnode_idx=localnode_feats_idx,
        feat_dim_offset=feat_dim_offset,
    )
    dist.barrier()
    print(
        f"[Note]Done cached feats: {num_cached_feat_nodes} of {num_total_nodes} = {round(1.*num_cached_feat_nodes / num_total_nodes, 2)}%, Mem:{get_total_mem_usage_in_gb()}"
    )

    # rebalance train_nid
    if args.rebalance_train_nid:
        all_train_nid = torch.masked_select(torch.arange(num_total_nodes), global_train_mask)
        (num_all_train_nids,) = all_train_nid.shape
        num_train_nids_per_rank = num_all_train_nids // args.world_size
        global_train_nid = all_train_nid[rank * num_train_nids_per_rank : (rank + 1) * num_train_nids_per_rank]
        # ref:
        num_trains_bef = torch.sum(global_train_mask[min_vids_list[rank] : min_vids_list[rank + 1]]).item()
        num_trains_aft = torch.numel(global_train_nid)
        print(f"[Note]#trains before:{num_trains_bef}\t after:{num_trains_aft}")

    else:
        local_train_mask = global_train_mask[min_vids_list[rank] : min_vids_list[rank + 1]]
        global_train_nid = torch.masked_select(global_nodes_id, local_train_mask)

    print(f"[Note]Rank#{rank} after rebalance")
    min_vids = torch.LongTensor(min_vids_list).to(device)
    global_train_nid = global_train_nid.to(device)
    global_labels = global_labels.to(device)
    cache_mask[cache_feat_node_idx] = True

    # register min_vids
    register_min_vids(min_vids)

    if args.cross_machine_feat_load:
        register_multi_machines_scheme(args)

    return PartData(
        min_vids,
        global_train_nid,
        global_labels,
        cache_mask,
        num_cached_feat_nodes,
        num_cached_feat_elements,
        num_cached_graph_nodes,
        num_cached_graph_elements,
    )


def register_min_vids(shuffle_min_vids: torch.Tensor, shuffle_id_offset: int = 0) -> None:
    torch.ops.npc.register_min_vids(shuffle_min_vids, shuffle_id_offset)


def register_multi_machines_scheme(args: argparse.Namespace) -> Optional[torch.Tensor]:
    gpu_remote_worker_map = torch.tensor(args.remote_worker_map).to(f"cuda:{args.local_rank}")
    remote_worker_id = torch.tensor(args.remote_worker_id)
    torch.ops.npc.register_multi_machines_scheme(gpu_remote_worker_map, remote_worker_id)


def _load_subtensor(
    args: argparse.Namespace,
    seeds: torch.Tensor,
) -> torch.Tensor:
    if not args.cross_machine_feat_load:
        return torch.ops.npc.load_subtensor(seeds)
    else:
        # multi-machine load subtensor
        ret = torch.ops.npc.crossmachine_load_subtensor(seeds)
        return ret


def load_subtensor(args, sample_result):
    if args.system == "NP":
        # [0]input_nodes, [1]seeds, [2]blocks, [3]perm, [4]send_offset, [5]recv_offset, [6]inverse_idx
        fsi = NPFeatureShuffleInfo(
            feat_dim=args.num_hidden,
            num_dst=None,
            permutation=sample_result[3],
            send_offset=sample_result[4].to("cpu"),
            recv_offset=sample_result[5].to("cpu"),
            inverse_idx=sample_result[6],
        )
        return (
            sample_result[2],
            _load_subtensor(args, sample_result[0]),
            fsi,
        )
    elif args.system == "SP":
        if args.model == "GAT":
            # [0]input_nodes, [1]seeds, [2]blocks, [3]perm, [4]send_offset, [5]recv_offset
            fsi = NPFeatureShuffleInfo(
                feat_dim=args.num_hidden,
                num_dst=sample_result[2][0].number_of_dst_nodes(),
                permutation=sample_result[3],
                send_offset=sample_result[4].to("cpu"),
                recv_offset=sample_result[5].to("cpu"),
            )
        elif args.shuffle_with_dst:
            # [0]input_nodes [1]seeds, [2]blocks [3]send_size [4]recv_size
            send_sizes = sample_result[3].to("cpu")
            recv_sizes = sample_result[4].to("cpu")
            num_send_dst = send_sizes[0::3].sum().item()
            num_recv_dst = recv_sizes[0::3].sum().item()
            num_dst = [num_send_dst, num_recv_dst]
            total_send_size = num_send_dst + send_sizes[2::3].sum().item()
            total_recv_size = num_recv_dst + recv_sizes[2::3].sum().item()

            fsi = SPFeatureShuffleInfo(
                feat_dim=args.num_hidden,
                send_sizes=send_sizes,
                recv_sizes=recv_sizes,
                num_dst=num_dst,
                total_send_size=total_send_size,
                total_recv_size=total_recv_size,
                shuffle_with_dst=args.shuffle_with_dst,
            )
        else:
            # elif args.model == "GCN" or args.model == "SAGE":
            # [0]input_nodes [1] seeds, [2]blocks [3]send_size [4]recv_size
            send_sizes = sample_result[3].to("cpu")
            recv_sizes = sample_result[4].to("cpu")
            num_dst = sample_result[2][2].number_of_src_nodes()
            total_send_size = send_sizes[1::2].sum().item()
            total_recv_size = recv_sizes[1::2].sum().item()

            fsi = SPFeatureShuffleInfo(
                feat_dim=args.num_hidden,
                send_sizes=send_sizes,
                recv_sizes=recv_sizes,
                num_dst=num_dst,
                total_send_size=total_send_size,
                total_recv_size=total_recv_size,
                shuffle_with_dst=args.shuffle_with_dst,
            )

        return (
            sample_result[2],
            _load_subtensor(args, sample_result[0]),
            fsi,
        )
    elif args.system == "DP":
        # [0]input_nodes, [1]seeds, [2]blocks
        return sample_result[2], _load_subtensor(args, sample_result[0])
    elif args.system == "MP":
        # [0]input_nodes, [1]seeds, [2]blocks, [3]send_size, [4]recv_size
        fsi = MPFeatureShuffleInfo(
            feat_dim=args.num_hidden,
            send_size=sample_result[3].to("cpu"),
            recv_size=sample_result[4].to("cpu"),
        )
        return (
            sample_result[2],
            _load_subtensor(args, sample_result[0]),
            fsi,
        )
    else:
        raise NotImplementedError


def shuffle_seeds(
    seeds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.npc.shuffle_seeds(seeds)


@dataclass
class NPFeatureShuffleInfo(object):
    feat_dim: int
    num_dst: int
    send_offset: List[int]
    recv_offset: List[int]
    permutation: torch.Tensor
    inverse_idx: torch.Tensor


class NPFeatureShuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fsi: NPFeatureShuffleInfo, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.fsi = fsi

        shuffle_result = feat_shuffle(
            input_tensor,
            fsi.send_offset,
            fsi.recv_offset,
            fsi.permutation,
            fsi.feat_dim,
            1,
        )
        return shuffle_result

    @staticmethod
    def backward(ctx, grad_output_tensor: torch.Tensor) -> torch.Tensor:
        fsi: NPFeatureShuffleInfo = ctx.fsi

        shuffle_grad = feat_shuffle(
            grad_output_tensor,
            fsi.recv_offset,
            fsi.send_offset,
            fsi.permutation,
            fsi.feat_dim,
            0,
        )
        return (None, shuffle_grad)


class SPFeatureShuffleGAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fsi: NPFeatureShuffleInfo, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.fsi = fsi

        shuffle_result = feat_shuffle(
            input_tensor,
            fsi.send_offset,
            fsi.recv_offset,
            fsi.permutation,
            fsi.feat_dim,
            1,
        )
        return shuffle_result

    @staticmethod
    def backward(ctx, grad_output_tensor: torch.Tensor) -> torch.Tensor:
        fsi: NPFeatureShuffleInfo = ctx.fsi

        shuffle_grad = feat_shuffle(
            grad_output_tensor,
            fsi.recv_offset,
            fsi.send_offset,
            fsi.permutation,
            fsi.feat_dim,
            0,
        )
        return (None, shuffle_grad)


@dataclass
class SPFeatureShuffleInfo(object):
    feat_dim: int
    send_sizes: torch.Tensor
    recv_sizes: torch.Tensor
    # int or tuple(int,int)
    num_dst: Union[int, Tuple[int, int]]
    total_send_size: int
    total_recv_size: int
    shuffle_with_dst: int = 0


class SPFeatureShuffle(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fsi: SPFeatureShuffleInfo,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        ctx.fsi = fsi
        shuffle_result = sp_feat_shuffle(
            input_tensor,
            fsi.recv_sizes,
            fsi.send_sizes,
            fsi.total_send_size,
            fsi.feat_dim,
            fsi.shuffle_with_dst,
        )
        return shuffle_result

    @staticmethod
    def backward(
        ctx,
        grad_output_tensor: torch.Tensor,
    ) -> torch.Tensor:
        fsi: SPFeatureShuffleInfo = ctx.fsi
        shuffle_grad = sp_feat_shuffle(
            grad_output_tensor,
            fsi.send_sizes,
            fsi.recv_sizes,
            fsi.total_recv_size,
            fsi.feat_dim,
            fsi.shuffle_with_dst,
        )
        return (None, shuffle_grad)


@dataclass
class MPFeatureShuffleInfo(object):
    feat_dim: int
    send_size: torch.Tensor
    recv_size: torch.Tensor


class MPFeatureShuffle(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fsi: MPFeatureShuffleInfo,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        ctx.fsi = fsi
        shuffle_result = mp_feat_shuffle_fwd(
            input_tensor,
            fsi.recv_size,
            fsi.send_size,
            fsi.feat_dim,
        )
        return shuffle_result

    @staticmethod
    def backward(ctx, grad_output_tensor: torch.Tensor) -> torch.Tensor:
        fsi: MPFeatureShuffleInfo = ctx.fsi
        shuffle_grad = mp_feat_shuffle_bwd(
            grad_output_tensor / fsi.recv_size.numel(),
            fsi.send_size,
            fsi.recv_size,
            fsi.feat_dim,
        )
        return (None, shuffle_grad)


def feat_shuffle(
    inputs: torch.Tensor,
    send_offset: torch.Tensor,
    recv_offset: torch.Tensor,
    permutation: torch.Tensor,
    feat_dim: int,
    fwd_flag: int,
):
    return torch.ops.npc.feat_shuffle(inputs, send_offset, recv_offset, permutation, feat_dim, fwd_flag)


def sp_feat_shuffle(
    input: torch.Tensor,
    send_sizes: torch.Tensor,
    recv_sizes: torch.Tensor,
    total_recv_size: int,
    feat_dim: int,
    shuffle_with_dst: int,
):
    return torch.ops.npc.sp_feat_shuffle(input, send_sizes, recv_sizes, total_recv_size, feat_dim, shuffle_with_dst)


def mp_feat_shuffle_fwd(
    input: torch.Tensor,
    send_size: torch.Tensor,
    recv_size: torch.Tensor,
    feat_dim: int,
):
    return torch.ops.npc.mp_feat_shuffle_fwd(input, send_size, recv_size, feat_dim)


def mp_feat_shuffle_bwd(
    input: torch.Tensor,
    send_size: torch.Tensor,
    recv_size: torch.Tensor,
    feat_dim: int,
):
    return torch.ops.npc.mp_feat_shuffle_bwd(input, send_size, recv_size, feat_dim)
