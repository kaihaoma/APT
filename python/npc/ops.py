import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Tuple, Optional
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


# -- utils --
# multi-thread cross-machine gloo all-to-all
class PCQueue(object):
    def __init__(self, capacity):
        self.capacity = threading.Semaphore(capacity)
        self.product = threading.Semaphore(0)
        self.buffer = queue.Queue()

    def get(self):
        self.product.acquire()
        item = self.buffer.get()
        self.buffer.task_done()
        self.capacity.release()
        return item

    def put(self, item):
        self.capacity.acquire()
        self.buffer.put(item)
        self.product.release()


class SendThread(threading.Thread):
    # sem_ = threading.Semaphore(0)
    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            to_send_rank, input_list, offset = self.in_queue.get()
            if to_send_rank is None:
                print(f"[Note]None to_send_rank, exit")
                break
            for i, dst in enumerate(to_send_rank):
                dist.send(tensor=input_list[i + offset], dst=dst)
            self.out_queue.put(1)


class RecvThread(threading.Thread):
    # sem_ = threading.Semaphore(0)
    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            to_recv_rank, output_list, offset = self.in_queue.get()
            if to_recv_rank is None:
                print(f"[Note]None to_recv_rank, exit")
                break
            for i, src in enumerate(to_recv_rank):
                dist.recv(tensor=output_list[i + offset], src=src)
            self.out_queue.put(1)


def terminate_multi_machine_comm_list(multi_machines_comm_list):
    if multi_machines_comm_list is None:
        return
    send_thread_in_queue_list = multi_machines_comm_list[2]
    recv_thread_in_queue_list = multi_machines_comm_list[4]
    # put terminte signal (None, None, None)
    for send_in_queue in send_thread_in_queue_list:
        send_in_queue.put((None, None, None))
    for recv_in_queue in recv_thread_in_queue_list:
        recv_in_queue.put((None, None, None))

    send_thread_list = multi_machines_comm_list[0]
    recv_thread_list = multi_machines_comm_list[1]
    for send_thread in send_thread_list:
        send_thread.join()
    for recv_thread in recv_thread_list:
        recv_thread.join()


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
        multi_machine_comm_list,
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
        self.multi_machine_comm_list = multi_machine_comm_list


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

    elif args.cache_mode == "dp":
        raise ValueError
        # temp cache no graph nodes
        print(f"[Note]Caching mode:{args.cache_mode} No cache graph nodes")
        num_cached_graph_nodes = 0
        cache_graph_node_idx = torch.tensor([], dtype=torch.long)

        print(f"[Note]Caching mode:{args.cache_mode} Cache node_feats")

        # Split Para: caching canditate nodes: local nodes
        system_to_id = {"DP": 0, "NP": 1, "SP": 2, "MP": 3}
        node_feats_freqs = torch.load(args.dp_freqs_path)[system_to_id[args.system]][rank][1]
        sorted_node_feats_freq, sorted_idx = torch.sort(node_feats_freqs, descending=True)
        base = 1
        if args.system == "MP":
            base = args.world_size
        num_cached_feat_nodes = int(args.cache_memory * base / (args.input_dim * BYTES_PER_ELEMENT))

        cache_feat_node_idx = sorted_idx[:num_cached_feat_nodes]
    elif args.cache_mode == "dryrun":
        # temp cache no graph nodes
        print(f"[Note]Caching mode:{args.cache_mode} No cache graph nodes")
        num_cached_graph_nodes = 0
        cache_graph_node_idx = torch.tensor([], dtype=torch.long)
        print(f"[Note]Caching mode:{args.cache_mode} Cache node_feats")

        # load dryrun result
        def replace_whitespace(s):
            return str(s).replace(" ", "")

        def sys_to_key(system):
            return "npc" if system == "NP" else "ori"

        config_key = args.configs_path.split("/")[-2]

        # if args.system in ["DP", "NP"]:
        if args.system in ["NP"]:
            caching_candidate_path = (
                f"{args.caching_candidate_path_prefix}/{sys_to_key(args.system)}_{config_key}_{replace_whitespace(args.fan_out)}/rk#{rank}_epo100.pt"
            )
            node_feats_freqs = torch.load(caching_candidate_path)[1]
        else:
            ori_freq_lists = [
                torch.load(
                    f"{args.caching_candidate_path_prefix}/{sys_to_key(args.system)}_{config_key}_{replace_whitespace(args.fan_out)}/rk#{r}_epo100.pt"
                )[1]
                for r in range(args.world_size)
            ]
            sum_freq_lists = torch.stack(ori_freq_lists, dim=0).sum(dim=0)
            if args.system == "MP" or args.system == "DP":
                node_feats_freqs = sum_freq_lists
            elif args.system == "SP":
                node_feats_freqs = torch.zeros(num_total_nodes, dtype=torch.long)
                node_feats_freqs[args.min_vids[rank] : args.min_vids[rank + 1]] = sum_freq_lists[args.min_vids[rank] : args.min_vids[rank + 1]]

        sorted_node_feats_freq, sorted_idx = torch.sort(node_feats_freqs, descending=True)
        del sorted_node_feats_freq
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
    print(f"[Note]{args.cache_mode}: #feats:{num_cached_feat_nodes}\t #graphs:{num_cached_graph_nodes}\t Mem:{get_total_mem_usage_in_gb()}")
    dist.barrier()
    if args.max_cache_feat_nodes >= 0 and num_cached_feat_nodes > args.max_cache_feat_nodes:
        num_cached_feat_nodes = args.max_cache_feat_nodes
        cache_feat_node_idx = cache_feat_node_idx[:num_cached_feat_nodes]

    if args.max_cache_graph_nodes >= 0 and num_cached_graph_nodes > args.max_cache_graph_nodes:
        num_cached_graph_nodes = args.max_cache_graph_nodes
        cache_graph_node_idx = cache_graph_node_idx[:num_cached_graph_nodes]

    print(f"[Note]Force Limit{args.cache_mode}: #feats:{num_cached_feat_nodes}\t #graphs:{num_cached_graph_nodes}\t")
    # cache graph
    if num_cached_graph_nodes > 0:
        cache_indptr = torch.hstack([indptr[pt + 1] - indptr[pt] for pt in cache_graph_node_idx])

        cache_indptr = torch.cat([torch.LongTensor([0]), torch.cumsum(cache_indptr, dim=0)]).to(device)

        cache_indices = torch.cat([indices[indptr[pt] : indptr[pt + 1]] for pt in cache_graph_node_idx]).to(device)

    else:
        cache_indptr = torch.empty(0, dtype=torch.long)
        cache_indices = torch.empty(0, dtype=torch.long)

    print(
        f"[Note]#graph cached:{num_cached_graph_nodes}\t indptr:{cache_indptr.shape}\t indices:{cache_indices.shape}\t Mem:{get_total_mem_usage_in_gb()}"
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
        print(f"[Note]cumsum_feat_dim:{args.cumsum_feat_dim}\t my:{args.cumsum_feat_dim[rank]} - {args.cumsum_feat_dim[rank+1]}")
        cached_feats = localnode_feats[
            cache_feat_node_pos,
            args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1],
        ].to(device)
        feat_dim_offset = args.cumsum_feat_dim[rank]
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

    # register multi-machines scheme & send & recv thread
    multi_machines_comm_list = None
    if args.cross_machine_feat_load:
        register_multi_machines_scheme(args)
        """
        # multi-machines send & recv thread
        num_peers = args.num_remote_worker
        send_thread_list = []
        recv_thread_list = []
        send_thread_in_queue_list = [PCQueue(1) for _ in range(num_peers)]
        send_thread_out_queue_list = [PCQueue(1) for _ in range(num_peers)]
        recv_thread_in_queue_list = [PCQueue(1) for _ in range(num_peers)]
        recv_thread_out_queue_list = [PCQueue(1) for _ in range(num_peers)]
        for i in range(num_peers):
            send_thread_list.append(
                SendThread(send_thread_in_queue_list[i], send_thread_out_queue_list[i])
            )
            recv_thread_list.append(
                RecvThread(recv_thread_in_queue_list[i], recv_thread_out_queue_list[i])
            )
            send_thread_list[i].start()
            recv_thread_list[i].start()

        multi_machines_comm_list = [
            send_thread_list,
            recv_thread_list,
            send_thread_in_queue_list,
            send_thread_out_queue_list,
            recv_thread_in_queue_list,
            recv_thread_out_queue_list,
        ]
        """

    return PartData(
        min_vids,
        global_train_nid,
        global_labels,
        cache_mask,
        num_cached_feat_nodes,
        num_cached_feat_elements,
        num_cached_graph_nodes,
        num_cached_graph_elements,
        multi_machines_comm_list,
    )


def register_min_vids(min_vids: torch.Tensor) -> None:
    torch.ops.npc.register_min_vids(min_vids)


def register_multi_machines_scheme(args: argparse.Namespace) -> Optional[torch.Tensor]:
    gpu_remote_worker_map = torch.tensor(args.remote_worker_map).to(f"cuda:{args.local_rank}")
    remote_worker_id = torch.tensor(args.remote_worker_id)
    print(f"[INFO] gpu_remote_worker_map:{gpu_remote_worker_map}\t remote_worker_id:{remote_worker_id}")
    torch.ops.npc.register_multi_machines_scheme(gpu_remote_worker_map, remote_worker_id)


# custom gloo all-to-all by (gloo.send, gloo.recv)
def thread_send(to_send_rank: List[int], input_list: List[torch.Tensor], offset=0):
    for i, dst in enumerate(to_send_rank):
        dist.send(tensor=input_list[i + offset], dst=dst)


def thread_recv(to_recv_rank: List[int], output_list: List[torch.Tensor], offset=0):
    for i, src in enumerate(to_recv_rank):
        dist.recv(tensor=output_list[i + offset], src=src)


def _load_subtensorV2(
    args: argparse.Namespace,
    seeds: torch.Tensor,
    multi_machine_comm_list,
) -> torch.Tensor:
    if not args.cross_machine_feat_load:
        return torch.ops.npc.load_subtensor(seeds)
    else:
        # multi-machine load subtensor
        rank = args.rank
        world_size = args.world_size
        num_peers = args.num_remote_worker
        remote_worker_id = args.remote_worker_id[1:]
        input_dim = args.input_dim
        # cluster seeds into local_reqs and remote_reqs
        send_size, sorted_req, permutation = torch.ops.npc.cluster_reqs(seeds)
        local_size = send_size[0].item()
        local_req = sorted_req[:local_size]
        remote_req = sorted_req[local_size:].to("cpu")
        send_size = send_size[1 : num_peers + 1].to("cpu")

        # [c++] load_subtensor on local_req on local machines
        local_subtensor = torch.ops.npc.load_subtensor(local_req)

        # all_to_all_single send_size
        input_split_sizes = [0 for _ in range(world_size)]
        output_split_sizes = [0 for _ in range(world_size)]
        for remote_worker in remote_worker_id:
            input_split_sizes[remote_worker] = 1
            output_split_sizes[remote_worker] = 1

        # print(f"[Note]input_split_sizes: {input_split_sizes}\t output_split_sizes: {output_split_sizes}")
        recv_size = torch.empty(num_peers, dtype=torch.long)
        dist.all_to_all_single(
            output=recv_size,
            input=send_size,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
        )
        # print(f"[Note]Rk#{rank} recv_size: {recv_size}\t send_size: {send_size}")

        # all-to-all req
        send_req_size = torch.sum(send_size).item()
        recv_req_size = torch.sum(recv_size).item()
        recv_req = torch.empty(recv_req_size, dtype=torch.int64)
        for i, remote_worker in enumerate(remote_worker_id):
            input_split_sizes[remote_worker] = send_size[i].item()
            output_split_sizes[remote_worker] = recv_size[i].item()

        dist.all_to_all_single(
            output=recv_req,
            input=remote_req,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
        )

        # local CPU index-select feature loading
        remote_req_subtensor = torch.ops.npc.cpu_load_subtensor(recv_req).flatten()

        # all-to-all remote req subtensor
        input_split_sizes = [v * input_dim for v in input_split_sizes]
        output_split_sizes = [v * input_dim for v in output_split_sizes]

        recv_remote_subtensor = torch.empty(send_req_size * input_dim, dtype=torch.float32)

        dist.all_to_all_single(
            output=recv_remote_subtensor,
            input=remote_req_subtensor,
            input_split_sizes=output_split_sizes,
            output_split_sizes=input_split_sizes,
        )
        recv_remote_subtensor = recv_remote_subtensor.to(seeds.device).reshape((-1, input_dim))
        ret = torch.cat([local_subtensor, recv_remote_subtensor])[permutation]
        return ret


def _load_subtensorV3(
    args: argparse.Namespace,
    seeds: torch.Tensor,
    multi_machine_comm_list,
) -> torch.Tensor:
    if not args.cross_machine_feat_load:
        return torch.ops.npc.load_subtensor(seeds)
    else:
        # crossmachine GPU-nccl load subtensor
        """
        ret = torch.ops.npc.crossmachine_load_subtensor(seeds)
        return ret
        """
        # crossmachine CPU-gloo load subtensor

        send_in_queue = multi_machine_comm_list[2]
        send_out_queue = multi_machine_comm_list[3]
        recv_in_queue = multi_machine_comm_list[4]
        recv_out_queue = multi_machine_comm_list[5]
        # multi-machine load subtensor
        num_peers = args.num_remote_worker
        # cluster seeds into local_reqs and remote_reqs
        send_size, sorted_req, permutation = torch.ops.npc.cluster_reqs(seeds)
        local_size = send_size[0].item()

        local_req = sorted_req[:local_size]
        remote_req = sorted_req[local_size:].to("cpu")
        send_size = send_size[1 : num_peers + 1].to("cpu")

        # [c++] load_subtensor on local_req on local machines
        local_subtensor = torch.ops.npc.load_subtensor(local_req)
        # gloo all-to-all sizes
        remote_worker_id = args.remote_worker_id[1:]
        recv_sizes_split = [torch.empty(1, dtype=torch.long) for _ in range(num_peers)]
        send_sizes_split = list(torch.split(send_size, 1))
        send_sizes_list = send_size.tolist()

        send_in_queue[0].put((remote_worker_id, send_sizes_split, 0))
        recv_in_queue[0].put((remote_worker_id, recv_sizes_split, 0))

        send_out_queue[0].get()
        recv_out_queue[0].get()
        # gloo all-to-all remote req
        output_tensor_list = [torch.empty(recv_sizes_split[i].item(), dtype=torch.int64) for i in range(num_peers)]
        input_tensor_list = list(torch.split(remote_req, send_size.tolist()))
        for i in range(num_peers):
            send_in_queue[i].put(([remote_worker_id[i]], input_tensor_list, i))
            recv_in_queue[i].put(([remote_worker_id[i]], output_tensor_list, i))

        for i in range(num_peers):
            send_out_queue[i].get()
            recv_out_queue[i].get()

        # local CPU index-select feature loading
        remote_req_subtensor = torch.ops.npc.cpu_load_subtensor(torch.cat(output_tensor_list))
        # gloo all-to-all remote subtensor
        remote_req_sub_list = list(torch.split(remote_req_subtensor, recv_sizes_split))
        remote_subtensor_list = [torch.empty((send_sizes_list[i], args.input_dim)) for i in range(num_peers)]
        for i in range(num_peers):
            send_in_queue[i].put(([remote_worker_id[i]], remote_req_sub_list, i))
            recv_in_queue[i].put(([remote_worker_id[i]], remote_subtensor_list, i))

        for i in range(num_peers):
            send_out_queue[i].get()
            recv_out_queue[i].get()
        # cat local_subtensor and remote subtensor
        device = seeds.device
        ret = torch.cat(([local_subtensor] + [ts.to(device) for ts in remote_subtensor_list]))[permutation]
        return ret


def load_subtensor(args, sample_result, multi_machines_comm_list):
    if args.system == "NP":
        # [0]input_nodes, [1]seeds, [2]blocks, [3]perm, [4]send_offset, [5]recv_offset
        fsi = NPFeatureShuffleInfo(
            feat_dim=args.num_hidden,
            permutation=sample_result[3],
            send_offset=sample_result[4].to("cpu"),
            recv_offset=sample_result[5].to("cpu"),
        )
        return (
            sample_result[2],
            _load_subtensorV2(args, sample_result[0], multi_machines_comm_list),
            fsi,
        )
    elif args.system == "SP":
        # [0]input_nodes [1] seeds, [2]blocks [3]send_size [4]recv_size
        send_sizes = sample_result[3].to("cpu")
        recv_sizes = sample_result[4].to("cpu")
        num_send_dst = send_sizes[0::3].sum().item()
        num_recv_dst = recv_sizes[0::3].sum().item()
        total_send_size = num_send_dst + send_sizes[2::3].sum().item()
        total_recv_size = num_recv_dst + recv_sizes[2::3].sum().item()
        fsi = SPFeatureShuffleInfo(
            feat_dim=args.num_hidden,
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
            num_send_dst=num_send_dst,
            num_recv_dst=num_recv_dst,
            total_send_size=total_send_size,
            total_recv_size=total_recv_size,
        )

        return (
            sample_result[2],
            _load_subtensorV2(args, sample_result[0], multi_machines_comm_list),
            fsi,
        )
    elif args.system == "DP":
        # [0]input_nodes, [1]seeds, [2]blocks
        return sample_result[2], _load_subtensorV2(args, sample_result[0], multi_machines_comm_list)
    elif args.system == "MP":
        # [0]input_nodes, [1]seeds, [2]blocks, [3]send_size, [4]recv_size
        fsi = MPFeatureShuffleInfo(
            feat_dim=args.num_hidden,
            send_size=sample_result[3].to("cpu"),
            recv_size=sample_result[4].to("cpu"),
        )
        return (
            sample_result[2],
            _load_subtensorV2(args, sample_result[0], multi_machines_comm_list),
            fsi,
        )
    else:
        raise NotImplementedError


# Loader
class MyEvent:
    def __init__(self):
        self.event = torch.cuda.Event(enable_timing=True)

    def to(self, device, non_blocking):
        return self.event

    def record(self):
        self.event.record()

    def elapsed_time(self, end_event):
        return self.event.elapsed_time(end_event)


class MixedNeighborSampler(object):
    def __init__(
        self,
        rank,
        fanouts,
        debug_info=None,
    ):
        self.rank = rank
        self.fir_fanouts = fanouts[1:]
        self.las_fanouts = fanouts[0]
        self.num_layers = len(fanouts)
        self.debug_flag = False
        if debug_info is not None:
            self.debug_graph, self.debug_min_vids, self.num_nodes = debug_info
            self.debug_flag = True
            print(f"[Note]debug:{self.debug_flag}\t graph:{self.debug_graph}\t min_vids:{self.debug_min_vids}\t #nodes:{self.num_nodes}")

    def debug_check(self, src, dst):
        cpu_src = src.detach().cpu()
        cpu_dst = dst.detach().cpu()
        debug_check_flag = torch.all(self.debug_graph.has_edges_between(cpu_src, cpu_dst))
        # print(f"[Note]Sampling check:{debug_check_flag}")
        assert debug_check_flag, "[Error]Sampling debug_check failed"

    def sample(self, graph, seeds):
        output_nodes = seeds
        blocks = []
        event = MyEvent()
        for fanout in reversed(self.fir_fanouts):
            seeds, neighbors = local_sample_one_layer(seeds, fanout)
            replicated_seeds = torch.repeat_interleave(seeds, fanout)
            if self.debug_flag:
                self.debug_check(neighbors, replicated_seeds)
            block_g = dgl.graph((neighbors, replicated_seeds))
            block = dgl.to_block(g=block_g, dst_nodes=seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        # last layer
        # Shape seeds = sum(send_offset)
        # Shape negibors = sum(send_offset) * self.las_fanouts
        event.record()
        seeds, neighbors, perm, send_offset, recv_offset = np_sample_and_shuffle(seeds, self.las_fanouts)
        replicated_seeds = torch.repeat_interleave(seeds, self.las_fanouts)
        if self.debug_flag:
            self.debug_check(neighbors, replicated_seeds)

        block_g = dgl.graph((neighbors, replicated_seeds))
        block = dgl.to_block(g=block_g, dst_nodes=seeds)
        blocks.insert(0, block)
        seeds = block.srcdata[dgl.NID]

        return seeds, output_nodes, blocks, perm, send_offset, recv_offset, event


class MixedPSNeighborSampler(object):
    def __init__(
        self,
        rank,
        world_size,
        fanouts,
        system,
        num_total_nodes,
        debug_info=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.fanouts = fanouts
        self.num_layers = len(fanouts)
        assert system in ["DP", "NP", "MP", "SP"]
        self.system = system
        self.debug_flag = False
        self.num_total_nodes = num_total_nodes
        self.sp_val = (rank << 20) * num_total_nodes
        print(f"[Note]debug_info:{debug_info}")
        if debug_info is not None:
            self.debug_graph, self.debug_min_vids, self.num_nodes = debug_info
            self.debug_flag = True
            print(f"[Note]debug:{self.debug_flag}\t graph:{self.debug_graph}\t min_vids:{self.debug_min_vids}\t #nodes:{self.num_nodes}")

    def sample(self, graph, seeds):
        output_nodes = seeds
        blocks = []

        for layer_id, fanout in enumerate(reversed(self.fanouts)):
            seeds, neighbors = local_sample_one_layer(
                seeds,
                fanout,
            )

            if self.debug_flag:
                replicated_seeds = torch.repeat_interleave(seeds, fanout)
                copy_seeds = replicated_seeds.detach().cpu()
                copy_neighbors = neighbors.detach().cpu()
                flag = torch.all(self.debug_graph.has_edges_between(copy_neighbors, copy_seeds))
                assert flag, f"[Note]Sys{self.system}\t layer_id:{layer_id}\t flag:{flag}"

            if layer_id == self.num_layers - 1:
                if self.system == "DP":
                    sampling_result = ()
                elif self.system == "NP":
                    (
                        shuffled_seeds,
                        neighbors,
                        perm,
                        send_offset,
                        recv_offset,
                    ) = np_sample_and_shuffle(seeds, fanout)
                    sampling_result = (perm, send_offset, recv_offset)

                elif self.system == "SP":
                    map_allnodes = srcdst_to_vir(fanout, seeds, neighbors)
                    sorted_allnodes, perm_allnodes = torch.sort(map_allnodes)
                    num_dst = seeds.numel()
                    map_src = sorted_allnodes[num_dst:]

                    unique_frontier, arange_src = torch.unique(map_src, return_inverse=True)
                    # build block1 by dgl.create_block
                    num_dst = seeds.numel()
                    device = seeds.device
                    arange_dst = torch.arange(num_dst, device=device).repeat_interleave(fanout)
                    block1 = dgl.create_block((arange_src, arange_dst))
                    blocks.insert(0, block1)
                    # send_frontier = [dst, (pack virtual nodes and original)]
                    send_frontier = torch.cat(
                        (
                            seeds[perm_allnodes[:num_dst]],
                            self.sp_val + (map_src % num_dst) * self.num_total_nodes + neighbors[perm_allnodes[num_dst:] - num_dst],
                        )
                    )

                    (
                        recv_dst,
                        recv_seeds,
                        recv_neighbors,
                        send_sizes,
                        recv_sizes,
                    ) = sp_sample_and_shuffle(
                        num_dst,  # num_dst
                        send_frontier,  # send_frontier
                        sorted_allnodes,  # sorted_allnodes
                        unique_frontier,  # unique_frontier
                    )

                    # build block2 by dgl.to_block
                    block2_graph = dgl.graph((recv_neighbors, recv_seeds))
                    block2 = dgl.to_block(block2_graph, include_dst_in_src=False)
                    seeds = torch.cat((recv_dst, block2.srcdata[dgl.NID]))
                    blocks.insert(0, block2)
                    sampling_result = (send_sizes, recv_sizes)

                elif self.system == "MP":
                    seeds, neighbors, send_size, recv_size = mp_sample_shuffle(seeds, neighbors)
                    sampling_result = (send_size, recv_size)

            if layer_id != self.num_layers - 1 or self.system != "SP":
                replicated_seeds = torch.repeat_interleave(seeds, fanout)
                block_g = dgl.graph((neighbors, replicated_seeds))
                block = dgl.to_block(g=block_g, dst_nodes=seeds)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)

        return (
            seeds,
            output_nodes,
            blocks,
        ) + sampling_result


class DGLNeighborSampler(dgl.dataloading.NeighborSampler):
    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__(
            fanouts,
            edge_dir,
            prob,
            mask,
            replace,
            prefetch_node_feats,
            prefetch_labels,
            prefetch_edge_feats,
            output_device,
        )

    def sample(self, g, seed_nodes, exclude_eids=None):  # pylint: disable=arguments-differ
        """Sample a list of blocks from the given seed nodes."""
        result = self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
        return result


def srcdst_to_vir(fanout: int, dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.ops.npc.srcdst_to_vir(fanout, dst, src)


def np_sample_and_shuffle(seeds: torch.Tensor, fanout: int):
    return torch.ops.npc.np_sample_and_shuffle(seeds, fanout)


def local_sample_one_layer(seeds: torch.Tensor, fanout: int, to_virtual: int = 0):
    return torch.ops.npc.local_sample_one_layer(seeds, fanout, to_virtual)


def shuffle_seeds(
    seeds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.npc.shuffle_seeds(seeds)


def sp_sample_and_shuffle(
    num_dst: int,
    send_frontier: torch.Tensor,
    sorted_allnodes: torch.Tensor,
    unique_frontier: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.npc.sp_sample_and_shuffle(num_dst, send_frontier, sorted_allnodes, unique_frontier)


def mp_sample_shuffle(seeds: torch.Tensor, neighs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.npc.mp_sample_shuffle(seeds, neighs)


@dataclass
class NPFeatureShuffleInfo(object):
    feat_dim: int
    send_offset: List[int]
    recv_offset: List[int]
    permutation: torch.Tensor


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


@dataclass
class SPFeatureShuffleInfo(object):
    feat_dim: int
    send_sizes: torch.Tensor
    recv_sizes: torch.Tensor
    num_send_dst: int
    num_recv_dst: int
    total_send_size: int
    total_recv_size: int


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
            grad_output_tensor,
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
):
    return torch.ops.npc.sp_feat_shuffle(input, send_sizes, recv_sizes, total_recv_size, feat_dim)


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
