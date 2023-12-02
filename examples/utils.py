import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
from typing import List, Tuple
import torchmetrics.functional as MF
from dgl.utils import gather_pinned_tensor_rows
import torch.profiler as profiler
from torch.profiler import record_function, tensorboard_trace_handler

# Loading DGL dataset
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset

# CPU memory usge
import psutil
import time
import dgl
import npc

# threading for gloo all-to-all
import threading

import json
import os
from itertools import accumulate

GB_TO_BYTES = 1024 * 1024 * 1024
MB_TO_BYTES = 1024 * 1024


def evaluate(args, model, labels, num_classes, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, sampling_result in enumerate(dataloader):
        loading_result = npc.load_subtensor(args, sampling_result)
        with torch.no_grad():
            # x = gather_pinned_tensor_rows(feats, input_nodes)
            # y = gather_pinned_tensor_rows(labels, output_nodes)
            y = labels[sampling_result[1]]
            ys.append(y)
            y_hats.append(model(loading_result))

    return MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=num_classes)


def get_tensor_mem_usage_in_gb(ts: torch.Tensor):
    return ts.numel() * ts.element_size() / GB_TO_BYTES


def get_total_mem_usage_in_gb():
    symem = psutil.virtual_memory()
    total = round(symem[0] / GB_TO_BYTES, 2)
    uses = round(symem[3] / GB_TO_BYTES, 2)
    return f"Mem usage: {uses} / {total}GB"


def get_cuda_mem_usage_in_gb():
    cuda_mem_usage = torch.cuda.mem_get_info()
    uses = round((cuda_mem_usage[1] - cuda_mem_usage[0]) / GB_TO_BYTES, 2)
    total = round(cuda_mem_usage[1] / GB_TO_BYTES, 2)
    return f"GPU Mem usage: {uses} / {total}GB"


def kill_proc(p):
    try:
        p.terminate()
    except Exception:
        pass


def print_cuda_memory_stats(file_path, tag):
    with open(file_path, "a") as f:
        f.write(f"-------------------{tag}-------------------\n")
        res = torch.cuda.memory_stats()
        for key, value in res.items():
            f.write(f"{key}\t value:{value}\n")


def pin_tensor(tensor: torch.Tensor):
    cudart = torch.cuda.cudart()
    r = cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
    # print(f"[Note]tensor_shared:{tensor.is_shared()}\t pinned:{tensor.is_pinned()}")


def save_tensor_to_file(ts, file_path="tmp.pt"):
    print(f"[Note]Save tensor of shape ({ts.shape}) to file {file_path}")
    torch.save(ts, file_path)


def get_time():
    torch.cuda.synchronize()
    dist.barrier()
    return time.time()


def get_time_straggler():
    torch.cuda.synchronize()
    t1 = time.time()
    dist.barrier()
    t2 = time.time()
    return t1, t2


def setup(rank, local_rank, world_size, args, backend=None):
    master_port = args.master_port
    master_addr = args.master_addr
    init_method = f"tcp://{master_addr}:{master_port}"
    torch.cuda.set_device(local_rank)
    print(f"[Note]dist setup: rank:{rank}\t world_size:{world_size}\t init_method:{init_method} \t backend:{backend}")
    dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
    print("[Note]Done dist init")


def cleanup():
    dist.destroy_process_group()


def clear_graph_data(graph):
    for k in list(graph.ndata.keys()):
        graph.ndata.pop(k)
    for k in list(graph.edata.keys()):
        graph.edata.pop(k)


def build_tensorboard_profiler(profiler_log_path):
    print(f"[Note]Save to {profiler_log_path}")
    activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
    schedule = torch.profiler.schedule(skip_first=0, wait=1, warmup=2, active=10, repeat=1)
    return profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=tensorboard_trace_handler(profiler_log_path),
    )


def determine_feature_reside_cpu(args, global_node_feats, shared_tensor_list: List):
    total_nodes = args.num_nodes
    input_dim = args.input_dim
    args.cross_machine_feat_load = False
    # multi-machine scenario
    # determine remote_worker_map & remote_worker_id
    # determine local uva feats
    if args.nproc_per_node != -1:
        remote_worker_map = [0 for i in range(args.world_size)]
        remote_worker_id = [0]
        st = args.node_rank * args.nproc_per_node
        en = st + args.nproc_per_node
        for r in range(args.world_size):
            if r < st or r >= en:
                remote_worker_map[r] = len(remote_worker_id)
                remote_worker_id.append(r)

        args.remote_worker_map = remote_worker_map
        args.remote_worker_id = remote_worker_id
        args.num_remote_worker = len(remote_worker_id) - 1
        print(f"[Note]Node#{args.node_rank}\t remote_worker_map:{remote_worker_map}")
        print(f"[Note]Node#{args.node_rank}\t remote_worker_id:{remote_worker_id}\t #remote_worker:{args.num_remote_worker}")

        # determine local uva feats
        # cache all feats
        if args.num_localnode_feats_in_workers == -1:
            args.num_localnode_feats = total_nodes
            localnode_feats_idx = torch.arange(total_nodes)
            localnode_feats = global_node_feats if global_node_feats is not None else torch.rand((total_nodes, input_dim), dtype=torch.float32)
            print(f"[Note]Localnode feats: ALL :{args.system}\t localnode_feats shape:{localnode_feats.shape}")
        elif args.system == "MP":
            args.num_localnode_feats = total_nodes
            localnode_feats_idx = torch.arange(total_nodes)
            mp_input_dim = args.cumsum_feat_dim[en] - args.cumsum_feat_dim[st]
            localnode_feats = (
                global_node_feats[:, args.cumsum_feaet_dim[st] : args.cumsum_feat_dim[en]]
                if global_node_feats is not None
                else torch.rand((args.num_localnode_feats, mp_input_dim), dtype=torch.float32)
            )
            print(f"[Note]MP Localnode feats: ALL :{args.system}\t localnode_feats shape:{localnode_feats.shape}")
        else:
            # part of feats, [local_partition_nods, total_nodes]
            min_req = args.min_vids[en] - args.min_vids[st]
            max_req = total_nodes

            # num_localnode_feats = int(args.num_localnode_feats_in_workers * total_nodes / args.world_size)
            num_localnode_feats = int(args.num_localnode_feats_in_workers * 1024 * 1024 * 1024 / (4 * input_dim))

            args.num_localnode_feats = max(min(num_localnode_feats, max_req), min_req)
            print(
                f"[Note]Localnode feats: PART :{args.system}\t user set:{num_localnode_feats}\t actual: {args.num_localnode_feats}\t range:[{min_req} - {max_req}]"
            )
            # min_req
            if args.num_localnode_feats <= min_req:
                print(f"[Note]Min req for localnode feats:{min_req}\t range:[{args.min_vids[st]} - {args.min_vids[en]}]")
                localnode_feats_idx = torch.arange(args.min_vids[st], args.min_vids[en])
            else:
                # For load dryrun result
                print(f"[Note]load dryrun for sys {args.system} from {args.dryrun_file_path}")
                if args.nproc_per_node != -1 and args.hybrid:
                    # hybrid parallelism
                    local_freq_lists = [torch.load(f"{args.dryrun_file_path}/rk#{r}_epo10.pt")[1] for r in range(st, en)]
                else:
                    local_freq_lists = [torch.load(f"{args.dryrun_file_path}/rk#{r}_epo100.pt")[1] for r in range(st, en)]
                sum_freqs = torch.stack(local_freq_lists, dim=0).sum(dim=0)
                sort_freqs_idx = torch.sort(sum_freqs, descending=True)[1]
                add_sort_freqs_idx = sort_freqs_idx[torch.logical_or(sort_freqs_idx < args.min_vids[st], sort_freqs_idx >= args.min_vids[en])]
                add_num_localnode_feats = args.num_localnode_feats - min_req
                print(f"[Note]add_num_localnode_feats:{add_num_localnode_feats}")
                localnode_feats_idx = torch.cat((torch.arange(args.min_vids[st], args.min_vids[en]), add_sort_freqs_idx[:add_num_localnode_feats]))
            localnode_feats = (
                global_node_feats[localnode_feats_idx]
                if global_node_feats is not None
                else torch.rand((localnode_feats_idx.numel(), input_dim), dtype=torch.float32)
            )
            args.cross_machine_feat_load = True
    else:
        localnode_feats_idx = torch.arange(total_nodes)
        localnode_feats = global_node_feats if global_node_feats is not None else torch.rand((total_nodes, input_dim), dtype=torch.float32)

    print(f"[Note]#localnode_feats: {localnode_feats_idx.numel()} of {total_nodes}")

    localnode_feats_idx.share_memory_()
    localnode_feats.share_memory_()
    return [localnode_feats_idx, localnode_feats] + shared_tensor_list


def load_graph(args):
    """
    if args.debug:
        print(f"[Note]Debug mode, load {args.dataset} by DGL api ")
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root="./dgl_dataset"))
        graph = dataset[0]
        graph = graph.remove_self_loop().add_self_loop()
        val_idx = dataset.val_idx
        return (graph, val_idx)
    """
    graph_path = args.graph_path_all if args.debug else args.graph_path
    print(f"[Note]Load graph from {graph_path}")
    dataset_tuple = dgl.load_graphs(graph_path)
    graph = dataset_tuple[0][0]
    return graph


# output: args, shared_tensor_list
def pre_spawn():
    args = init_args()

    mp.set_start_method("spawn", force=True)

    graph = load_graph(args)

    print(f"[Note]After load whole graph, {get_total_mem_usage_in_gb()}")

    def find_key_in_graph(graph, key):
        for graph_key in list(graph.ndata.keys()):
            if key in graph_key:
                return graph_key
        return None

    global_train_mask = graph.ndata[find_key_in_graph(graph, "train_mask")].bool()

    total_nodes = graph.num_nodes()
    args.num_nodes = total_nodes
    if args.debug:
        # load whole graph
        # global_node_feats = graph.ndata["feat"]
        global_node_feats = graph.ndata[find_key_in_graph(graph, "feat")]
        global_labels = graph.ndata[find_key_in_graph(graph, "label")].long().nan_to_num()
    else:
        # load pure graph without node features & labels
        # global_node_feats = torch.rand((total_nodes, input_dim), dtype=torch.float32)
        global_node_feats = None
        global_labels = torch.randint(args.num_classes, (total_nodes,))

    indptr, indices, edges_ids = graph.adj_tensors("csc")

    shared_tensor_list = [global_labels, global_train_mask, indptr, indices]
    # append val_idx for validation
    if args.debug:
        shared_tensor_list.append(global_node_feats.clone())
        global_val_mask = graph.ndata[find_key_in_graph(graph, "val_mask")].bool()
        val_idx = torch.masked_select(torch.arange(graph.num_nodes()), global_val_mask)
        shared_tensor_list.append(val_idx)

    del edges_ids, graph

    for tensor in shared_tensor_list:
        tensor.share_memory_()

    return args, shared_tensor_list, global_node_feats


def show_args(args: argparse.Namespace) -> None:
    for k, v in args.__dict__.items():
        print(f"[Note]args.{k}:{v}")


def init_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NPC args 0.1")
    parser.add_argument("--tag", type=str, default="empty_tag", help="tag")
    parser.add_argument("--logs_dir", type=str, default="./logs/time.csv", help="log file dir")
    parser.add_argument("--machine", type=str, default="4*T4 GPUs", help="machine config")
    parser.add_argument("--dataset", type=str, default="ogbn-products", help="dataset name")
    parser.add_argument("--model", type=str, default="SAGE", choices=["SAGE", "GCN", "GAT"], help="model name")
    parser.add_argument("--system", type=str, default="NP", choices=["DP", "NP", "SP", "MP"])
    # ---For cost model of adj. & node feat. ---
    parser.add_argument(
        "--cache_mode",
        type=str,
        choices=["costmodel", "greedy", "dryrun", "dp", "none"],
        default="none",
        help="mode of deciding caching graph topo and node feat",
    )
    parser.add_argument("--cache_memory", type=float, default=0, help="bytes to cache graph topology and node features (in gbs)")
    parser.add_argument("--caching_candidate_path_prefix", type=str, default="/efs/khma/Projects/NPC/sampling_all/ap_simulation")
    parser.add_argument("--dp_freqs_path", type=str, default="./dp_freqs/papers_w8_metis.pt", help="dp freq of all nodes for caching adj. & feats")
    # --- ---
    parser.add_argument("--greedy_feat_ratio", type=float, default=0.0, help="greedy caching adjacent list ratio")
    parser.add_argument("--greedy_sorted_idx_path", type=str, help="input path of sorted idx when using greedy mode")
    # --- ---
    parser.add_argument("--feat_cache_ratio", type=float, default=0.0, help="Ratio of the number of node features cached in GPU memory")
    parser.add_argument("--graph_cache_ratio", type=float, default=0.0, help="Ratio of the number of graph topology cached in GPU memory")
    parser.add_argument("--max_cache_graph_nodes", type=int, default=9151541, help="Ratio of the number of graph topology cached in GPU memory")
    parser.add_argument("--max_cache_feat_nodes", type=int, default=-1, help="Ratio of the number of graph topology cached in GPU memory")

    parser.add_argument("--batch_size", type=int, default=1024, help="local batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--fan_out", type=str, default="10,10,10", help="Fanout, in reverse order from k-hop to 1-hop")
    parser.add_argument("--dropout", default=0.0)
    parser.add_argument("--num_nodes", type=int, default=-1, help="number of total nodes")
    parser.add_argument("--input_dim", type=int, default=100, help="input dimension")
    parser.add_argument("--num_classes", type=int, default=47, help="number of node classes")
    parser.add_argument("--num_hidden", type=int, default=16, help="size of hidden dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="number of attention heads, only for GAT")
    parser.add_argument("--world_size", type=int, default=4, help="number of workers")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--training_mode", type=str, default="training", choices=["training", "sampling"])
    parser.add_argument("--rebalance_train_nid", type=bool, default=True)
    # distributed
    parser.add_argument("--master_addr", default="localhost", type=str, help="Master address")
    parser.add_argument("--master_port", default="12345", type=str, help="Master port")
    parser.add_argument("--nproc_per_node", default=-1, type=int, help="Distributed process per node")
    parser.add_argument("--local_rank", default=-1, type=int, help="Distributed local rank")
    parser.add_argument("--node_rank", default=-1, type=int, help="Distributed node rank")

    parser.add_argument(
        "--num_localnode_feats_in_workers", default=-1, type=float, help="number of node feats in local nodes, -1 means feats of all nodes"
    )
    # from config
    parser.add_argument("--configs_path", default="None", type=str, help="the path to the graph configs.json")

    parser.add_argument("--graph_path", default="./npc_dataset/products_w4/ogbn-productsM4.bin", type=str, help="the path to the global shared graph")
    parser.add_argument("--min_vids", default="0,582891,1199182,1829748,2449029", type=str, help="provide min_vids for partition(type: List[int])")
    parser.add_argument("--sorted_idx_path", type=str, default="", help="path of pre-processed sorted idx")
    parser.add_argument(
        "--idx_mem_path", type=str, default="./sampling_all/npc/sorted_idx_mem.pt", help="path of pre-processed mem usage of sorted idx"
    )
    parser.add_argument("--hybrid", type=bool, default=False)
    # For debug
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args(args)

    # load config and set args
    if args.configs_path is not None and os.path.exists(args.configs_path):
        print(f"[Note]configs: {args.configs_path} exists")
        configs = json.load(open(args.configs_path))
        for key, value in configs.items():
            print(f"[Note]Set args {key} = {value}")
            setattr(args, key, value)
    else:
        raise ValueError(f"[Error]configs: {args.configs_path} not exists")

    if not isinstance(args.min_vids, list):
        args.min_vids = list(map(int, args.min_vids.split(",")))
    if not isinstance(args.fan_out, list):
        args.fan_out = list(map(int, args.fan_out.split(",")))

    # cache_memory to bytes
    if args.cache_memory > 0:
        args.cache_memory = args.cache_memory * 1024 * 1024 * 1024

    # define dryrun file path
    args.dryrun_file_path = ""

    # For convenience, we use the same args for all systems
    # input dim for each process
    input_dim = args.input_dim
    world_size = args.world_size
    mp_input_dim_list = [int(input_dim // world_size) for r in range(world_size)]
    lef = input_dim % world_size
    for r in range(lef):
        mp_input_dim_list[r] += 1

    args.mp_input_dim_list = mp_input_dim_list
    args.cumsum_feat_dim = list(accumulate([0] + mp_input_dim_list))

    # set ranks
    if args.nproc_per_node != -1:
        nproc = args.nproc_per_node
        node_rank = args.node_rank
        args.ranks = [i + node_rank * nproc for i in range(nproc)]
        args.local_ranks = [i % nproc for i in range(args.world_size)]
    else:
        nproc = args.world_size
        args.ranks = [i for i in range(nproc)]
        args.local_ranks = [i for i in range(nproc)]
    print(f"[Note]procs:{nproc}\t ranks:{args.ranks}\t local_ranks:{args.local_ranks}")

    # set dgl backend to pytorch
    os.environ["DGLBACKEND"] = "pytorch"
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nproc)
    return args
