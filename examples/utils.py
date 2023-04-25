import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import torch.profiler as profiler
from torch.profiler import record_function, tensorboard_trace_handler

# CPU memory usge
import psutil
import time

GB_TO_BYTES = 1024 * 1024 * 1024


def get_tensor_mem_usage_in_gb(ts: torch.Tensor):
    return ts.numel() * ts.element_size() / GB_TO_BYTES


def get_total_mem_usage_in_gb():
    symem = psutil.virtual_memory()
    total = symem[0] / GB_TO_BYTES
    uses = symem[3] / GB_TO_BYTES
    return f"Mem usage: {uses} / {total}GB"


def kill_proc(p):
    try:
        p.terminate()
    except Exception:
        pass


def get_tensor_info_str(ts):
    return f"max: {torch.max(ts)}\t min: {torch.min(ts)}"


def print_cuda_memory_stats(file_path, tag):
    dist.barrier()
    with open(file_path, 'a') as f:
        f.write(f"-------------------{tag}-------------------\n")
        res = torch.cuda.memory_stats()
        for key, value in res.items():
            f.write(f"{key}\t value:{value}\n")


def save_tensor_to_file(ts, file_path="tmp.pt"):
    print(f"[Note]Save tensor of shape ({ts.shape}) to file {file_path}")
    torch.save(ts, file_path)


def setup(rank, world_size, backend='nccl'):
    print(f'Start rank {rank}, world_size {world_size} backend:{backend}.')
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def clear_graph_data(graph):
    for k in list(graph.ndata.keys()):
        graph.ndata.pop(k)
    for k in list(graph.edata.keys()):
        graph.edata.pop(k)


def cleanup():
    dist.destroy_process_group()


def build_tensorboard_profiler(profiler_log_path):
    print(f"[Note]Save to {profiler_log_path}")
    activities = [profiler.ProfilerActivity.CPU,
                  profiler.ProfilerActivity.CUDA]
    schedule = torch.profiler.schedule(
        skip_first=0, wait=1, warmup=2, active=10, repeat=1
    )
    return profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=tensorboard_trace_handler(profiler_log_path),
    )


def pin_tensor(tensor: torch.Tensor):
    # tensor.share_memory_()
    cudart = torch.cuda.cudart()
    r = cudart.cudaHostRegister(
        tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
    # print(f"[Note]tensor_shared:{tensor.is_shared()}\t pinned:{tensor.is_pinned()}")


def init_args():
    parser = argparse.ArgumentParser(description="NPC args 0.1")
    parser.add_argument("--tag", type=str, default="empty_tag", help="tag")
    parser.add_argument("--logs_dir", type=str,
                        default="./logs/time.csv", help="log file dir")
    parser.add_argument("--machine", type=str,
                        default="4*T4 GPUs", help="machine config")
    parser.add_argument("--dataset", type=str,
                        default="ogbn-products", help="dataset name")
    parser.add_argument("--model", type=str,
                        default="graphsage", help="model name")
    parser.add_argument("--system", type=str, default="NPC",
                        choices=["NPC", "DGL-global", "DSP", "DSP-1hop"])
    parser.add_argument("--feat_mode", type=str,
                        default="inp", help="node features cache mode")
    parser.add_argument("--sorted_idx_path", type=str, default="./sampling_all/uva48",
                        help="path of pre-processed sorted idx")
    parser.add_argument("--cache_memory", type=int, default=-1,
                        help="bytes to cache graph topology and node features")
    parser.add_argument("--feat_cache_ratio", type=float, default=0.,
                        help="Ratio of the number of node features cached in GPU memory")
    parser.add_argument("--graph_cache_ratio", type=float, default=0.,
                        help="Ratio of the number of graph topology cached in GPU memory")
    parser.add_argument("--part_config", default='./ogbn-productsM4/ogbn-products.json',
                        type=str, help='The path to the partition config file')
    parser.add_argument("--graph_path", default='./npc_dataset/ogbn-productsM4.bin',
                        type=str, help='the path to the global shared graph')
    parser.add_argument("--min_vids", default="0,582891,1199182,1829748,2449029",
                        type=str, help='provide min_vids for partition(type: List[int])')
    parser.add_argument("--batch_size", type=int,
                        default=1024, help="local batch size")
    parser.add_argument("--num_epochs", type=int,
                        default=10, help="number of epochs")
    parser.add_argument("--fan_out", type=str,
                        default="5, 10, 15", help="Fanout")
    parser.add_argument("--dropout", default=0.5)
    parser.add_argument("--num_nodes", type=int,
                        default=-1, help="number of total nodes")
    parser.add_argument("--input_dim", type=int,
                        default=100, help="input dimension")
    parser.add_argument("--num_classes", type=int,
                        default=47, help="number of node classes")
    parser.add_argument("--num_hidden", type=int, default=16,
                        help="size of hidden dimension")
    parser.add_argument("--world_size", type=int,
                        default=4, help="number of workers")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--training_mode", type=str,
                        default='training', choices=['training', 'sampling'])
    parser.add_argument("--rebalance_train_nid", type=bool, default=True)

    args = parser.parse_args()
    args.min_vids = list(map(int, args.min_vids.split(",")))
    return args
