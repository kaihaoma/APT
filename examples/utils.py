import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import torch.profiler as profiler
from torch.profiler import record_function, tensorboard_trace_handler


def kill_proc(p):
    try:
        p.terminate()
    except Exception:
        pass


def get_tensor_info_str(ts):
    return f"max: {torch.max(ts)}\t min: {torch.min(ts)}"


def save_tensor_to_file(ts, file_path="tmp.pt"):
    print(f"[Note]Save tensor of shape ({ts.shape}) to file {file_path}")
    torch.save(ts, file_path)


def setup(rank, world_size, backend='nccl'):
    print(f'Start rank {rank}, world_size {world_size} backend:{backend}.')
    if backend == "nccl":
        torch.cuda.set_device(rank)
    master_addr = "localhost"
    master_port = '12306'
    init_method = 'tcp://{master_addr}:{master_port}'.format(
        master_addr=master_addr, master_port=master_port)
    dist.init_process_group(backend, init_method=init_method,
                            rank=rank, world_size=world_size)


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
    parser.add_argument("--feat_mode", type=str,
                        default="inp", help="node features cache mode")
    parser.add_argument("--cache_ratio", type=float, default=0.,
                        help="Ratio of the number of node features cached in GPU memory")
    parser.add_argument("--part_config", default='./ogbn-productsM4/ogbn-products.json',
                        type=str, help='The path to the partition config file')
    parser.add_argument("--graph_path", default='./npc_dataset/ogbn-productsM4.bin',
                        type=str, help='the path to the global shared graph')
    parser.add_argument("--min_vids", default=[0, 582891, 1199182, 1829748, 2449029],
                        type=list, help='provide min_vids for partition(type: List[int])')
    parser.add_argument("--batch_size", type=int,
                        default=1024, help="local batch size")
    parser.add_argument("--num_epochs", type=int,
                        default=10, help="number of epochs")
    parser.add_argument("--fan_out", type=str,
                        default="15,10,5", help="Fanout")
    parser.add_argument("--dropout", default=0.5)
    # ogbn-products: 2449029 nodes, input_dim 100
    parser.add_argument("--num_nodes", type=int,
                        default=2449029, help="number of nodes")
    parser.add_argument("--input_dim", type=int,
                        default=100, help="input dimension")
    parser.add_argument("--num_classes", type=int,
                        default=47, help="number of node classes")
    parser.add_argument("--num_hidden", type=int, default=16,
                        help="size of hidden dimension")
    parser.add_argument("--world_size", type=int,
                        default=4, help="number of workers")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--rebalance_train_nid", type=bool, default=True)
    args = parser.parse_args()
    return args
