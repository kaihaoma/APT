import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse

def kill_proc(p):
    try:
        p.terminate()
    except Exception:
        pass

def get_tensor_info_str(ts):
    return f"sum:{torch.sum(ts)} max: {torch.max(ts)}\t min: {torch.min(ts)}"

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
        
def cleanup():
    dist.destroy_process_group()
    
def init_args():
    parser = argparse.ArgumentParser(description="NPC args 0.1")
    parser.add_argument("--tag", type=str, default="empty_tag", help="tag")
    parser.add_argument("--dataset", type=str,
                        default="ogbn-products", help="dataset name")
    parser.add_argument("--model", type=str,
                        default="graphsage", help="model name")
    parser.add_argument("--fanout", type=str, default="25,10", help="fanout")
    parser.add_argument("--feat_mode", type=str,
                        default="inp", help="node features cache mode")
    parser.add_argument("--num_cached_nodes", type=int, default=0,
                        help="number of node features cached in GPU memory")
    parser.add_argument('--part_config', default='/efs/khma/Projects/dsdgl/examples/pytorch/graphsage/ds/data/ogb-product4/ogb-product.json',
                        type=str, help='The path to the partition config file')
    parser.add_argument("--batch_size", type=int,
                        default=1024, help="local batch size")
    parser.add_argument("--num_epochs", type=int,
                        default=10, help="number of epochs")
    parser.add_argument("--fan_out", type=str, default="25,10", help="Fanout")
    parser.add_argument("--dropout", default=0.5)
    #ogbn-products: 2449029 nodes, input_dim 100
    parser.add_argument("--num_nodes", type=int, default=2449029, help="number of nodes")
    parser.add_argument("--input_dim", type=int, default=100, help="input dimensioon")
    parser.add_argument("--world_size", type=int, default=4, help="number of workers")

    args = parser.parse_args()
    return args 