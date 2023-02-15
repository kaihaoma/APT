import os
import dgl
import npc
import torch
import model
import time

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import utils


class NeighborSampler():
    def __init__(self, g, fanouts, sample_neighbors):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors

    def sample_blocks(self, g, seed, exclude_eids=None):
        blocks = []
        for layer, fanout in enumerate(self.fanouts):
            # TODO
            frontier, seeds = self.sample_neighbors()
            block = dgl.to_block(g=frontier, dst_nodes=seeds)
            seeds = block.srcdata[dgl.NID]
            print(f"[Note]Layer#{layer}: after seeds:{seeds.shape}")
            blocks.insert(0, block)
        return blocks


def run(rank, args, shared_queue, shared_tensor_list):
    world_size = args.world_size
    utils.setup(rank=rank, world_size=world_size, backend='nccl')
    npc.init(rank=rank, world_size=world_size, shared_queue=shared_queue, init_mp=True)

    npc.load_partition(args=args, rank=rank, shared_tensor_list=shared_tensor_list)

    # define model
    '''
    model = model.SAGE(args.input_dim, args.num_hidden, args.num_classes, len(args.fanout), torch.relu, args.dropout)
    
    for epoch in range(args.num_epochs):
        
        t0 = time.time()
        t1 = time.time()
        #train code 
    '''
    utils.cleanup()


if __name__ == "__main__":
    args = utils.init_args()

    print(args)
    mp.set_start_method("spawn", force=True)

    shared_queue = mp.Queue()

    global_node_features = torch.zeros((args.num_nodes, args.input_dim)).share_memory_()
    global_test_mask = torch.zeros((args.num_nodes,)).share_memory_()
    shared_tensor_list = [global_node_features, global_test_mask]

    mp.spawn(run,
             args=(args, shared_queue, shared_tensor_list),
             nprocs=args.world_size,
             join=True)
