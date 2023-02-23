import os
import dgl
import npc
import torch
from model import SAGE
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import utils
import sklearn.metrics
import torchmetrics.functional as MF


def run(rank, args, shared_queue, shared_tensor_list):
    world_size = args.world_size
    utils.setup(rank=rank, world_size=world_size, backend='nccl')
    npc.init(rank=rank, world_size=world_size,
             shared_queue=shared_queue, init_mp=True)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Class partition_data: graph, node_feats, edge_feats, gpb, min_vids, train_nid, labels
    partition_data = npc.load_partition(args=args, rank=rank, shared_queue=shared_queue,
                                        shared_tensor_list=shared_tensor_list)
    print(f"[Note]Done load parititon data")
    shared_graph = partition_data.graph
    train_nid = partition_data.train_nid.to(device)
    min_vids = partition_data.min_vids.to(device)
    labels = partition_data.labels.to(device)

    # define define sampler dataloader
    fanout = [int(fanout) for fanout in args.fan_out.split(',')]
    num_layers = len(fanout)
    npc_sampler = npc.NPCNeighborSampler(
        rank=rank, min_vids=min_vids, fanouts=fanout)

    dataloader = dgl.dataloading.DataLoader(
        graph=shared_graph,
        indices=train_nid,
        graph_sampler=npc_sampler,
        device=device,
        use_uva=True,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    dist.barrier()
    print(f"[Note]Rank#{rank} Done define sampler & dataloader")
    # define model
    training_model = SAGE(in_feats=args.input_dim,
                          n_hidden=args.num_hidden,
                          n_classes=args.num_classes,
                          n_layers=num_layers,
                          activation=torch.relu,
                          dropout=args.dropout).to(device)
    print(f"[Note]Rank#{rank} Done define training model")
    if args.world_size > 1:
        training_model = DDP(training_model, device_ids=[
                             device], output_device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(training_model.parameters(), lr=0.003)
    print(f"[Note]Rank#{rank} Ready to train")
    for epoch in range(args.num_epochs):
        for step, (input_nodes, seeds, blocks, perm, send_offset, recv_offset) in enumerate(dataloader):
            # transform global nid to local nid
            batch_labels, input_feats = npc.load_subtensor(
                labels, seeds-min_vids[rank], input_nodes)
            # forward
            fsi = npc.FeatureShuffleInfo(feat_dim=args.num_hidden,
                                         send_offset=send_offset.to("cpu"),
                                         recv_offset=recv_offset.to("cpu"),
                                         permutation=perm,)

            batch_pred = training_model(blocks, input_feats, fsi)
            loss = loss_fcn(batch_pred, batch_labels)
            accuracy = MF.accuracy(batch_pred, batch_labels)
            dist.all_reduce(loss)
            dist.all_reduce(accuracy)
            loss /= world_size
            accuracy /= world_size
            if rank == 0:
                print(
                    f"[Note]Rank#{rank} epoch#{epoch},batch#{step} Loss: {loss:.3f}\t acc:{accuracy:.3f}")
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if step > 30:
                break

    utils.cleanup()


if __name__ == "__main__":
    args = utils.init_args()

    print(args)
    mp.set_start_method("spawn", force=True)

    shared_queue = mp.Queue()

    global_node_features = torch.zeros(
        (args.num_nodes, args.input_dim)).share_memory_()
    global_test_mask = torch.zeros((args.num_nodes,)).share_memory_()
    global_uv = torch.empty((2, args.num_edges),
                            dtype=torch.int64).share_memory_()
    indptr = torch.empty((args.num_nodes+1,),
                         dtype=torch.int64).share_memory_()
    indices = torch.empty((args.num_csr_edges,),
                          dtype=torch.int64).share_memory_()
    shared_tensor_list = [global_node_features,
                          global_test_mask, global_uv, indptr, indices]

    mp.spawn(run,
             args=(args, shared_queue, shared_tensor_list),
             nprocs=args.world_size,
             join=True)
