import csv
import os
import dgl
import npc
import torch
import time
from model import SAGE
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import utils
import torchmetrics.functional as MF


def run(rank, args, graph, shared_queue, shared_tensor_list):
    world_size = args.world_size
    utils.setup(rank=rank, world_size=world_size, backend='nccl')
    npc.init(rank=rank, world_size=world_size,
             shared_queue=shared_queue, init_mp=True)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"[Note]args:{args}")
    # Class partition_data: graph, node_feats, edge_feats, gpb, min_vids, train_nid, labels
    partition_data = npc.load_partition(args=args, rank=rank, graph=graph, shared_tensor_list=shared_tensor_list)
    print(f"[Note]Done load parititon data")
    dist.barrier()

    train_nid = partition_data.train_nid.to(device)
    min_vids = partition_data.min_vids.to(device)
    labels = partition_data.labels.to(device)

    # define define sampler dataloader
    fanout = [int(fanout) for fanout in args.fan_out.split(',')]
    num_layers = len(fanout)
    npc_sampler = npc.NPCNeighborSampler(
        rank=rank, min_vids=min_vids, fanouts=fanout)

    dataloader = dgl.dataloading.DataLoader(
        graph=graph,
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
    total_time = [0, 0, 0]
    num_batches_per_epoch = len(dataloader)
    num_epochs = args.num_epochs
    warmup_epochs = args.warmup_epochs
    num_record_epochs = num_epochs - warmup_epochs

    if True:
        for epoch in range(num_epochs):
            dist.barrier()
            torch.cuda.synchronize()
            t2 = time.time()
            for step, (input_nodes, seeds, blocks, perm, send_offset, recv_offset) in enumerate(dataloader):
                dist.barrier()
                torch.cuda.synchronize()
                t0 = time.time()
                # transform global nid to local nid
                # with record_function("## Loading feature ##"):
                batch_labels, input_feats = npc.load_subtensor(
                    labels, seeds, input_nodes)
                # forward
                # with record_function("## Building info ##"):
                fsi = npc.FeatureShuffleInfo(feat_dim=args.num_hidden,
                                             send_offset=send_offset.to("cpu"),
                                             recv_offset=recv_offset.to("cpu"),
                                             permutation=perm,)
                dist.barrier()
                torch.cuda.synchronize()
                t1 = time.time()
                # with record_function("## training ##"):
                batch_pred = training_model(blocks, input_feats, fsi)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # prof.step()
                # accuracy
                '''
                accuracy = MF.accuracy(batch_pred, batch_labels)
                dist.all_reduce(loss)
                dist.all_reduce(accuracy)
                loss /= world_size
                accuracy /= world_size
                if rank == 0:
                    print(
                        f"[Note]Rank#{rank} epoch#{epoch},batch#{step} Loss: {loss:.3f}\t acc:{accuracy:.3f}")
                '''
                ms_samping_time = (t0 - t2)
                dist.barrier()
                torch.cuda.synchronize()
                t2 = time.time()
                ms_loading_time = (t1 - t0)
                ms_training_time = (t2 - t1)
                if epoch >= warmup_epochs:
                    total_time[0] += ms_samping_time
                    total_time[1] += ms_loading_time
                    total_time[2] += ms_training_time

    if rank == 0:
        avg_time_epoch_sampling = round(total_time[0] * 1000. / num_record_epochs, 4)
        avg_time_epoch_loading = round(total_time[1] * 1000. / num_record_epochs, 4)
        avg_time_epoch_training = round(total_time[2] * 1000. / num_record_epochs, 4)
        print(f"[Note]Write to logs file {args.logs_dir}")
        with open(args.logs_dir, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            # Tag, Dataset, Model, Machines, local batch_size, fanout, cache ratio, num_epochs, num batches per epoch, Sampling time, Loading time, Training time,
            log_info = [args.tag, args.dataset, args.model, args.machine, args.batch_size,
                        args.fan_out.replace(',', ' '), args.cache_ratio, num_record_epochs, num_batches_per_epoch, avg_time_epoch_sampling, avg_time_epoch_loading, avg_time_epoch_training]
            writer.writerow(log_info)
    dist.barrier()

    utils.cleanup()


if __name__ == "__main__":
    args = utils.init_args()
    print(args)

    mp.set_start_method("spawn", force=True)
    dataset_tuple = dgl.load_graphs(args.graph_path)
    graph = dataset_tuple[0][0]
    print(f"[Note]Load data from {args.graph_path}, result:{graph}")
    global_node_feats = graph.ndata['_N/feat']
    global_labels = graph.ndata['_N/labels']
    global_train_mask = graph.ndata['_N/train_mask'].bool()
    global_val_mask = graph.ndata['_N/val_mask'].bool()
    global_test_mask = graph.ndata['_N/test_mask'].bool()

    # clear graph ndata & edata
    for k in list(graph.ndata.keys()):
        graph.ndata.pop(k)
    for k in list(graph.edata.keys()):
        graph.edata.pop(k)

    graph = graph.formats("csc")
    graph.pin_memory_()

    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // args.world_size)
    shared_queue = mp.Queue()
    global_node_feats.share_memory_()
    global_labels.share_memory_()
    global_train_mask.share_memory_()
    global_val_mask.share_memory_()
    global_test_mask.share_memory_()
    shared_tensor_list = [global_node_feats, global_labels, global_train_mask, global_val_mask, global_test_mask]

    mp.spawn(run,
             args=(args, graph, shared_queue, shared_tensor_list,),
             nprocs=args.world_size,
             join=True)
