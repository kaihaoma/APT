import csv
import dgl
import npc
import torch
from gat_model import *
from sage_model import *
from gcn_model import *
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.distributed as dist
import npc.utils as utils
import atexit


def run(rank, local_rank, world_size, args, shared_tensor_list):
    print(f"[Note] Starting run on Rank#{rank}, local:{local_rank} of W{world_size}\t")

    device = torch.device(f"cuda:{local_rank}")
    args.rank = rank
    args.local_rank = local_rank
    args.device = device
    backend = "NCCL"
    utils.setup(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        args=args,
        backend=backend,
    )

    node_size = world_size
    if args.nproc_per_node != -1 and args.hybrid:
        node_size = args.nproc_per_node

    npc.init(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        node_size=node_size,
        num_nccl_comms=1,
        device=device,
        init_mp=True,
    )

    for ts in shared_tensor_list:
        utils.pin_tensor(ts)

    partition_data = npc.load_partition(
        args=args, rank=rank, device=device, shared_tensor_list=shared_tensor_list
    )
    print(
        f"[Note]Done load parititon data, {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}"
    )

    train_nid = partition_data.train_nid
    min_vids = partition_data.min_vids
    labels = partition_data.labels

    num_cached_feat_nodes = partition_data.num_cached_feat_nodes
    num_cached_feat_elements = partition_data.num_cached_feat_elements

    # decide shuffle_with_dst
    args.shuffle_with_dst = args.model != "GCN" and args.nproc_per_node != -1

    print(f"[Note]shuffle_with_dst:{args.shuffle_with_dst}")

    if args.system == "NP":
        sampler = npc.MixedNeighborSampler(
            rank=rank,
            fanouts=args.fan_out,
        )

    else:
        sampler = npc.MixedPSNeighborSampler(
            rank=rank,
            world_size=world_size,
            fanouts=args.fan_out,
            system=args.system,
            model=args.model,
            num_total_nodes=min_vids[-1],
            shuffle_with_dst=args.shuffle_with_dst,
        )

    fake_graph = dgl.rand_graph(1, 1)
    dataloader = dgl.dataloading.DataLoader(
        graph=fake_graph,
        indices=train_nid,
        graph_sampler=sampler,
        device=device,
        use_uva=True,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    print(
        f"[Note]Done define dataloader , {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}"
    )
    num_batches_per_epoch = len(dataloader)

    dist.barrier()
    print(
        f"[Note]Rank#{rank} Done define sampler & dataloader, #batches:{num_batches_per_epoch}\n {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}"
    )

    # define model
    if args.model == "SAGE":
        if args.system == "DP":
            training_model = DGLSAGE(
                args=args,
                activation=torch.relu,
            ).to(device)
        elif args.system == "NP":
            training_model = NPCSAGE(
                args=args,
                activation=torch.relu,
            ).to(device)
        elif args.system == "SP":
            training_model = SPSAGE(
                args=args,
                activation=torch.relu,
            ).to(device)
        elif args.system == "MP":
            training_model = MPSAGE(
                args=args,
                activation=torch.relu,
            ).to(device)
            fc_self = torch.empty(args.num_hidden, args.input_dim, device=device)
            fc_neigh = torch.empty(args.num_hidden, args.input_dim, device=device)
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_uniform_(fc_self, gain=gain)
            nn.init.xavier_uniform_(fc_neigh, gain=gain)
            dist.broadcast(fc_self, 0)
            dist.broadcast(fc_neigh, 0)
            training_model.mp_layers.fc_self.weight.data = (
                fc_self[:, args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
                .clone()
                .detach()
                .to(device)
            )
            training_model.mp_layers.fc_neigh.weight.data = (
                fc_neigh[:, args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
                .clone()
                .detach()
                .to(device)
            )
        else:
            raise ValueError(f"Invalid system:{args.system}")
    elif args.model == "GAT":
        heads = [args.num_heads] * len(args.fan_out)
        if args.system == "DP":
            training_model = DGLGAT(
                args=args,
                heads=heads,
                activation=F.elu,
            ).to(device)
        elif args.system == "NP":
            training_model = NPCGAT(
                args=args,
                heads=heads,
                activation=F.elu,
            ).to(device)
        elif args.system == "SP":
            training_model = SPGAT(
                args=args,
                heads=heads,
                activation=F.elu,
            ).to(device)
        elif args.system == "MP":
            training_model = MPGAT(
                args=args,
                heads=heads,
                activation=F.relu,
            ).to(device)
            fc = torch.empty(args.num_hidden * heads[0], args.input_dim, device=device)
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_uniform_(fc, gain=gain)
            dist.broadcast(fc, 0)
            training_model.fc.weight.data = (
                fc[:, args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
                .clone()
                .detach()
                .to(device)
            )
        else:
            raise ValueError(f"Invalid system:{args.system}")
    elif args.model == "GCN":
        if args.system == "DP":
            training_model = DGLGCN(
                args=args,
                activation=torch.relu,
            ).to(device)
        elif args.system == "NP":
            training_model = NPCGCN(
                args=args,
                activation=torch.relu,
            ).to(device)
        elif args.system == "SP":
            training_model = SPGCN(
                args=args,
                activation=torch.relu,
            ).to(device)
        elif args.system == "MP":
            training_model = MPGCN(
                args=args,
                activation=torch.relu,
            ).to(device)
            weight = torch.empty(args.input_dim, args.num_hidden, device=device)
            nn.init.xavier_uniform_(weight)
            dist.broadcast(weight, 0)
            training_model.mp_layers.weight.data = (
                weight[args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
                .clone()
                .detach()
                .to(device)
            )
        else:
            raise ValueError(f"Invalid system:{args.system}")

    print(
        f"[Note]Rank#{rank} Done define training model\t {utils.get_total_mem_usage_in_gb()}\t {utils.get_cuda_mem_usage_in_gb()}"
    )

    if args.world_size > 1:
        if args.system == "MP":
            # check training model
            for name, param in training_model.named_parameters():
                print(f"[Note]name:{name}\t param:{param.shape}\t dev:{param.device}")

        else:
            print(f"[Note] {args.system} training model: {type(training_model)}")
            training_model = DDP(
                training_model,
                device_ids=[device],
                output_device=device,
            )
    optimizer = torch.optim.Adam(
        training_model.parameters(), lr=1e-3, weight_decay=5e-4
    )
    dist.barrier()
    print(
        f"[Note]Rank#{rank} Ready to train\t {utils.get_total_mem_usage_in_gb()}\t {utils.get_cuda_mem_usage_in_gb()}"
    )

    total_time = [0, 0, 0]
    num_epochs = args.num_epochs
    warmup_epochs = args.warmup_epochs
    num_record_epochs = num_epochs - warmup_epochs

    for epoch in range(args.num_epochs):
        training_model.train()
        t2 = utils.get_time()
        total_sampling_time = 0
        total_loading_time = 0
        total_training_time = 0
        for step, sample_result in enumerate(dataloader):
            t0 = utils.get_time()

            batch_labels = labels[sample_result[1]]
            loading_result = npc.load_subtensor(args, sample_result)

            t1 = utils.get_time()

            batch_pred = training_model(loading_result)
            loss = F.cross_entropy(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ms_sampling_time = 1000.0 * (t0 - t2)
            t2 = utils.get_time()
            ms_loading_time = 1000.0 * (t1 - t0)
            ms_training_time = 1000.0 * (t2 - t1)

            total_sampling_time += ms_sampling_time
            total_loading_time += ms_loading_time
            total_training_time += ms_training_time
            if epoch >= warmup_epochs:
                total_time[0] += ms_sampling_time
                total_time[1] += ms_loading_time
                total_time[2] += ms_training_time

            t2 = utils.get_time()

        if args.rank == 0:
            epoch_time = total_sampling_time + total_loading_time + total_training_time
            epoch_info = f"Rank: {rank} | Epoch: {epoch} | Sampling time: {total_sampling_time:.0f}ms | Loading time: {total_loading_time:.0f}ms | Training time: {total_training_time:.0f}ms | Epoch time: {epoch_time:.0f}ms\n"
            print(epoch_info)

    print(f"[Note]Rank#{rank} Train Complete")

    if rank == 0 and args.num_epochs > 1:
        avg_time_epoch_sampling = round(total_time[0] / num_record_epochs, 4)
        avg_time_epoch_loading = round(total_time[1] / num_record_epochs, 4)
        avg_time_epoch_training = round(total_time[2] / num_record_epochs, 4)

        with open(args.logs_dir, "a") as f:
            writer = csv.writer(f, lineterminator="\n")
            cache_memory = f"{round(args.cache_memory / (1024*1024*1024), 1)}GB"
            cache_value = (
                args.greedy_feat_ratio
                if args.cache_mode == "greedy"
                else args.tag.split("_")[-1]
            )
            avg_epoch_time = round(
                avg_time_epoch_sampling
                + avg_time_epoch_loading
                + avg_time_epoch_training,
                2,
            )

            dataset_name = args.configs_path.split("/")[-2]
            write_tag = f"{dataset_name}_{args.tag}"

            # Tag, System, Dataset, Model, Machines, local batch_size, fanout, cache_mode, cache_memory, cache_value, feat cache node, feat cache element, graph cache node, graph cache element, num_epochs, num batches per epoch, Sampling time, Loading time, Training time
            log_info = [
                write_tag,
                world_size,
                args.model,
                args.batch_size,
                args.input_dim,
                args.num_hidden,
                args.num_heads,
                args.fan_out,
                args.cache_mode,
                cache_memory,
                cache_value,
                num_cached_feat_nodes,
                num_cached_feat_elements,
                num_record_epochs,
                num_batches_per_epoch,
                round(avg_time_epoch_sampling, 2),
                round(avg_time_epoch_loading, 2),
                round(avg_time_epoch_training, 2),
                avg_epoch_time,
            ]
            writer.writerow(log_info)
    dist.barrier()
    utils.cleanup()


if __name__ == "__main__":
    args, shared_tensor_list = utils.pre_spawn()
    world_size = args.world_size
    nproc = world_size
    processes = []
    ranks = [i for i in range(nproc)]
    local_ranks = [i for i in range(nproc)]
    for i in range(nproc):
        p = mp.Process(
            target=run,
            args=(ranks[i], local_ranks[i], world_size, args, shared_tensor_list),
        )
        atexit.register(utils.kill_proc, p)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
