import csv
import dgl
import npc
import torch
from model import SAGE, GCN, GAT
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
import npc.utils as utils
import atexit


def train_with_strategy(rank, local_rank, world_size, args, shared_tensor_list):
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

    print("Pinning shared tensor.")
    for ts in shared_tensor_list:
        utils.pin_tensor(ts)
    print("Shared tensore are pinned.")

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

    # single-GPU model definition
    if args.model == "SAGE":
        model = SAGE(args=args, activation=torch.relu).to(device)
    elif args.model == "GAT":
        heads = [args.num_heads] * len(args.fan_out)
        model = GAT(args=args, heads=heads, activation=F.elu).to(device)
    elif args.model == "GCN":
        model = GCN(args=args, activation=torch.relu).to(device)
    else:
        raise ValueError(f"{args.model} Not supported")
    # distributed model adaptation
    adapted_model = npc.adapt(args, model, args.system, rank)

    print(
        f"[Note]Rank#{rank} Done define training model\t {utils.get_total_mem_usage_in_gb()}\t {utils.get_cuda_mem_usage_in_gb()}"
    )

    optimizer = torch.optim.Adam(adapted_model.parameters(), lr=1e-3, weight_decay=5e-4)
    dist.barrier()
    print(
        f"[Note]Rank#{rank} Ready to train\t {utils.get_total_mem_usage_in_gb()}\t {utils.get_cuda_mem_usage_in_gb()}"
    )

    total_time = [0, 0, 0]
    num_epochs = args.num_epochs
    warmup_epochs = args.warmup_epochs
    num_record_epochs = num_epochs - warmup_epochs

    for epoch in range(args.num_epochs):
        adapted_model.train()
        t2 = utils.get_time()
        total_sampling_time = 0
        total_loading_time = 0
        total_training_time = 0
        for step, sample_result in enumerate(dataloader):
            t0 = utils.get_time()

            batch_labels = labels[sample_result[1]]
            loading_result = npc.load_subtensor(args, sample_result)

            t1 = utils.get_time()

            batch_pred = adapted_model(loading_result)
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
            cache_memory = f"{round(args.cache_memory / (1024 * 1024 * 1024))}GB"
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
    args, shared_tensor_list, global_nfeat = utils.pre_spawn()

    if args.nproc_per_node == -1:
        print("Train with single machine")
    elif args.nproc_per_node > 0:
        print(
            f"Train with multiple machine with each machine holding {args.nproc_per_node} workers"
        )
    else:
        raise ValueError(args.nproc_per_node)
    world_size = args.world_size
    nproc = world_size if args.nproc_per_node == -1 else args.nproc_per_node
    ranks = args.ranks
    local_ranks = args.local_ranks
    fanout_info = str(args.fan_out).replace(" ", "")
    config_key = args.configs_path.split("/")[-2]
    print(
        f"[Note]procs:{nproc}\t world_size:{world_size}\t ranks:{ranks}\t local_ranks:{local_ranks}"
    )

    # determine the best parallelism strategy for each config
    args.tag = f"{args.model}_nl{args.num_localnode_feats_in_workers}of8_cm{round(args.cache_memory / (1024 * 1024 * 1024))}GB"
    utils.show_args(args)

    best_strategy = npc.determine_best_strategy(
        nproc, ranks, local_ranks, world_size, args, shared_tensor_list, 1, 5
    )

    # start training with the best parallelism strategy
    args.system = best_strategy
    args.tag = f"{best_strategy}_{args.model}_nl{args.num_localnode_feats_in_workers}of8_cm{round(args.cache_memory / (1024 * 1024 * 1024))}GB"
    key = "npc" if best_strategy == "NP" else "ori"
    args.dryrun_file_path = (
        f"{args.caching_candidate_path_prefix}/{key}_{config_key}_{fanout_info}"
    )
    utils.show_args(args)

    # load CPU resident features
    shared_tensors_with_nfeat = utils.determine_feature_reside_cpu(
        args, global_nfeat, shared_tensor_list
    )

    # start training process
    processes = []
    for i in range(nproc):
        p = mp.Process(
            target=train_with_strategy,
            args=(
                ranks[i],
                local_ranks[i],
                world_size,
                args,
                shared_tensors_with_nfeat,
            ),
        )
        atexit.register(utils.kill_proc, p)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
