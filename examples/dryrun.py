import os
import dgl
import npc
import torch
import torch.multiprocessing as mp
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

    npc.init(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        node_size=world_size if args.nproc_per_node == -1 else args.nproc_per_node,
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

    # decide shuffle_with_dst
    args.shuffle_with_dst = args.model != "GCN" and args.nproc_per_node != -1

    print(f"[Note]shuffle_with_dst:{args.shuffle_with_dst}")

    if args.system == "NP":
        sampler = npc.MixedNeighborSampler(rank=rank, fanouts=args.fan_out)

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

    dist.barrier()

    counter_list = [torch.zeros(args.num_nodes, dtype=torch.long) for _ in range(2)]
    for epoch in range(args.num_epochs):
        total_sampling_time = 0
        total_loading_time = 0
        total_training_time = 0
        for step, sample_result in enumerate(dataloader):
            counter_list[0][sample_result[1].cpu()] += 1
            counter_list[1][sample_result[0].cpu()] += 1

        if args.rank == 0:
            epoch_time = total_sampling_time + total_loading_time + total_training_time
            epoch_info = f"Rank: {rank} | Epoch: {epoch} | Sampling time: {total_sampling_time:.0f}ms | Loading time: {total_loading_time:.0f}ms | Training time: {total_training_time:.0f}ms | Epoch time: {epoch_time:.0f}ms\n"
            print(epoch_info)

    dryrun_savedir = "../sampling_all/ap_simulation"
    fanout_info = str(args.fan_out).replace(" ", "")
    config_key = args.configs_path.split("/")[-2]
    save_path_prefix = os.path.join(
        dryrun_savedir, f"hybrid_{args.system}_{config_key}_{fanout_info}"
    )
    save_path = os.path.join(save_path_prefix, f"rk#{rank}_epo10.pt")
    print(f"[Note]Rank#{rank},epoch#{epoch} Save to {save_path}")
    torch.save(counter_list, save_path)

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
