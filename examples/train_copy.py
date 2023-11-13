import csv
import os
import dgl
import npc
import torch
import time
from gat_model import *
from sage_model import *
from gcn_model import *
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.distributed as dist
import utils
import torchmetrics.functional as MF
import statistics
import atexit

# torch.profiler
import torch.profiler as profiler
from torch.profiler import record_function, tensorboard_trace_handler

TEST_EPOCHS = 1
TEST_BATCHES = 15
LIMIT_BATCHES = 200


def run(rank, local_rank, world_size, args, shared_tensor_list):
    print(f"[Note] Starting run on Rank#{rank}, local:{local_rank} of W{world_size}\t debug:{args.debug}")

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
        num_nccl_comms=1,
        device=device,
        init_mp=True,
    )

    for ts in shared_tensor_list:
        utils.pin_tensor(ts)

    partition_data = npc.load_partition(args=args, rank=rank, device=device, shared_tensor_list=shared_tensor_list)
    print(f"[Note]Done load parititon data, {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}")

    train_nid = partition_data.train_nid
    min_vids = partition_data.min_vids
    labels = partition_data.labels

    # define define sampler dataloader
    if args.debug:
        num_nodes = min_vids[-1]
        debug_min_vids = torch.empty(num_nodes, dtype=torch.long, device=device)
        for i in range(len(min_vids) - 1):
            debug_min_vids[min_vids[i] : min_vids[i + 1]] = i
        index = shared_tensor_list[-4].detach().cpu()
        indices = shared_tensor_list[-3].detach().cpu()
        debug_graph = dgl.graph(("csc", (index, indices, [])))
        debug_global_features = shared_tensor_list[-2]
        val_idx = shared_tensor_list[-1]
        # manually rebalance val_idx
        num_total_val = val_idx.numel()
        num_val_per_rank = int(num_total_val // args.world_size)
        rank_val_idx = val_idx[rank * num_val_per_rank : (rank + 1) * num_val_per_rank]

        if rank == 0:
            acc_file = open(f"./logs/accuracy/{args.model}_{args.system}_{args.dataset}_{world_size}.txt", "w")

    sampler = npc.MixedPSNeighborSampler(
        rank=rank,
        world_size=world_size,
        fanouts=args.fan_out,
        system=args.system,
        model=args.model,
        num_total_nodes=min_vids[-1],
        debug_info=(debug_graph, debug_min_vids, num_nodes) if args.debug else None,
    )

    args.batch_size = 1
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
    print(f"[Note]Done define dataloader , {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}")
    if args.debug:
        val_dataloader = dgl.dataloading.DataLoader(
            graph=fake_graph,
            indices=rank_val_idx,
            graph_sampler=sampler,
            device=device,
            use_uva=True,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
    num_batches_per_epoch = len(dataloader)

    dist.barrier()
    print(
        f"[Note]Rank#{rank} Done define sampler & dataloader, #batches:{num_batches_per_epoch}\n {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}"
    )

    # define model
    training_model = DGLGCN(args=args, activation=torch.relu).to(device)
    print(f"[Note] {args.system} training model: {type(training_model)}")
    training_model = DDP(training_model, device_ids=[device], output_device=device)

    print(f"[Note]Rank#{rank} Done define training model\t {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}")
    dist.barrier()
    print(f"[Note]Rank#{rank} Ready to train\t {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}")

    training_mode = args.training_mode
    if training_mode == "training":
        total_time = [0, 0, 0]
        num_epochs = args.num_epochs
        warmup_epochs = args.warmup_epochs
        num_record_epochs = num_epochs - warmup_epochs

        torch.set_deterministic_debug_mode("warn")
        # generating 2nd model & sampler
        tocheck_model = MPGCN(
            args=args,
            activation=torch.relu,
        ).to(device)
        tocheck_model.mp_layers.weight.data = (
            training_model.module.layers[0].weight.clone().detach()[args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
        )
        tocheck_model.ddp_modules.module.layers[0].weight.data = training_model.module.layers[1].weight.clone().detach()
        tocheck_model.ddp_modules.module.layers[1].weight.data = training_model.module.layers[2].weight.clone().detach()

        # check training model
        if rank == 0:
            for name, param in training_model.named_parameters():
                print(f"[Note]name:{name}\t shape:{param.shape}\t dev:{param.device}")
            for name, param in tocheck_model.named_parameters():
                print(f"[Note]name:{name}\t shape:{param.shape}\t dev:{param.device}")

        tocheck_sampler = npc.RefSampler(
            rank=rank,
            world_size=world_size,
            fanouts=args.fan_out,
            system=args.system,
            model=args.model,
            num_total_nodes=min_vids[-1],
            debug_info=(debug_graph, debug_min_vids, num_nodes) if args.debug else None,
        )
        # get one batch of seeds
        sample_result = next(iter(dataloader))
        tocheck_sample_result = tocheck_sampler.sample(None, sample_result[1])
        dp_sample_result = tocheck_sample_result[:3]
        mp_sample_result = tocheck_sample_result[3:]
        batch_labels = labels[sample_result[1]]

        # feature loading
        args.system = "DP"
        partition_data = npc.load_partition(args=args, rank=rank, device=device, shared_tensor_list=shared_tensor_list)
        print(f"[Note]Done load parititon data, {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}")
        dp_loading_result = npc.load_subtensor(args, dp_sample_result)
        args.system = "MP"
        partition_data = npc.load_partition(args=args, rank=rank, device=device, shared_tensor_list=shared_tensor_list)
        print(f"[Note]Done load parititon data, {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}")
        mp_loading_result = npc.load_subtensor(args, mp_sample_result)

        # fwd
        dp_batch_pred = training_model(dp_loading_result)
        mp_batch_pred = tocheck_model(mp_loading_result)

        dp_loss = F.cross_entropy(dp_batch_pred, batch_labels)
        mp_loss = F.cross_entropy(mp_batch_pred, batch_labels)

        dp_loss.backward()
        mp_loss.backward()

        grad = training_model.module.layers[0].weight.grad.data.clone().detach()
        dist.all_reduce(grad)
        grad = grad / args.world_size

        # data1 = tocheck_model.ddp_modules.module.layers[1].weight.grad.data
        # data2 = training_model.module.layers[2].weight.grad.data
        data1 = tocheck_model.mp_layers.weight.grad.data
        data2 = grad[args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1]]
        # print(
        #     f"[Note]rank:{rank}, equal:{torch.allclose(tocheck_model.ddp_modules.module.layers[0].weight.grad.data, training_model.module.layers[1].weight.grad.data)}\t"
        # )
        print(f"[Note]rank:{rank}, equal:{torch.allclose(data1, data2)}\t, {data1}\t {data2}\t")

        # print(f"[Note]equal:{torch.allclose(dp_batch_pred, mp_batch_pred)}\t shape:{dp_batch_pred}\t {mp_batch_pred}")
        # print(f"[Note]loss:{dp_loss}\t {mp_loss}")
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
