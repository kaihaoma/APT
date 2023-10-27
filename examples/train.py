import csv
import os
import dgl
import npc
import torch
import time
from model import NPCSAGE, DGLSAGE, MPSAGE, SPSAGE
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
    backend = "NCCL" if args.nproc_per_node == -1 else None
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
    cache_mask = partition_data.cache_mask

    num_cached_feat_nodes = partition_data.num_cached_feat_nodes
    num_cached_feat_elements = partition_data.num_cached_feat_elements
    num_cached_graph_nodes = partition_data.num_cached_graph_nodes
    num_cached_graph_elements = partition_data.num_cached_graph_elements

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
            acc_file = open(f"./logs/accuracy/{args.system}_{args.dataset}_{world_size}.txt", "w")

    if args.system == "NP":
        sampler = npc.MixedNeighborSampler(
            rank=rank,
            fanouts=args.fan_out,
            debug_info=(debug_graph, debug_min_vids, num_nodes) if args.debug else None,
        )

    else:
        sampler = npc.MixedPSNeighborSampler(
            rank=rank,
            world_size=world_size,
            fanouts=args.fan_out,
            system=args.system,
            num_total_nodes=min_vids[-1],
            debug_info=(debug_graph, debug_min_vids, num_nodes) if args.debug else None,
        )

    fake_graph = dgl.rand_graph(1, 1)
    # fake_graph = None
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

    num_layers = len(args.fan_out)
    # define model

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
        )
    else:
        raise ValueError(f"Invalid system:{args.system}")

    print(f"[Note]Rank#{rank} Done define training model\t {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}")

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
    print(f"[Note]Rank#{rank} Done define training model\t {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}")
    optimizer = torch.optim.Adam(training_model.parameters(), lr=0.001, weight_decay=5e-4)
    dist.barrier()
    print(f"[Note]Rank#{rank} Ready to train\t {utils.get_total_mem_usage_in_gb()}\n {utils.get_cuda_mem_usage_in_gb()}")

    training_mode = args.training_mode
    if training_mode == "training":
        total_time = [0, 0, 0]
        num_epochs = args.num_epochs
        warmup_epochs = args.warmup_epochs
        num_record_epochs = num_epochs - warmup_epochs
        # test
        """
        args.num_epochs = TEST_EPOCHS
        args.num_batches_per_epoch = TEST_BATCHES

        import torch.cuda.nvtx as nvtx

        prof = utils.build_tensorboard_profiler(
            f"./torch_profiler/multi-machines/{args.tag}_{args.system}"
        )
        args.num_epochs = 1
        """
        featloading_log_list = []
        record_list = [[] for _ in range(6)]
        for epoch in range(args.num_epochs):
            epoch_tic_start = utils.get_time()
            # t2 = utils.get_time()
            bt2, t2 = utils.get_time_straggler()
            total_loss = 0
            # nvtx.range_push("Sampling")
            for step, sample_result in enumerate(dataloader):
                # t0 = utils.get_time()
                bt0, t0 = utils.get_time_straggler()
                # nvtx.range_pop()
                # nvtx.range_push("Loading")

                batch_labels = labels[sample_result[1]]
                loading_result = npc.load_subtensor(args, sample_result)
                # check feature loading
                if args.debug:
                    feat_dim_slice = (
                        slice(
                            args.cumsum_feat_dim[rank],
                            args.cumsum_feat_dim[rank + 1],
                            1,
                        )
                        if args.system == "MP"
                        else slice(None)
                    )

                    debug_loading_result = debug_global_features[sample_result[0].to("cpu"), feat_dim_slice]
                    debug_loading_flag = torch.all(torch.eq(loading_result[1].detach().cpu(), debug_loading_result))
                    print(f"[Note]Feature loading check: {debug_loading_flag}")
                    assert debug_loading_flag

                # t1 = utils.get_time()
                bt1, t1 = utils.get_time_straggler()
                # nvtx.range_pop()
                # nvtx.range_push("Training")
                batch_pred = training_model(loading_result)
                loss = F.cross_entropy(batch_pred, batch_labels)
                if args.debug:
                    total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                """
                # accuracy
                accuracy = MF.accuracy(batch_pred, batch_labels)
                dist.all_reduce(loss)
                dist.all_reduce(accuracy)
                loss /= world_size
                accuracy /= world_size
                if rank == 0:
                    print(
                        f"[Note]Rank#{rank} epoch#{epoch},batch#{step} Loss: {loss:.3f}\t acc:{accuracy:.3f}")
                """
                ms_samping_time = t0 - t2
                # t2 = utils.get_time()
                bt2, t2 = utils.get_time_straggler()
                # prof.step()
                # nvtx.range_pop()
                # nvtx.range_push("Sampling")
                ms_loading_time = t1 - t0
                ms_training_time = t2 - t1

                if epoch >= warmup_epochs:
                    total_time[0] += ms_samping_time
                    total_time[1] += ms_loading_time
                    total_time[2] += ms_training_time

                    # recording logging info
                    record_val = [
                        ms_samping_time,
                        t0 - bt0,
                        ms_loading_time,
                        t1 - bt1,
                        ms_training_time,
                        t2 - bt2,
                    ]
                    for i in range(6):
                        record_list[i].append(record_val[i])
                    """
                    # Feat loading
                    num_gpu_cache_hit = torch.sum(
                        cache_mask[sample_result[0].cpu()]
                    ).item()
                    num_total_nodes = sample_result[0].numel()

                    to_send_tensor = torch.FloatTensor(
                        [
                            ms_samping_time,
                            ms_loading_time,
                            ms_training_time,
                            num_gpu_cache_hit,
                            num_total_nodes,
                        ]
                    ).to(device)
                    if args.system != "DP":
                        to_send_tensor = torch.cat(
                            (to_send_tensor, sample_result[-2], sample_result[-1])
                        )
                    output_tensor_list = (
                        [torch.empty_like(to_send_tensor) for _ in range(world_size)]
                        if rank == 0
                        else None
                    )
                    dist.gather(to_send_tensor, output_tensor_list, 0)
                    if rank == 0:
                        output_tensor_list = torch.cat(output_tensor_list).cpu()
                        featloading_log_list.append(output_tensor_list)
                    """
                if step >= LIMIT_BATCHES:
                    break

                t2 = utils.get_time()

            epoch_tic_end = utils.get_time()
            print(f"Rank: {rank} | Epoch: {epoch} | Epoch time: {epoch_tic_end - epoch_tic_start:.3f} s")

            # evaluate
            if args.debug:
                acc = (
                    utils.evaluate(
                        args,
                        training_model,
                        labels,
                        args.num_classes,
                        val_dataloader,
                    ).to(device)
                    / world_size
                )
                dist.reduce(acc, 0)
                if rank == 0:
                    acc_str = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}\n".format(
                        epoch,
                        total_loss / (step + 1),
                        acc.item(),
                    )
                    print(f"[Note]{acc_str}")
                    acc_file.write(acc_str)
        dist.barrier()
        for i in range(6):
            # print record_list[i] max min avg
            maxx = round(max(record_list[i]) * 1000, 2)
            minn = round(min(record_list[i]) * 1000, 2)
            avgg = round(sum(record_list[i]) / len(record_list[i]) * 1000, 2)
            print(f"[Note]Rank#{rank} epoch#{epoch} step#{i} max:{maxx} min:{minn} avg:{avgg}")
        if args.debug and rank == 0:
            acc_file.close()
        if not args.debug and rank == 0 and args.num_epochs > 1:
            # costmodel log

            """
            featloading_log = torch.stack(featloading_log_list)
            print(f"[Note]featloading_log:{featloading_log.shape}")
            featloading_log_mean = torch.mean(featloading_log, dim=0).tolist()
            with open(f"./logs/costmodel/{args.system}.csv", "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(featloading_log_mean)
            """

            avg_time_epoch_sampling = round(total_time[0] * 1000.0 / num_record_epochs, 4)
            avg_time_epoch_loading = round(total_time[1] * 1000.0 / num_record_epochs, 4)
            avg_time_epoch_training = round(total_time[2] * 1000.0 / num_record_epochs, 4)
            print(f"[Note]Write to logs file {args.logs_dir}\t tag:{args.tag}")
            with open(args.logs_dir, "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                # Tag, System, Dataset, Model, Machines, local batch_size, fanout, cache_mode, cache_memory, cache_value, feat cache node, feat cache element, graph cache node, graph cache element, num_epochs, num batches per epoch, Sampling time, Loading time, Training time,
                cache_memory = f"{round(args.cache_memory / (1024*1024*1024), 1)}GB"
                cache_value = args.greedy_feat_ratio if args.cache_mode == "greedy" else args.tag.split("_")[-1]
                avg_epoch_time = round(avg_time_epoch_sampling + avg_time_epoch_loading + avg_time_epoch_training, 2)
                write_tag = f"{args.tag}_{args.system}"
                log_info = [
                    write_tag,
                    # args.system,
                    # args.dataset,
                    # args.model,
                    # args.machine,
                    args.batch_size,
                    args.input_dim,
                    args.fan_out,
                    args.cache_mode,
                    cache_memory,
                    cache_value,
                    num_cached_feat_nodes,
                    num_cached_feat_elements,
                    num_cached_graph_nodes,
                    num_cached_graph_elements,
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
