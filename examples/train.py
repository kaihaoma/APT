import csv
import os
import dgl
import npc
import torch
import time
from model import NPCSAGE, DGLSAGE
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import utils
import torchmetrics.functional as MF
import statistics

# torch.profiler
import torch.profiler as profiler
from torch.profiler import record_function, tensorboard_trace_handler


class Result:
    def __init__(self, name):
        self._name = name
        self._max = 0
        self._min = 0
        self._sum = 0
        self._num = 0
        self._empty_flag = True

    def update(self, val):
        self._sum += val
        self._num += 1
        if self._empty_flag:
            self._empty_flag = False
            self._max = val
            self._min = val
        else:
            self._max = max(self._max, val)
            self._min = min(self._min, val)

    def get_max_min_avg(self):
        return [self._max, self._min, self._sum / self._num]


def early_exit():
    dist.barrier()
    utils.cleanup()


def run(rank, args, shared_queue, shared_tensor_list):
    world_size = args.world_size

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    torch.cuda.init()
    utils.setup(rank=rank, world_size=world_size, backend="nccl")
    npc.init(rank=rank, world_size=world_size, shared_queue=shared_queue, init_mp=True)
    device = torch.device(f"cuda:{rank}")

    for ts in shared_tensor_list:
        utils.pin_tensor(ts)
    dist.barrier()
    print(f"[Note]mp.spawn init, {utils.get_total_mem_usage_in_gb()}")
    file_path = f"logs_mem/g_{args.graph_cache_ratio}_node_{args.feat_cache_ratio}_{args.dataset}_w{rank}_of{world_size}.txt"

    partition_data = npc.load_partition(
        args=args, rank=rank, shared_tensor_list=shared_tensor_list
    )
    print(f"[Note]Done load parititon data")

    train_nid = partition_data.train_nid.to(device)
    min_vids = partition_data.min_vids.to(device)
    labels = partition_data.labels.to(device)

    cache_mask = partition_data.cache_mask

    # define define sampler dataloader
    fanout = [int(fanout) for fanout in args.fan_out.split(",")]
    num_layers = len(fanout)
    if args.system == "NPC":
        if args.graph_cache_ratio >= 0.0:
            sampler = npc.MixedNeighborSampler(
                rank=rank, min_vids=min_vids, fanouts=fanout
            )
        else:
            raise NotImplementedError
            # sampler = npc.NPCNeighborSampler(
            #     rank=rank, min_vids=min_vids, fanouts=fanout)
    else:
        sampler = npc.DGLNeighborSampler(fanout, replace=True)
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

    dist.barrier()
    print(f"[Note]Rank#{rank} Done define sampler & dataloader")

    # define model
    if args.system == "NPC":
        training_model = NPCSAGE(
            in_feats=args.input_dim,
            n_hidden=args.num_hidden,
            n_classes=args.num_classes,
            n_layers=num_layers,
            activation=torch.relu,
            dropout=args.dropout,
        ).to(device)
    else:
        training_model = DGLSAGE(
            in_size=args.input_dim,
            hid_size=args.num_hidden,
            out_size=args.num_classes,
            num_layers=num_layers,
        ).to(device)
    print(f"[Note]Rank#{rank} Done define training model")
    if args.world_size > 1:
        training_model = DDP(training_model, device_ids=[device], output_device=device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(training_model.parameters(), lr=0.003)

    print(f"[Note]Rank#{rank} Ready to train")

    training_mode = args.training_mode
    if training_mode == "sampling":
        num_epochs = args.num_epochs
        num_data_types = num_layers + 3
        name = [f"#layer{i}" for i in range(num_layers)]
        name.extend(["#cache", "total load", "cache hit"])
        records = [Result(name[i]) for i in range(num_data_types)]
        for epoch in range(num_epochs):
            for step, sample_result in enumerate(dataloader):
                print(f"[Note]epoch#{epoch}, step#{step}")
                blocks = sample_result[2]
                batch_labels = labels[sample_result[1]]
                pts = cache_mask[sample_result[0]]
                num_sample_nodes = pts.numel()
                num_cached = sum(pts).item()
                num_uva = num_sample_nodes - num_cached
                record = [block.num_src_nodes() for block in blocks]
                record.extend(
                    [num_cached, num_sample_nodes, num_cached / num_sample_nodes]
                )
                for i in range(num_data_types):
                    records[i].update(record[i])
        all_result = []
        for i in range(num_data_types):
            all_result.extend(records[i].get_max_min_avg())
        result_tensor = torch.as_tensor(all_result, device=device)
        tensor_list = [
            torch.zeros(num_data_types * 3, device=device) for _ in range(world_size)
        ]
        dist.all_gather(tensor_list, result_tensor)
        if rank == 0:
            all_result = torch.vstack(tensor_list).tolist()
            print(all_result)
            write_result = []
            for i in range(num_data_types):
                # max
                data_max = max([val[3 * i + 0] for val in all_result])
                data_min = min([val[3 * i + 1] for val in all_result])
                data_avg = statistics.mean([val[3 * i + 1] for val in all_result])
                print(
                    f"[Note]name{i}:{name[i]}: max:{data_max}\t min:{data_min}\t avg:{data_avg}"
                )
                write_result.extend([data_max, data_min, data_avg])

            print(f"[Note]Write to logs file {args.logs_dir}")
            if True:
                with open(args.logs_dir, "a") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    # Tag, System, Dataset, Model, Machines, local batch_size, fanout, hidden size, cache ratio, num_epochs, num batches per epoch, Sampling time, Loading time, Training time,
                    log_info = [
                        args.tag,
                        args.system,
                        args.dataset,
                        args.model,
                        args.machine,
                        args.batch_size,
                        args.fan_out.replace(",", " "),
                        args.num_hidden,
                        args.feat_cache_ratio,
                    ]
                    log_info.extend(write_result)
                    writer.writerow(log_info)

    elif training_mode == "training":
        total_time = [0, 0, 0]
        num_batches_per_epoch = len(dataloader)
        num_epochs = args.num_epochs
        warmup_epochs = args.warmup_epochs
        num_record_epochs = num_epochs - warmup_epochs
        """
        # print(f"[Note]Rank#{rank} record:{num_record_epochs} of {num_epochs}, #batches{num_batches_per_epoch}")
        profiler_log_path = f"./logs_torch_profiler_papers100M/{args.system}_{args.tag}_fanout{args.fan_out}_cache{args.feat_cache_ratio}"
        print(f"[Note]Save to {profiler_log_path}")
        activities = [profiler.ProfilerActivity.CPU,
                      profiler.ProfilerActivity.CUDA]
        schedule = torch.profiler.schedule(
            skip_first=0, wait=1, warmup=2, active=10, repeat=1
        )
        with profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=tensorboard_trace_handler(profiler_log_path),
        ) as prof:
        """
        for epoch in range(num_epochs):
            dist.barrier()
            torch.cuda.synchronize()
            t2 = time.time()
            for step, sample_result in enumerate(dataloader):
                dist.barrier()
                torch.cuda.synchronize()
                t0 = time.time()

                batch_labels = labels[sample_result[1]]
                loading_result = npc.load_subtensor(args, sample_result)
                dist.barrier()
                torch.cuda.synchronize()
                t1 = time.time()

                batch_pred = training_model(loading_result)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # prof.step()
                # accuracy
                """
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
                dist.barrier()
                torch.cuda.synchronize()
                t2 = time.time()
                ms_loading_time = t1 - t0
                ms_training_time = t2 - t1
                if epoch >= warmup_epochs:
                    total_time[0] += ms_samping_time
                    total_time[1] += ms_loading_time
                    total_time[2] += ms_training_time

        if rank == 0:
            avg_time_epoch_sampling = round(
                total_time[0] * 1000.0 / num_record_epochs, 4
            )
            avg_time_epoch_loading = round(
                total_time[1] * 1000.0 / num_record_epochs, 4
            )
            avg_time_epoch_training = round(
                total_time[2] * 1000.0 / num_record_epochs, 4
            )
            print(f"[Note]Write to logs file {args.logs_dir}")
            with open(args.logs_dir, "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                # Tag, System, Dataset, Model, Machines, local batch_size, fanout, hidden size, feat cache ratio, graph cache ratio, num_epochs, num batches per epoch, Sampling time, Loading time, Training time,
                log_info = [
                    args.tag,
                    args.system,
                    args.dataset,
                    args.model,
                    args.machine,
                    args.batch_size,
                    args.fan_out.replace(",", " "),
                    args.num_hidden,
                    args.feat_cache_ratio,
                    args.graph_cache_ratio,
                    num_record_epochs,
                    num_batches_per_epoch,
                    avg_time_epoch_sampling,
                    avg_time_epoch_loading,
                    avg_time_epoch_training,
                ]
                writer.writerow(log_info)

    dist.barrier()
    utils.cleanup()


def tensor_loc(ts: torch.Tensor):
    return f"pin:{ts.is_pinned()}\t shared:{ts.is_shared()}\t device:{ts.device}"


if __name__ == "__main__":
    args = utils.init_args()
    print(args)
    mp.set_start_method("spawn", force=True)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    t0 = time.time()
    dataset_tuple = dgl.load_graphs(args.graph_path)
    loading_dataset_time = round(time.time() - t0, 2)
    graph = dataset_tuple[0][0]
    print(
        f"[Note]Load data from {args.graph_path}, Time:{loading_dataset_time} seconds\n result:{graph}\n\n"
    )
    print(f"[Note]After load whole graph, {utils.get_total_mem_usage_in_gb()}")

    global_train_mask = graph.ndata["_N/train_mask"].bool()

    if len(list(graph.ndata.keys())) > 1:
        # load whole graph
        global_node_feats = graph.ndata["_N/feat"]
        global_labels = graph.ndata["_N/labels"]
    else:
        # load pure graph without node features & labels
        input_dim = args.input_dim
        num_nodes = graph.num_nodes()
        global_node_feats = torch.rand((num_nodes, input_dim), dtype=torch.float32)
        global_labels = torch.randint(args.num_classes, (num_nodes,))

    # clear graph ndata & edata
    for k in list(graph.ndata.keys()):
        graph.ndata.pop(k)
    for k in list(graph.edata.keys()):
        graph.edata.pop(k)

    graph = graph.formats("csc")
    shared_queue = mp.Queue()

    indptr, indices, edges_ids = graph.adj_sparse("csc")
    shared_tensor_list = [
        global_node_feats,
        global_labels,
        global_train_mask,
        indptr,
        indices,
    ]

    for tensor in shared_tensor_list:
        tensor.share_memory_()

    mp.spawn(
        run,
        args=(args, shared_queue, shared_tensor_list),
        nprocs=args.world_size,
        join=True,
    )
