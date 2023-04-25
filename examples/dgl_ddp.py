import csv
import time
import os
import utils
import dgl.nn as dglnn

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from model import DGLSAGE
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel


def evaluate(model, g, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(
    proc_id, device, g, nid, model, use_uva, batch_size=2**16
):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference(g, device, batch_size, use_uva)
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(pred, labels)
        print("Test Accuracy {:.4f}".format(acc.item()))


def train(proc_id, nprocs, device, g, train_idx, val_idx, model, fanout, args):
    use_uva = args.mode == "mixed"
    batch_size = args.batch_size
    use_evaluate = False
    logs_dir = args.logs_dir
    sampler = NeighborSampler(
        fanout, prefetch_node_feats=["feat"], prefetch_labels=["label"], replace=True,
    )
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    total_time = [0, 0]
    num_epochs = args.num_epochs
    warmup_epochs = args.warmup_epochs
    num_record_epochs = num_epochs - warmup_epochs
    num_batches_per_epoch = len(train_dataloader)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_time = [0, 0]
        torch.cuda.synchronize()
        dist.barrier()
        t2 = time.time()
        # (input_nodes, output_nodes, blocks)
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            torch.cuda.synchronize()
            dist.barrier()
            t1 = time.time()
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"].long()
            y_hat = model((blocks, x,))
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
            batch_sample_and_loading_time = t1 - t2
            torch.cuda.synchronize()
            dist.barrier()
            t2 = time.time()
            batch_training_time = t2 - t1
            if epoch >= warmup_epochs:
                total_time[0] += batch_sample_and_loading_time
                total_time[1] += batch_training_time

        if use_evaluate:
            acc = evaluate(model, g, val_dataloader).to(device) / nprocs
            dist.reduce(acc, 0)
            if proc_id == 0:
                print(
                    "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                        epoch, total_loss / (it + 1), acc.item()
                    )
                )
    if proc_id == 0:
        print(f"[Note] Write to file {logs_dir}")
        avg_time_epoch_sampling_and_loading = round(
            total_time[0] * 1000. / num_record_epochs, 4)
        avg_time_epoch_training = round(
            total_time[1] * 1000. / num_record_epochs, 4)
        with open(logs_dir, 'a') as f:
            # Tag, Dataset, Model, Machines, local batch_size, fanout, cache ratio, num_epochs, num batches per epoch, Sampling time and Loading time, Training time,
            writer = csv.writer(f, lineterminator='\n')
            log_info = [args.tag, args.dataset, args.model, args.machine, args.batch_size,
                        args.fan_out.replace(',', ' '), num_record_epochs, num_batches_per_epoch, avg_time_epoch_sampling_and_loading, avg_time_epoch_training]
            writer.writerow(log_info)


def run(proc_id, g, data, args):
    # find corresponding device for my rank
    device = f"cuda:{proc_id}"
    torch.cuda.set_device(device)
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=args.world_size,
        rank=proc_id,
    )
    out_size, train_idx, val_idx, test_idx = data
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    g = g.to(device if args.mode == "puregpu" else "cpu")
    # create GraphSAGE model (distributed)
    in_size = g.ndata["feat"].shape[1]
    fanout = [int(fanout) for fanout in args.fan_out.split(',')]
    num_layers = len(fanout)
    model = DGLSAGE(in_size, args.num_hidden, out_size, num_layers).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # training + testing
    train(proc_id, args.world_size, device, g,
          train_idx, val_idx, model, fanout, args)
    # layerwise_infer(proc_id, device, g, test_idx, model, use_uva)
    # cleanup process group
    dist.destroy_process_group()


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["mixed", "puregpu"],
        help="Training mode. 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    args = parser.parse_args()

    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    '''
    args = utils.init_args()
    args.mode = "mixed"
    print(f"Training in {args.mode} mode using {args.world_size} GPU(s)")
    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
    g = dataset[0]
    # avoid creating certain graph formats in each sub-process to save momory
    g = g.formats("csc")
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // args.world_size)
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )

    mp.spawn(run, args=(g, data, args), nprocs=args.world_size)
