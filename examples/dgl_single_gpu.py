import argparse
import dgl
import utils
import csv
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
import time
from ogb.nodeproppred import DglNodePropPredDataset


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
            for i in range(1, num_layers - 1):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(pred, label)


def check_tensor(ts):
    return f"max:{torch.max(ts)}\t min:{torch.min(ts)}"


def train(args, device, g, dataset, model):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    fanout = [int(fanout) for fanout in args.fan_out.split(',')]
    num_layers = len(fanout)
    sampler = NeighborSampler(
        # [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        fanout,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch_time = [0, 0]
    num_epochs = 0
    num_batches = 0
    for epoch in range(10):
        model.train()
        total_loss = 0
        batch_time = [0, 0]
        t2 = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            t1 = time.time()
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            # print(f"[Note]pred:{check_tensor(y_hat)}\t")
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_sample_and_loading_time = t1 - t2
            t2 = time.time()
            batch_training_time = t2 - t1
            if epoch > args.warmup_epoch:
                batch_time[0] += batch_sample_and_loading_time
                batch_time[1] += batch_training_time
                num_batches += 1

        if epoch > args.warmup_epoch:
            num_epochs += 1
            for i in range(2):
                epoch_time[i] += batch_time[i]

        '''
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )
        '''
    # write to log file
    print(f"[Note]Record {num_epochs} epochs and {num_batches} batches")
    with open(args.logs_dir, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        log_info = [args.dataset, args.model, args.machine, args.batch_size, args.fan_out]
        log_info.extend([round(1000 * val / num_batches, 2) for val in epoch_time])
        log_info.extend([round(1000 * val / num_epochs, 2) for val in epoch_time])
        writer.writerow(log_info)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode",
    )
    parser.add_argument("--batch_size", default=1024, type=int, help="batch_size")
    parser.add_argument("--log_file", default="./logs/dgl_sample.csv", help="dir to wirte logs")
    parser.add_argument("--num_epochs", type=int,
                        default=10, help="number of epochs")
    parser.add_argument("--fan_out", type=str, default="25,10", help="Fanout")
    parser.add_argument("--dropout", default=0.5)
    parser.add_argument("--num_hidden", type=int, default=16,
                        help="size of hidden dimension")
    parser.add_argument("--dataset", type=str, default="ogbn-products", help="dataset name")
    parser.add_argument("--model", type=str, default="graphsage", help="model name")
    parser.add_argument("--machine", type=str, default="1*T4", help="machine config")
    parser.add_argument("--logs_dir", type=str,
                        default="./logs/dgl_time.csv", help="log file dir")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = init_args()
    args = utils.init_args()
    args.mode = "mixed"
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
    g = dataset[0]
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    print(f"[Note]graph:{g} in_size:{in_size}\t out_size:{out_size}")
    exit(0)
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model

    fanout = [int(fanout) for fanout in args.fan_out.split(',')]
    num_layers = len(fanout)
    model = SAGE(in_size, args.num_hidden, out_size, num_layers).to(device)

    # model training
    print("Training...")
    train(args, device, g, dataset, model)

    # test the model
    if args.test:
        print("Testing...")
        acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
        print("Test Accuracy {:.4f}".format(acc.item()))
