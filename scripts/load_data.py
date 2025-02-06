import os
import dgl
import torch
import argparse
import torch.multiprocessing as mp
import logging

# from dgl.nn import SAGEConv
from ogb.nodeproppred import DglNodePropPredDataset


# import time
# import utils
def add_loop_and_tobidirect(graph):
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    return graph


def load_igb():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/efs/rjliu/dataset/igb_full",
        help="path containing the datasets",
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        default="full",
        choices=["tiny", "small", "medium", "large", "full"],
        help="size of the datasets",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=19,
        choices=[19, 2983],
        help="number of classes",
    )
    parser.add_argument(
        "--in_memory",
        type=int,
        default=1,
        choices=[0, 1],
        help="0:read only mmap_mode=r, 1:load into memory",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=1,
        choices=[0, 1],
        help="0:nlp-node embeddings, 1:random",
    )
    args = parser.parse_args()
    print(f"[Note]args:{args}")
    logging.warning(f"args:{args}")
    from igb.dataloader import IGB260MDGLDataset

    dataset = IGB260MDGLDataset(args)
    graph = dataset[0]

    return graph, graph.ndata["train_mask"]


def load_ogb_dataset(name):
    data = DglNodePropPredDataset(name=name)
    graph, labels = data[0]
    graph = graph.remove_self_loop().add_self_loop()
    labels = labels[:, 0]
    graph.ndata["labels"] = labels
    splitted_idx = data.get_idx_split()
    # train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_nid = splitted_idx["train"]
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    # val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    # val_mask[val_nid] = True
    # test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    # test_mask[test_nid] = True
    # utils.clear_graph_data(graph)
    # clear graph ndata & edata
    for k in list(graph.ndata.keys()):
        graph.ndata.pop(k)
    for k in list(graph.edata.keys()):
        graph.edata.pop(k)
    graph.ndata["train_mask"] = train_mask
    return graph, train_nid


def save_friendster(input_path="/efs/rjliu/dataset/friendster", save_path="./oridataset/friendster"):
    file_lists = os.listdir(input_path)
    for file in file_lists:
        if file.endswith("bin"):
            print(f"[Note] find friendster bin file: {file}")

            dataset = dgl.load_graphs(os.path.join(input_path, file))
            print(f"[Note]dataset: {dataset}")
            fs_graph = dataset[0][0].remove_self_loop().add_self_loop()
            print(f"[Note]fs_graph:{fs_graph}")

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(f"[Note] save graph to {save_path}")
            dgl.save_graphs(os.path.join(save_path, file), [fs_graph])
            break


def partition_graph(
    name="ogbn-products",
    num_parts=4,
    save_path_prefix="./npc_dataset/",
    partition_flag=False,
):
    out_path = f"{name}M{num_parts}"
    print(f"[Note]Partition Graph {name} into {num_parts}parts")
    graph = load_ogb_dataset(name)[0]
    # graph.ndata['val_mask'] = val_mask
    # graph.ndata['test_mask'] = test_mask
    print(f"[Note]Graph: {graph}")
    print(f"[Note]Ndata: {graph.ndata.keys()}")
    print(f"[Note]edata: {graph.edata.keys()}")
    print(f"[Note]Ndata: {graph.ndata}")
    print(f"[Note]edata: {graph.edata}")
    for key, values in graph.ndata.items():
        print(f"[Note]keys:{key}\t values:{values.shape}")
    short_name = name.split("-")[-1]
    ds_path = os.path.join(save_path_prefix, f"{short_name}_w{num_parts}")
    os.makedirs(ds_path, exist_ok=True)
    if partition_flag:
        print(f"[Note]Save to {ds_path}\t parts:{num_parts}")
        node_parts = dgl.distributed.partition_graph(
            graph,
            graph_name=name,
            num_parts=num_parts,
            out_path=out_path,
            balance_ntypes=graph.ndata["train_mask"],
            balance_edges=True,
        )
        print(f"[Note]node_parts shape:{node_parts.shape}")
        torch.save(node_parts, os.path.join(ds_path, "node_parts.pt"))

        _, counts = torch.unique(node_parts, return_counts=True)
        _, indices = torch.sort(node_parts, stable=True)
        reortered_graph = dgl.reorder_graph(graph, node_permute_algo="custom", permute_config={"nodes_perm": indices})
        print(f"[Note]Reordered graph:{reortered_graph}")

        graph_path = os.path.join(ds_path, f"{short_name}_pure.bin")
        dgl.save_graphs(graph_path, [reortered_graph])
        min_vids = torch.cumsum(counts, dim=0)
        min_vids = torch.cat((torch.LongTensor([0]), min_vids))
        print(f"[Note]min_vids:{min_vids}")
        json_dict = {
            "name": short_name,
            "num_workers": num_parts,
            "world_size": num_parts,
            "input_dim": 100,
            "num_classes": 47,
            "min_vids": min_vids.tolist(),
            "graph_path": graph_path,
        }
        import json

        json_path = os.path.join(ds_path, "configs.json")
        with open(json_path, "w") as f:
            json.dump(json_dict, f)


def init_args():
    parser = argparse.ArgumentParser(description="NPC args 0.1")
    parser.add_argument("--tag", type=str, default="empty_tag", help="tag")
    parser.add_argument("--dataset", type=str, default="ogbn-products", help="dataset name")
    parser.add_argument("--model", type=str, default="graphsage", help="model name")

    parser.add_argument(
        "--part_config",
        default="./ogbn-productsM4/ogbn-products.json",
        type=str,
        help="The path to the partition config file",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="local batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--fan_out", type=str, default="25,10", help="Fanout")
    parser.add_argument("--dropout", default=0.5)
    parser.add_argument("--num_hidden", type=int, default=16, help="size of hidden dimension")
    parser.add_argument("--world_size", type=int, default=4, help="number of workers")

    args = parser.parse_args()
    return args


def save_as_xtrapulp_format(name="ogbn-papers100M", output_dir="./xtrapulp_input"):
    graph, train_nodes = load_ogb_dataset(name=name)

    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()

    # 2 vertex weight
    train_mask = graph.ndata["_N/train_mask"].long().tolist()
    visit_freq = torch.load("ori_0722_ogbn-papers100M_[10,10,10].pt").tolist()

    print(f"[Note] #nodes:{num_nodes}\t #edges:{num_edges}\t train_mask:{len(train_mask)}\t visit_freq:{len(visit_freq)}")
    indptr, indices, edge_ids = graph.adj_sparse("csr")

    # output
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(output_dir, name + ".txt")
    print(f"[Note]write to file:{output_file}")
    from tqdm import tqdm

    with open(output_file, "a") as output_file:
        # header num_nodes, num_edges, 10, num_vwets
        output_file.write(f"{num_nodes} {num_edges} 10 2\n")
        for i in tqdm(range(num_nodes)):
            output_file.write(f"{train_mask[i]} {visit_freq[i]} ")
            neighbors = str(indices[indptr[i] : indptr[i + 1]].tolist())[1:-1]
            output_file.write(neighbors)
            output_file.write("\n")


if __name__ == "__main__":
    # whole_graph_sample()
    # ogbn-papers100M
    # save_friendster()
    # for num_parts in [8, 16]:
    # partition_graph(name="ogbn-products", num_parts=num_parts, partition_flag=True)
    # partition_graph(name="ogbn-papers100M", num_parts=4)
    # partition_graph(name="ogbn-papers100M", num_parts=8)
    # save_as_xtrapulp_format()
    load_igb()
    # npc-preprocess data
    # npc_load_data()
