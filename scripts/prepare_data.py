import dgl
import torch

# import utils
import os
import torch.multiprocessing as mp
import time
import numpy as np
import sys
import json

sys.path.append("./examples")


# Auto Parallel need partitioned graph and popularity of node


def get_inputdim(ds_name):
    if ds_name == "papers":
        return 128
    elif ds_name == "friendster":
        return 256
    elif ds_name == "igbfull":
        return 128
    else:
        raise NotImplementedError


def get_num_classes(ds_name):
    if ds_name == "papers":
        return 172
    elif ds_name == "friendster":
        return 3
    elif ds_name == "igbfull":
        return 19
    else:
        raise NotImplementedError


def clear_graph_data_except_train_mask(graph):
    # clear graph ndata & edata
    for k in list(graph.ndata.keys()):
        if "train" not in k:
            print(f"[Note]Pop ndata: {k}")
            graph.ndata.pop(k)
    for k in list(graph.edata.keys()):
        print(f"[Note]Pop edata: {k}")
        graph.edata.pop(k)


def load_rawdata(path_prefix="/efs/khma/Projects/NPC/original_dataset"):
    # papers
    import load_data

    """
    graph, _ = load_data.load_ogb_dataset("ogbn-papers100M")
    print(f"[Note]Graph:{graph}")

    save_graph_path = os.path.join(path_prefix, "papers.bin")
    print(f"[Note]Save graph to {save_graph_path}")
    dgl.save_graphs(save_graph_path, [graph])
    """
    # friendster

    # igb-full
    graph, _ = load_data.load_igb()
    print(f"[Note]Graph:{graph}")

    save_graph_path = os.path.join(path_prefix, "igbfull.bin")
    print(f"[Note]Save graph to {save_graph_path}")
    dgl.save_graphs(save_graph_path, [graph])


# load_original_graph ()
def load_original_graph(path_prefix="/efs/khma/Projects/NPC/original_dataset", ds_name="papers"):
    # load preprocessed graph
    load_graph_path = os.path.join(path_prefix, f"{ds_name}.bin")
    print(f"[Note]Load graph from {load_graph_path}")
    graph = dgl.load_graphs(load_graph_path)[0][0]
    return graph


def partition_graph(output_path="/efs/khma/Projects/NPC/npc_dataset", ds_name="papers", world_size_list=[8], part_method="metis"):
    graph = load_original_graph(ds_name=ds_name)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # check train nodes, downscale if necessary
    train_mask = graph.ndata["train_mask"].bool()
    train_nodes = torch.nonzero(train_mask).squeeze()
    n_train_nodes = train_nodes.numel()
    # downscale to upper bound (8 * 1024 * 200)
    ub = 8 * 1024 * 200
    if n_train_nodes > ub:
        print(f"[Note]Downscale train nodes from {n_train_nodes} to {ub}")
        # random downscale to ub
        train_nodes = train_nodes[torch.randperm(n_train_nodes)[:ub]]
        train_mask = torch.zeros((n_nodes,), dtype=torch.uint8)
        train_mask[train_nodes] = 1
        graph.ndata["train_mask"] = train_mask
        print(f"[Note]Downscale train nodes from {n_train_nodes} to {train_nodes.numel()}")

    for world_size in world_size_list:
        output_dir = os.path.join(output_path, f"{ds_name}_w{world_size}_{part_method}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # check whether the graph is already partitioned
        node_part_path = os.path.join(output_dir, f"{ds_name}_{part_method}.pt")
        exist_flag = os.path.exists(node_part_path)

        if not exist_flag:
            print(f"[Note]partition graph with {part_method} method into {world_size} partitions")
            # partition graph
            node_part = dgl.distributed.partition_graph(
                g=graph,
                graph_name=ds_name,
                num_parts=world_size,
                out_path=output_dir,
                part_method=part_method,
                balance_ntypes=graph.ndata["train_mask"],
                balance_edges=True,
            )

            torch.save(node_part, node_part_path)
        else:
            print(f"[Note]Load partitioned graph from {node_part_path}")
            node_part = torch.load(node_part_path)

        # calculate min_vids
        (
            pid,
            counts,
        ) = torch.unique(node_part, return_counts=True)
        sorted, indices = torch.sort(node_part, stable=True)
        reordered_graph = dgl.reorder_graph(graph, node_permute_algo="custom", permute_config={"nodes_perm": indices})
        clear_graph_data_except_train_mask(reordered_graph)
        graph_path = os.path.join(output_dir, f"{ds_name}_w{world_size}_{part_method}.bin")
        dgl.save_graphs(graph_path, [reordered_graph])
        min_vids = torch.cumsum(counts, dim=0)
        min_vids = torch.cat((torch.LongTensor([0]), min_vids))
        print(f"[Note]min_vids:{min_vids}")

        json_dict = {
            "name": ds_name,
            "num_workers": world_size,
            "world_size": world_size,
            "input_dim": get_inputdim(ds_name),
            "num_classes": get_num_classes(ds_name),
            "min_vids": min_vids.tolist(),
            "graph_path": graph_path,
            "number_nodes": reordered_graph.number_of_nodes(),
            "number_edges": reordered_graph.number_of_edges(),
        }

        json_path = os.path.join(output_dir, "configs.json")
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=4)

        # dryrun
        from dryrun import sample_profile, npc_sample_profile

        dryrun_savedir = "/efs/khma/Projects/NPC/sampling_all/ap_simulation"
        full_graph_name = f"{ds_name}_w{world_size}_{part_method}"

        sample_profile(
            graph_name=f"ori_{full_graph_name}",
            graph_path=graph_path,
            world_size=world_size,
            fanout=[10, 10, 10],
            batch_size=1024,
            save_dir=dryrun_savedir,
            run_epochs=10,
            save_to_path=True,
        )

        npc_sample_profile(
            graph_name=f"npc_{full_graph_name}",
            graph_path=graph_path,
            world_size=world_size,
            fanout=[10, 10, 10],
            min_vids=min_vids,
            batch_size=1024,
            save_dir=dryrun_savedir,
            run_epochs=10,
            save_to_path=True,
        )


if __name__ == "__main__":
    # partitioned graph
    # load_rawdata()
    # partition_graph(ds_name="papers", world_size_list=[32], part_method="metis")
    # partition_graph(ds_name="friendster", world_size_list=[32], part_method="metis")
    partition_graph(ds_name="igbfull", world_size_list=[32], part_method="metis")
