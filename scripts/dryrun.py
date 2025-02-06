import dgl
import torch
from dgl.dataloading import DataLoader, NeighborSampler
import logging
import os


def replace_whitespace(s):
    return str(s).replace(" ", "")


def find_train_mask(ndata_keys):
    for key in ndata_keys:
        if "train_mask" in key:
            return key


def sample_profile(
    graph_name,
    graph_path,
    world_size,
    fanout,
    batch_size=1024,
    save_dir="./",
    run_epochs=10,
    save_to_path=True,
):
    print(
        f"[Note]Func sample_profile graph_name:{graph_name}, graph_path:{graph_path}]\t world_size:{world_size}"
    )
    dataset_tuple = dgl.load_graphs(graph_path)
    graph = dataset_tuple[0][0]
    print(f"[Note] Load graph:{graph}")
    train_mask = find_train_mask(graph.ndata.keys())
    global_train_mask = graph.ndata[train_mask].bool()
    graph = graph.formats("csc")
    num_total_nodes = graph.num_nodes()
    all_train_nid = torch.masked_select(
        torch.arange(num_total_nodes), global_train_mask
    )
    (num_all_train_nids,) = all_train_nid.shape
    num_train_nids_per_rank = num_all_train_nids // world_size
    logging.info(f"#train:{num_all_train_nids}\t per rank:{num_train_nids_per_rank}")
    global_train_nid_list = []
    dataloader_list = []
    counter_list = []
    sampler = NeighborSampler(fanout, replace=True)
    logging.info(f"fanout:{fanout}\t batch_size:{batch_size}")
    for rank in range(world_size):
        global_train_nid_list.append(
            all_train_nid[
                rank * num_train_nids_per_rank : (rank + 1) * num_train_nids_per_rank
            ]
        )
        logging.info(f"train_nid[{rank}] = {global_train_nid_list[rank]}")
        dataloader = DataLoader(
            graph,
            global_train_nid_list[rank],
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        num_batches_per_epoch = len(dataloader)
        dataloader_list.append(dataloader)
        # 0: graph 1: feat
        counter_list.append(
            [torch.zeros(num_total_nodes, dtype=torch.long) for _ in range(2)]
        )
    save_path_prefix = os.path.join(
        save_dir, f"{graph_name}_{replace_whitespace(fanout)}"
    )
    logging.info(f"[Note]#batch per epoch:{num_batches_per_epoch}")
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    elif save_to_path:
        print(f"[Note]sampling path exists!")
        return

    for epoch in range(1, run_epochs + 1):
        for rank in range(world_size):
            for input_nodes, output_nodes, blocks in dataloader_list[rank]:
                dst_nodes = blocks[0].dstdata[dgl.NID]
                src_nodes = blocks[0].srcdata[dgl.NID]
                num_dst = dst_nodes.numel()
                num_src = src_nodes.numel()
                logging.info(f"dst:{dst_nodes}\t src:{src_nodes}")
                logging.info(f"#dst:{num_dst}\t #src:{num_src}")

                counter_list[rank][0][dst_nodes] += 1
                counter_list[rank][1][src_nodes] += 1

    if save_to_path:
        for rank in range(world_size):
            save_path = os.path.join(save_path_prefix, f"rk#{rank}_epo100.pt")
            print(f"[Note]Rank#{rank},epoch#{epoch} Save to {save_path}")
            torch.save(counter_list[rank], save_path)


def npc_sample_profile(
    graph_name,
    graph_path,
    world_size,
    fanout,
    min_vids=None,
    batch_size=1024,
    save_dir="./",
    run_epochs=100,
    save_to_path=True,
):
    logging.info(f"min_vids len:{len(min_vids)}\t min_vids:{min_vids}")
    dataset_tuple = dgl.load_graphs(graph_path)
    graph = dataset_tuple[0][0]
    print(f"[Note] Load graph:{graph}")
    train_mask = find_train_mask(graph.ndata.keys())
    global_train_mask = graph.ndata[train_mask].bool()
    graph = graph.formats("csc")
    num_total_nodes = graph.num_nodes()
    all_train_nid = torch.masked_select(
        torch.arange(num_total_nodes), global_train_mask
    )
    (num_all_train_nids,) = all_train_nid.shape
    num_train_nids_per_rank = num_all_train_nids // world_size
    global_train_nid_list = []
    dataloader_list = []
    counter_list = []
    pre_fanout = fanout[1:]
    las_fanout = fanout[:1]
    sampler = NeighborSampler(pre_fanout, replace=True)
    las_sampler = NeighborSampler(las_fanout, replace=True)
    logging.info(f"#train:{num_all_train_nids}\t per rank:{num_train_nids_per_rank}")
    logging.info(
        f"fanout: {fanout} = {pre_fanout} + {las_fanout}\t batch_size:{batch_size}"
    )
    for rank in range(world_size):
        global_train_nid_list.append(
            all_train_nid[
                rank * num_train_nids_per_rank : (rank + 1) * num_train_nids_per_rank
            ]
        )
        logging.info(f"train_nid[{rank}] = {global_train_nid_list[rank]}")
        dataloader = DataLoader(
            graph,
            global_train_nid_list[rank],
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        dataloader_list.append(dataloader)
        num_batches_per_epoch = len(dataloader)
        # 0: graph 1: feat
        counter_list.append(
            [torch.zeros(num_total_nodes, dtype=torch.long) for _ in range(2)]
        )

    for epoch in range(1, run_epochs + 1):
        iter_dataloader = [iter(dataloader_list[rank]) for rank in range(world_size)]
        for bid in range(num_batches_per_epoch):
            seed_nodes_list = []
            for rank in range(world_size):
                (input_nodes, output_nodes, blocks) = next(iter_dataloader[rank])
                logging.info(f"Rk#{rank} input:{input_nodes}\t blocks:{blocks}")
                seed_nodes_list.append(input_nodes)

            seed_nodes_list = torch.cat(seed_nodes_list)
            for rank in range(world_size):
                local_nodes = seed_nodes_list[
                    torch.logical_and(
                        seed_nodes_list >= min_vids[rank],
                        seed_nodes_list < min_vids[rank + 1],
                    )
                ]
                logging.info(f"Rk#{rank}: local_nodes:{local_nodes}")
                las_seed_nodes, las_output_nodes, last_blocks = las_sampler.sample(
                    graph, local_nodes
                )
                logging.info(f"Rk#{rank}: las_seed_nodes:{las_seed_nodes}")
                counter_list[rank][0][local_nodes] += 1
                counter_list[rank][1][las_seed_nodes] += 1

    if save_to_path:
        save_path_prefix = os.path.join(
            save_dir, f"{graph_name}_{replace_whitespace(fanout)}"
        )
        if not os.path.exists(save_path_prefix):
            os.makedirs(save_path_prefix)
        for rank in range(world_size):
            save_path = os.path.join(save_path_prefix, f"rk#{rank}_epo100.pt")
            print(f"[Note]Rank#{rank},epoch#{epoch} Save to {save_path}")
            torch.save(counter_list[rank], save_path)


def dryrun():
    world_size = 4
    # ds_name_list = ["papers", "friendster", "igbfull"]
    ds_name_list = ["papers"]
    # fanout_list = [[10, 15], [10, 25]]
    # fanout_list = [[5,10], [20,20,20]]
    # fanout_list = [[20,20,20],[10,10,10,10]]
    fanout_list = [[15, 15, 15]]
    for fanout in fanout_list:
        for ds_name in ds_name_list:
            # ds_name = "igbfull"
            part_method = "metis"
            # graph_path = "../npc_dataset/friendster_w4_metis/friendster_w4_metis.bin"
            dryrun_savedir = "../sampling_all/ap_simulation"
            # backup_dryrun_savedir = "../sampling_all/ap_simulation2"
            full_graph_name = f"{ds_name}_w{world_size}_{part_method}"
            configs_path = f"../npc_dataset_acc/{full_graph_name}/configs.json"
            # graph_path = f"../npc_dataset/{full_graph_name}/{full_graph_name}.bin"
            import json

            configs = json.load(open(configs_path, "r"))
            graph_path = configs["graph_path"]
            min_vids = configs["min_vids"]
            if not isinstance(min_vids, list):
                min_vids = list(map(int, min_vids.split(",")))
            print(f"[Note]graph_path:{graph_path}")
            print(f"[Note]min_vids:{min_vids}")

            sample_profile(
                graph_name=f"ori_{full_graph_name}",
                graph_path=graph_path,
                world_size=world_size,
                fanout=fanout,
                batch_size=1024,
                save_dir=dryrun_savedir,
                run_epochs=10,
                save_to_path=True,
            )

            # min_vids = [0, 15720465, 32776440, 48553474, 65608366]
            npc_sample_profile(
                graph_name=f"npc_{full_graph_name}",
                graph_path=graph_path,
                world_size=world_size,
                fanout=fanout,
                min_vids=min_vids,
                batch_size=1024,
                save_dir=dryrun_savedir,
                run_epochs=10,
                save_to_path=True,
            )


if __name__ == "__main__":
    dryrun()
