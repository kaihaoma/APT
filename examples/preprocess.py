import dgl
import torch
import utils
import torch.multiprocessing as mp


def data_preprocess(args):
    world_size = args.world_size
    num_nodes = args.num_nodes
    input_dim = args.input_dim
    global_labels = torch.empty((num_nodes,), dtype=torch.long)
    global_train_mask = torch.empty((num_nodes,), dtype=torch.bool)
    # global_val_mask = torch.empty((num_nodes,), dtype=torch.bool)
    # global_test_mask = torch.empty((num_nodes,), dtype=torch.bool)
    global_node_feats = torch.empty((num_nodes, input_dim), dtype=torch.float32)
    all_u = []
    all_v = []
    for rank in range(world_size):
        graph, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(
            args.part_config, rank)
        # extract data and delete graph node&edge data
        # [NOTE] graph.ndata
        # 'part_id', '_ID', 'inner_node'
        # [NOTE] graph.edata
        # '_ID', 'inner_edge'
        # [NOTE] node_feats
        # '_N/feat', '_N/labels', '_N/train_mask', '_N/test_mask', '_N/val_mask'
        print(f"[Note]Load from partition#{rank}\t graph:{graph}\t edge_feats:{edge_feats}\t node_feats:{node_feats}")
        local_node_feats = node_feats['_N/feat']
        local_train_mask = node_feats['_N/train_mask'].bool()
        # local_val_mask = node_feats['_N/val_mask'].bool()
        # local_test_mask = node_feats['_N/test_mask'].bool()
        labels = torch.nan_to_num(node_feats['_N/labels']).long()
        # inner_node_mask = graph.ndata['inner_node'].bool()
        local_node_lid = torch.nonzero(graph.ndata['inner_node']).squeeze()
        print(f"[Note]Rank#{rank}: local_node_id: {local_node_lid}")
        global_nid = graph.ndata[dgl.NID]

        graph = dgl.add_self_loop(graph)

        num_edges = graph.num_edges()

        u, v = graph.edges()
        all_u.append(global_nid[u])
        all_v.append(global_nid[v])

        global_labels[global_nid[local_node_lid]] = labels
        global_node_feats[global_nid[local_node_lid]] = local_node_feats
        global_train_mask[global_nid[local_node_lid]] = local_train_mask
        # global_val_mask[global_nid[local_node_lid]] = local_val_mask
        # global_test_mask[global_nid[local_node_lid]] = local_test_mask

    # build the whole graph
    all_u = torch.hstack(all_u)
    all_v = torch.hstack(all_v)
    shared_graph = dgl.graph((all_u, all_v))
    shared_graph = dgl.add_self_loop(shared_graph)
    shared_graph = dgl.to_bidirected(shared_graph)
    ndata_keys = shared_graph.ndata.keys()
    print(f"[Note]Before ndata:{ndata_keys}")
    print(f"[Note]Share graph: #nodes:{shared_graph.num_nodes()}\t #edges:{shared_graph.num_edges()}")
    print(
        f"[Note]node data: node_feats:{global_node_feats.shape}\t labels:{global_labels.shape}\t train_mask:{global_train_mask.shape}")
    shared_graph.ndata['_N/feat'] = global_node_feats
    shared_graph.ndata['_N/labels'] = global_labels
    shared_graph.ndata['_N/train_mask'] = global_train_mask
    # shared_graph.ndata['_N/val_mask'] = global_val_mask
    # shared_graph.ndata['_N/test_mask'] = global_test_mask
    ndata_keys = shared_graph.ndata.keys()
    print(f"[Note]After ndata:{ndata_keys}")
    save_path = f"./npc_dataset/{args.dataset}M{world_size}.bin"
    print(f"[Note]Save graph{shared_graph} to dir: {save_path}")
    dgl.data.utils.save_graphs(save_path, [shared_graph])


if __name__ == "__main__":
    args = utils.init_args()
    print(args)
    data_preprocess(args)
