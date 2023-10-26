import torch
import dgl
from typing import Tuple, List
from dgl.heterograph import DGLBlock


def local_sample_one_layer(seeds: torch.Tensor, fanout: int, to_virtual: int = 0):
    return torch.ops.npc.local_sample_one_layer(seeds, fanout, to_virtual)


def np_sample_and_shuffle(seeds: torch.Tensor, fanout: int):
    return torch.ops.npc.np_sample_and_shuffle(seeds, fanout)


def srcdst_to_vir(fanout: int, dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.ops.npc.srcdst_to_vir(fanout, dst, src)


def sp_sample_and_shuffle(
    num_dst: int,
    send_frontier: torch.Tensor,
    sorted_allnodes: torch.Tensor,
    unique_frontier: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.npc.sp_sample_and_shuffle(num_dst, send_frontier, sorted_allnodes, unique_frontier)


def mp_sample_shuffle(seeds: torch.Tensor, unique_frontier: torch.Tensor, coo_row: torch.Tensor) -> List[torch.Tensor]:
    return torch.ops.npc.mp_sample_shuffle(seeds, unique_frontier, coo_row)


def create_block_from_csc(indptr, indices, e_ids, num_src, num_dst) -> DGLBlock:
    hgidx = dgl.heterograph_index.create_unitgraph_from_csr(
        2,
        num_src,
        num_dst,
        indptr,
        indices,
        e_ids,
        formats=["coo", "csr", "csc"],
        transpose=True,
    )
    retg = DGLBlock(hgidx, (["_N"], ["_N"]), ["_E"])
    return retg


def create_block_from_coo(row, col, num_src, num_dst) -> DGLBlock:
    hgidx = dgl.heterograph_index.create_unitgraph_from_coo(
        2,
        num_src,
        num_dst,
        row,
        col,
        formats=["coo", "csr", "csc"],
    )
    retg = DGLBlock(hgidx, (["_N"], ["_N"]), ["_E"])
    return retg


def tensor_relabel_csc(seeds, neighbors) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.npc.relabel_csc(seeds, neighbors)


def create_dgl_block(seeds, neighbors, fanout):
    unique_frontier, indices = tensor_relabel_csc(seeds, neighbors)
    coo_col = torch.arange(0, seeds.numel(), device=indices.device).repeat_interleave(fanout)

    block = create_block_from_coo(
        indices,
        coo_col,
        num_src=unique_frontier.numel(),
        num_dst=seeds.numel(),
    )
    block.srcdata["_ID"] = unique_frontier

    return block


# Loader
class MyEvent:
    def __init__(self):
        self.event = torch.cuda.Event(enable_timing=True)

    def to(self, device, non_blocking):
        return self.event

    def record(self):
        self.event.record()

    def elapsed_time(self, end_event):
        return self.event.elapsed_time(end_event)


class MixedNeighborSampler(object):
    def __init__(
        self,
        rank,
        fanouts,
        debug_info=None,
    ):
        self.rank = rank
        self.fir_fanouts = fanouts[1:]
        self.las_fanouts = fanouts[0]
        self.num_layers = len(fanouts)
        self.debug_flag = False
        if debug_info is not None:
            self.debug_graph, self.debug_min_vids, self.num_nodes = debug_info
            self.debug_flag = True
            print(f"[Note]debug:{self.debug_flag}\t graph:{self.debug_graph}\t min_vids:{self.debug_min_vids}\t #nodes:{self.num_nodes}")

    def debug_check(self, src, dst):
        cpu_src = src.detach().cpu()
        cpu_dst = dst.detach().cpu()
        debug_check_flag = torch.all(self.debug_graph.has_edges_between(cpu_src, cpu_dst))
        # print(f"[Note]Sampling check:{debug_check_flag}")
        assert debug_check_flag, "[Error]Sampling debug_check failed"

    def sample(self, graph, seeds):
        output_nodes = seeds
        blocks = []
        event = MyEvent()
        for fanout in reversed(self.fir_fanouts):
            seeds, neighbors = local_sample_one_layer(seeds, fanout)
            replicated_seeds = torch.repeat_interleave(seeds, fanout)
            if self.debug_flag:
                self.debug_check(neighbors, replicated_seeds)
            block_g = dgl.graph((neighbors, replicated_seeds))
            block = dgl.to_block(g=block_g, dst_nodes=seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        # last layer
        # Shape seeds = sum(send_offset)
        # Shape negibors = sum(send_offset) * self.las_fanouts
        event.record()
        seeds, neighbors, perm, send_offset, recv_offset = np_sample_and_shuffle(seeds, self.las_fanouts)
        replicated_seeds = torch.repeat_interleave(seeds, self.las_fanouts)
        if self.debug_flag:
            self.debug_check(neighbors, replicated_seeds)

        block_g = dgl.graph((neighbors, replicated_seeds))
        block = dgl.to_block(g=block_g, dst_nodes=seeds)
        blocks.insert(0, block)
        seeds = block.srcdata[dgl.NID]

        return seeds, output_nodes, blocks, perm, send_offset, recv_offset, event


class MixedPSNeighborSampler(object):
    def __init__(
        self,
        rank,
        world_size,
        fanouts,
        system,
        num_total_nodes,
        debug_info=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.fanouts = fanouts
        self.num_layers = len(fanouts)
        assert system in ["DP", "NP", "MP", "SP"]
        self.system = system
        self.debug_flag = False
        self.num_total_nodes = num_total_nodes
        self.sp_val = (rank << 20) * num_total_nodes
        print(f"[Note]debug_info:{debug_info}")
        if debug_info is not None:
            self.debug_graph, self.debug_min_vids, self.num_nodes = debug_info
            self.debug_flag = True
            print(f"[Note]debug:{self.debug_flag}\t graph:{self.debug_graph}\t min_vids:{self.debug_min_vids}\t #nodes:{self.num_nodes}")

    def sample(self, graph, seeds):
        torch.cuda.nvtx.range_push("sample")
        output_nodes = seeds
        blocks = []

        for layer_id, fanout in enumerate(reversed(self.fanouts)):
            seeds, neighbors = local_sample_one_layer(seeds, fanout)

            if self.debug_flag:
                replicated_seeds = torch.repeat_interleave(seeds, fanout)
                copy_seeds = replicated_seeds.detach().cpu()
                copy_neighbors = neighbors.detach().cpu()
                flag = torch.all(self.debug_graph.has_edges_between(copy_neighbors, copy_seeds))
                assert flag, f"[Note]Sys{self.system}\t layer_id:{layer_id}\t flag:{flag}"

            if layer_id == self.num_layers - 1:
                if self.system == "DP":
                    sampling_result = ()
                elif self.system == "NP":
                    (
                        shuffled_seeds,
                        neighbors,
                        perm,
                        send_offset,
                        recv_offset,
                    ) = np_sample_and_shuffle(seeds, fanout)
                    sampling_result = (perm, send_offset, recv_offset)

                elif self.system == "SP":
                    map_allnodes = srcdst_to_vir(fanout, seeds, neighbors)
                    sorted_allnodes, perm_allnodes = torch.sort(map_allnodes)
                    num_dst = seeds.numel()
                    map_src = sorted_allnodes[num_dst:]

                    unique_frontier, arange_src = torch.unique(map_src, return_inverse=True)
                    # build block1 by dgl.create_block
                    num_dst = seeds.numel()
                    device = seeds.device
                    arange_dst = torch.arange(num_dst, device=device).repeat_interleave(fanout)
                    block1 = dgl.create_block((arange_src, arange_dst))
                    blocks.insert(0, block1)
                    # send_frontier = [dst, (pack virtual nodes and original)]
                    send_frontier = torch.cat(
                        (
                            seeds[perm_allnodes[:num_dst]],
                            self.sp_val + (map_src % num_dst) * self.num_total_nodes + neighbors[perm_allnodes[num_dst:] - num_dst],
                        )
                    )

                    (
                        recv_dst,
                        recv_seeds,
                        recv_neighbors,
                        send_sizes,
                        recv_sizes,
                    ) = sp_sample_and_shuffle(
                        num_dst,  # num_dst
                        send_frontier,  # send_frontier
                        sorted_allnodes,  # sorted_allnodes
                        unique_frontier,  # unique_frontier
                    )

                    # build block2 by dgl.to_block
                    block2_graph = dgl.graph((recv_neighbors, recv_seeds))
                    block2 = dgl.to_block(block2_graph, include_dst_in_src=False)
                    seeds = torch.cat((recv_dst, block2.srcdata[dgl.NID]))
                    blocks.insert(0, block2)
                    sampling_result = (send_sizes, recv_sizes)

                elif self.system == "MP":
                    unique_frontier, coo_row = tensor_relabel_csc(seeds, neighbors)
                    (
                        all_frontier,
                        all_coo_row,
                        send_size,
                        recv_size,
                        recv_frontier_size,
                        recv_coo_size,
                    ) = mp_sample_shuffle(seeds, unique_frontier, coo_row)
                    all_coo_col = torch.cat([torch.arange(0, i, device=all_coo_row.device).repeat_interleave(fanout) for i in recv_size])
                    blocks.insert(0, (all_coo_row, all_coo_col, recv_frontier_size, recv_coo_size, recv_size))
                    sampling_result = (send_size, recv_size)
                    seeds = all_frontier

            if layer_id != self.num_layers - 1 or self.system not in ("SP", "MP"):
                torch.cuda.nvtx.range_push("construct block")
                block = create_dgl_block(seeds, neighbors, fanout)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)
                torch.cuda.nvtx.range_pop()

        input_nodes = seeds
        torch.cuda.nvtx.range_pop()
        return (input_nodes, output_nodes, blocks) + sampling_result


class DGLNeighborSampler(dgl.dataloading.NeighborSampler):
    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__(
            fanouts,
            edge_dir,
            prob,
            mask,
            replace,
            prefetch_node_feats,
            prefetch_labels,
            prefetch_edge_feats,
            output_device,
        )

    def sample(self, g, seed_nodes, exclude_eids=None):  # pylint: disable=arguments-differ
        """Sample a list of blocks from the given seed nodes."""
        result = self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
        return result
