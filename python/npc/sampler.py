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


def src_to_vir(fanout: int, num_dst: int, src: torch.Tensor) -> torch.Tensor:
    return torch.ops.npc.src_to_vir(fanout, num_dst, src)


def sp_sample_and_shuffle(
    num_dst: int,
    send_frontier: torch.Tensor,
    sorted_allnodes: torch.Tensor,
    unique_frontier: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.npc.sp_sample_and_shuffle(num_dst, send_frontier, sorted_allnodes, unique_frontier)


def sp_sample_shuffle_src(
    unique_src: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.npc.sp_sample_shuffle_src(unique_src)


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


def create_dgl_block(seeds, neighbors, fanout, return_coo=False):
    unique_frontier, indices = tensor_relabel_csc(seeds, neighbors)
    coo_col = torch.arange(0, seeds.numel(), device=indices.device).repeat_interleave(fanout)

    block = create_block_from_coo(
        indices,
        coo_col,
        num_src=unique_frontier.numel(),
        num_dst=seeds.numel(),
    )
    block.srcdata["_ID"] = unique_frontier

    if return_coo:
        return block, (indices, coo_col)
    else:
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
        assert debug_check_flag, "[Error]Sampling debug_check failed"

    def sample(self, graph, seeds):
        output_nodes = seeds
        blocks = []
        # event = MyEvent()
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
        # event.record()

        seeds, neighbors, perm, send_offset, recv_offset = np_sample_and_shuffle(seeds, self.las_fanouts)
        block = create_dgl_block(seeds, neighbors, self.las_fanouts)
        blocks.insert(0, block)
        seeds = block.srcdata[dgl.NID]

        return seeds, output_nodes, blocks, perm, send_offset, recv_offset


class MixedPSNeighborSampler(object):
    def __init__(
        self,
        rank,
        world_size,
        fanouts,
        system,
        model,
        num_total_nodes,
        debug_info=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.fanouts = fanouts
        self.num_layers = len(fanouts)
        assert system in ["DP", "NP", "MP", "SP"]
        self.system = system
        self.model = model
        self.debug_flag = False
        self.num_total_nodes = num_total_nodes
        self.sp_base = 10000000
        if debug_info is not None:
            self.debug_graph, self.debug_min_vids, self.num_nodes = debug_info
            self.debug_flag = True
            print(f"[Note]debug:{self.debug_flag}\t graph:{self.debug_graph}\t min_vids:{self.debug_min_vids}\t #nodes:{self.num_nodes}")

    def sample(self, graph, seeds):
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
                    if self.model == "GAT":
                        num_dst = seeds.numel()
                        unique_neigh, arange_src = torch.unique(neighbors, return_inverse=True)
                        arange_dst = torch.arange(num_dst, device=seeds.device).repeat_interleave(fanout)
                        block = create_block_from_coo(arange_src, arange_dst, unique_neigh.numel(), num_dst)

                        (
                            shuffled_neigh,
                            perm,
                            send_offset,
                            recv_offset,
                        ) = sp_sample_shuffle_src(unique_neigh)

                        blocks.insert(0, block)
                        sampling_result = (perm, send_offset, recv_offset)

                        # seeds contains original dst nodes and recv src nodes
                        seeds = torch.cat((seeds, shuffled_neigh))

                    else:
                        num_dst = seeds.numel()
                        device = seeds.device
                        # [device_id, dst_id]
                        # map_src = device_id * (num_dst) + dst_id
                        map_src = src_to_vir(fanout, num_dst, neighbors)

                        sorted_mapsrc, perm_mapsrc = torch.sort(map_src)

                        unique_frontier, arange_src = torch.unique(map_src, return_inverse=True)

                        arange_dst = unique_frontier % num_dst  # [0, num_dst)
                        arange_src = torch.arange(0, unique_frontier.numel())  # [0, #unique_frontier)

                        # build block1 by dgl.create_block
                        arange_dst = torch.arange(num_dst, device=device).repeat_interleave(fanout)
                        block1 = create_block_from_coo(arange_src, arange_dst, unique_frontier.numel(), num_dst)
                        # block1
                        # src [0, num_src-1]
                        # dst [0, num_dst-1]

                        blocks.insert(0, block1)

                        # send_frontier = (pack virtual nodes and original)
                        # [from_rank, dst_id, ori_src]
                        # rules of send_frontier: from_rank * (#total_nodes*sp_val) (#total_nodes) * dst_id + ori_src
                        # perm_dst, range: [0, num_dst)

                        perm_dst = sorted_mapsrc % num_dst
                        send_frontier = self.rank * (self.sp_base * self.num_total_nodes) + perm_dst * self.num_total_nodes + neighbors[perm_mapsrc]

                        # perm_dst = seeds.repeat_interleave(fanout)[perm_mapsrc]
                        # send_frontier = (
                        #    self.rank * (self.num_total_nodes * self.num_total_nodes) + perm_dst * self.num_total_nodes + neighbors[perm_mapsrc])
                        
                        (
                            recv_seeds,
                            recv_neighbors,
                            send_sizes,
                            recv_sizes,
                        ) = sp_sample_and_shuffle(
                            num_dst,  # num_dst
                            send_frontier,  # send_frontier
                            sorted_mapsrc,  # sorted_mapsrc
                            unique_frontier,  # unique_frontier
                        )
                        # [from_rank, dst_id, ori_src]
                        # recv_seeds: [from_rank, dst_id]
                        # recv_neighbors: [ori_src](neighbors)
                        
                        h_vir = block2.dstdata["h"]
                        shuffle(h_vir)
                        num_dst2 = unique_dst.numel()
                        [0, 1, 2, 3 , 4 ]
                        
                        
                        [0000000, 1111111, 2222222, 333333, ...., world_size-1 ]
                        

                        
                        unique_src, arange_src = torch.unique(recv_neighbors, return_inverse=True)
                        unique_dst, arange_dst = torch.unique(recv_seeds, return_inverse=True)

                        block2 = create_block_from_coo(arange_src, arange_dst, unique_src.numel(), unique_dst.numel())

                        blocks.insert(0, block2)
                        sampling_result = (send_sizes, recv_sizes)

                        # seeds contains original dst nodes and recv src nodes
                        seeds = torch.cat((seeds, unique_src))

                elif self.system == "MP":
                    if self.model == "GAT":
                        block, (coo_row, coo_col) = create_dgl_block(seeds, neighbors, fanout, True)
                        unique_frontier = block.srcdata["_ID"]
                        send_frontier_size = torch.tensor([unique_frontier.numel()])
                    else:
                        unique_frontier, coo_row = tensor_relabel_csc(seeds, neighbors)

                    (
                        all_frontier,
                        all_coo_row,
                        send_size,
                        recv_size,
                        recv_frontier_size,
                        recv_coo_size,
                    ) = mp_sample_shuffle(seeds, unique_frontier, coo_row)

                    if self.model == "GAT":
                        blocks.insert(0, block)
                        sampling_result = (send_frontier_size, recv_frontier_size)
                    else:
                        all_coo_col = torch.cat([torch.arange(0, i, device=all_coo_row.device).repeat_interleave(fanout) for i in recv_size])
                        blocks.insert(0, (all_coo_row, all_coo_col, recv_frontier_size, recv_coo_size, recv_size))
                        sampling_result = (send_size, recv_size)
                    seeds = all_frontier

            if layer_id != self.num_layers - 1 or self.system not in ("SP", "MP"):
                block = create_dgl_block(seeds, neighbors, fanout)
                seeds = block.srcdata[dgl.NID]
                blocks.insert(0, block)

        input_nodes = seeds
        return (input_nodes, output_nodes, blocks) + sampling_result


class RefSampler(object):
    def __init__(
        self,
        rank,
        world_size,
        fanouts,
        system,
        model,
        num_total_nodes,
        debug_info=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.fanouts = fanouts
        self.num_layers = len(fanouts)
        assert system in ["DP", "NP", "MP", "SP"]
        self.system = system
        self.model = model
        self.debug_flag = False
        self.num_total_nodes = num_total_nodes
        self.sp_base = 10000000
        if debug_info is not None:
            self.debug_graph, self.debug_min_vids, self.num_nodes = debug_info
            self.debug_flag = True
            print(f"[Note]debug:{self.debug_flag}\t graph:{self.debug_graph}\t min_vids:{self.debug_min_vids}\t #nodes:{self.num_nodes}")

    def sample(self, graph, seeds):
        output_nodes = seeds
        blocks = []
        sp_blocks = []

        for layer_id, fanout in enumerate(reversed(self.fanouts)):
            seeds, neighbors = local_sample_one_layer(seeds, fanout)

            if self.debug_flag:
                replicated_seeds = torch.repeat_interleave(seeds, fanout)
                copy_seeds = replicated_seeds.detach().cpu()
                copy_neighbors = neighbors.detach().cpu()
                flag = torch.all(self.debug_graph.has_edges_between(copy_neighbors, copy_seeds))
                assert flag, f"[Note]Sys{self.system}\t layer_id:{layer_id}\t flag:{flag}"

            if layer_id == self.num_layers - 1:
                """
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
                """
                # elif self.system == "SP":
                """
                    if self.model == "GAT":
                        num_dst = seeds.numel()
                        unique_neigh, arange_src = torch.unique(neighbors, return_inverse=True)
                        arange_dst = torch.arange(num_dst, device=seeds.device).repeat_interleave(fanout)
                        block = create_block_from_coo(arange_src, arange_dst, unique_neigh.numel(), num_dst)

                        (
                            shuffled_neigh,
                            perm,
                            send_offset,
                            recv_offset,
                        ) = sp_sample_shuffle_src(unique_neigh)

                        blocks.insert(0, block)
                        sampling_result = (perm, send_offset, recv_offset)

                        # seeds contains original dst nodes and recv src nodes
                        seeds = torch.cat((seeds, shuffled_neigh))
                """
                if True:
                    num_dst = seeds.numel()
                    device = seeds.device
                    # [device_id, dst_id]
                    # map_src = device_id * num_dst + dst_id
                    map_src = src_to_vir(fanout, num_dst, neighbors)
                    """
                    if self.debug_flag:
                        arange_dst = torch.arange(num_dst, device=device).repeat_interleave(fanout)
                        ref_map_src = self.debug_min_vids[neighbors] * num_dst + arange_dst
                        flag_map_src = torch.equal(ref_map_src, map_src)
                        print(f"[Note]flag_map_src:{flag_map_src}")
                    """
                    sorted_mapsrc, perm_mapsrc = torch.sort(map_src)

                    unique_frontier, arange_src = torch.unique(map_src, return_inverse=True)
                    # build block1 by dgl.create_block
                    arange_dst = torch.arange(num_dst, device=device).repeat_interleave(fanout)
                    block1 = create_block_from_coo(arange_src, arange_dst, unique_frontier.numel(), num_dst)
                    sp_blocks.insert(0, block1)

                    # send_frontier = (pack virtual nodes and original)
                    # [from_rank, dst_id, ori_src]
                    # rules of send_frontier: from_rank * (#total_nodes*sp_val) (#total_nodes) * dst_id + ori_src
                    # perm_dst, range: [0, num_dst)

                    perm_dst = sorted_mapsrc % num_dst
                    send_frontier = self.rank * (self.sp_base * self.num_total_nodes) + perm_dst * self.num_total_nodes + neighbors[perm_mapsrc]

                    # perm_dst = seeds.repeat_interleave(fanout)[perm_mapsrc]
                    # send_frontier = (
                    #    self.rank * (self.num_total_nodes * self.num_total_nodes) + perm_dst * self.num_total_nodes + neighbors[perm_mapsrc])

                    (
                        recv_seeds,
                        recv_neighbors,
                        send_sizes,
                        recv_sizes,
                    ) = sp_sample_and_shuffle(
                        num_dst,  # num_dst
                        send_frontier,  # send_frontier
                        sorted_mapsrc,  # sorted_mapsrc
                        unique_frontier,  # unique_frontier
                    )

                    # build block2 by dgl.to_block
                    unique_src, arange_src = torch.unique(recv_neighbors, return_inverse=True)
                    unique_dst, arange_dst = torch.unique(recv_seeds, return_inverse=True)
                    block2 = create_block_from_coo(arange_src, arange_dst, unique_src.numel(), unique_dst.numel())

                    sp_blocks.insert(0, block2)
                    sampling_result = (send_sizes, recv_sizes)

                    # seeds contains original dst nodes and recv src nodes
                    sp_seeds = torch.cat((seeds, unique_src))
                """
                elif self.system == "MP":
                    if self.model == "GAT":
                        block, (coo_row, coo_col) = create_dgl_block(seeds, neighbors, fanout, True)
                        unique_frontier = block.srcdata["_ID"]
                        send_frontier_size = torch.tensor([unique_frontier.numel()])
                    else:
                        unique_frontier, coo_row = tensor_relabel_csc(seeds, neighbors)

                    (
                        all_frontier,
                        all_coo_row,
                        send_size,
                        recv_size,
                        recv_frontier_size,
                        recv_coo_size,
                    ) = mp_sample_shuffle(seeds, unique_frontier, coo_row)

                    if self.model == "GAT":
                        blocks.insert(0, block)
                        sampling_result = (send_frontier_size, recv_frontier_size)
                    else:
                        all_coo_col = torch.cat([torch.arange(0, i, device=all_coo_row.device).repeat_interleave(fanout) for i in recv_size])
                        blocks.insert(0, (all_coo_row, all_coo_col, recv_frontier_size, recv_coo_size, recv_size))
                        sampling_result = (send_size, recv_size)
                    seeds = all_frontier
                """
                # if layer_id != self.num_layers - 1 or self.system not in ("SP", "MP"):
                # if layer_id != self.num_layers - 1:
            block = create_dgl_block(seeds, neighbors, fanout)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
            if layer_id != self.num_layers - 1:
                sp_blocks.insert(0, block)

        # input_nodes = seeds
        # DP(0,1,2) + SP(3,4,5,6,7)
        return (seeds, output_nodes, blocks) + (sp_seeds, output_nodes, sp_blocks) + sampling_result
