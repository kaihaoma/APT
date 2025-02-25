import torch
import dgl
from typing import Tuple, List
from dgl.heterograph import DGLBlock
from .utils import get_time


def local_sample_one_layer(seeds: torch.Tensor, fanout: int):
    return torch.ops.npc.local_sample_one_layer(seeds, fanout)


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
    shuffle_with_dst: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.npc.sp_sample_and_shuffle(num_dst, send_frontier, sorted_allnodes, unique_frontier, shuffle_with_dst)


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
            if self.debug_flag:
                self.debug_check(neighbors, torch.repeat_interleave(seeds, fanout))
            block = create_dgl_block(seeds, neighbors, fanout)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        # last layer
        # Shape seeds = sum(send_offset)
        # Shape negibors = sum(send_offset) * self.las_fanouts
        # event.record()
        seeds, neighbors, perm, send_offset, recv_offset, inverse_idx = np_sample_and_shuffle(seeds, self.las_fanouts)

        if self.debug_flag:
            self.debug_check(neighbors, torch.repeat_interleave(seeds, self.las_fanouts))

        block = create_dgl_block(seeds, neighbors, self.las_fanouts)
        blocks.insert(0, block)
        seeds = block.srcdata[dgl.NID]

        return seeds, output_nodes, blocks, perm, send_offset, recv_offset, inverse_idx


class MixedPSNeighborSampler(object):
    def __init__(
        self,
        rank,
        world_size,
        fanouts,
        system,
        model,
        num_total_nodes,
        shuffle_with_dst=False,
        debug_info=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.fanouts = fanouts
        self.num_layers = len(fanouts)
        assert system in ["DP", "NP", "MP", "SP"]
        self.system = system
        self.model = model
        self.shuffle_with_dst = shuffle_with_dst
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
                        if not self.shuffle_with_dst:
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
                            shuffled_seeds_and_neigh, perm, send_offset, recv_offset = sp_sample_shuffle_src(torch.cat((seeds, unique_neigh)))
                            blocks.insert(0, block)
                            sampling_result = (perm, send_offset, recv_offset)
                            seeds = shuffled_seeds_and_neigh

                    else:
                        device = seeds.device
                        num_dst = seeds.numel()

                        if self.shuffle_with_dst:
                            # map src&dst to vir
                            # rules: [is_src, belong, dst_idx]
                            # map = is_src * base1 + belong * base2 + dst_idx
                            # base2 = num_seeds
                            # base1 = num_seeds * world_size
                            map_allnodes = srcdst_to_vir(fanout, seeds, neighbors)
                            sorted_allnodes, perm_allnodes = torch.sort(map_allnodes)
                            map_src = sorted_allnodes[num_dst:]
                        else:
                            # map src to vir
                            # rules: [belong, dst_idx]
                            # map = belong * num_dst + dst_idx
                            map_src = src_to_vir(fanout, num_dst, neighbors)
                            sorted_mapsrc, perm_mapsrc = torch.sort(map_src)

                        unique_frontier, arange_src = torch.unique(map_src, return_inverse=True)
                        arange_dst = unique_frontier % num_dst  # [0, num_dst)
                        arange_src = torch.arange(0, unique_frontier.numel(), device=device)  # [0, #unique_frontier)
                        block1 = create_block_from_coo(arange_src, arange_dst, unique_frontier.numel(), num_dst)
                        blocks.insert(0, block1)

                        if self.shuffle_with_dst:
                            # send_frontier = (perm_dst, (pack virtual node and original))
                            # perm_dst = seeds[perm_allnodes[:num_dst]]
                            # pack virtual and original nodes = [from_rank, dst_id, ori_src]
                            # packed = from_rank * (sp_base * num_total_nodes) + dst_id * (num_total_nodes) + ori_src

                            send_frontiers = torch.cat(
                                (
                                    seeds[perm_allnodes[:num_dst]],
                                    self.rank * (self.sp_base * self.num_total_nodes)
                                    + (map_src % num_dst) * self.num_total_nodes
                                    + neighbors[perm_allnodes[num_dst:] - num_dst],
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
                                send_frontiers,  # send_frontiers
                                sorted_allnodes,  # sorted_allnodes
                                unique_frontier,  # unique_frontier
                                self.shuffle_with_dst,
                            )
                        else:
                            # send_frontier = (pack virtual nodes(with global id) and original)
                            # [from_rank, dst_id, ori_src]
                            # rules of send_frontier: from_rank * (sp_base * num_total_nodes) + perm_st * num_total_nodes + neighbors[perm_mapsrc]
                            perm_dst = sorted_mapsrc % num_dst
                            send_frontier = (
                                self.rank * (self.sp_base * self.num_total_nodes) + perm_dst * self.num_total_nodes + neighbors[perm_mapsrc]
                            )

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
                                self.shuffle_with_dst,  # shuffle_with_dst
                            )

                        # build block2 by dgl.to_block
                        unique_src, arange_src = torch.unique(recv_neighbors, return_inverse=True)
                        unique_dst, arange_dst = torch.unique(recv_seeds, return_inverse=True)
                        block2 = create_block_from_coo(arange_src, arange_dst, unique_src.numel(), unique_dst.numel())

                        blocks.insert(0, block2)
                        sampling_result = (send_sizes, recv_sizes)

                        # seeds contains original dst nodes and recv src nodes
                        if self.shuffle_with_dst:
                            seeds = torch.cat((recv_dst, unique_src))
                        else:
                            if self.model == "GCN":
                                seeds = unique_src
                            else:
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


class AllPSNeighborSampler(object):
    def __init__(
        self,
        rank,
        world_size,
        fanouts,
        model,
        num_total_nodes,
        shuffle_with_dst=False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.fanouts = fanouts
        self.num_layers = len(fanouts)
        self.model = model
        self.shuffle_with_dst = shuffle_with_dst
        self.num_total_nodes = num_total_nodes
        self.sp_base = 10000000
        self.num_sample_times = 0
        self.time_list = [0, 0, 0, 0]
        self.sampling_algo_time = [0 for _ in range(self.num_layers)]

    def get_time_list(self):
        print(f"[Note] #sample times: {self.num_sample_times}")
        return (
            self.num_sample_times,
            self.sampling_algo_time,
            self.time_list,
        )

    def clear_time_list(self):
        self.num_sample_times = 0
        self.time_list = [0, 0, 0, 0]
        self.sampling_algo_time = [0 for _ in range(self.num_layers)]

    def sample(self, graph, seeds):
        # output_nodes = seeds
        # blocks = []
        self.num_sample_times += 1

        for layer_id, fanout in enumerate(reversed(self.fanouts)):
            # NP shuffle in last layer
            if layer_id == self.num_layers - 1:
                np_shuffle_begin = get_time()
                np_seeds, np_neighbors, np_perm, np_send_offset, np_recv_offset, np_inverse_idx = np_sample_and_shuffle(seeds, fanout)
                np_shuffle_end = get_time()
                # print(f"[Note]np_shuffle_time:{np_shuffle_end - np_shuffle_begin:4f}")
                self.time_list[1] += np_shuffle_end - np_shuffle_begin

            sampling_algo_begin = get_time()
            seeds, neighbors = local_sample_one_layer(seeds, fanout)
            sampling_algo_end = get_time()
            self.sampling_algo_time[layer_id] += sampling_algo_end - sampling_algo_begin

            if layer_id != self.num_layers - 1:
                sampling_algo_begin = get_time()
                block = create_dgl_block(seeds, neighbors, fanout)
                seeds = block.srcdata[dgl.NID]
                sampling_algo_end = get_time()
                self.sampling_algo_time[layer_id] += sampling_algo_end - sampling_algo_begin
                # blocks.insert(0, block)
            else:
                # DP
                dp_build_block_begin = get_time()
                dp_block = create_dgl_block(seeds, neighbors, fanout)
                dp_seeds = dp_block.srcdata[dgl.NID]
                dp_build_block_end = get_time()
                self.time_list[0] += dp_build_block_end - dp_build_block_begin
                # NP
                np_build_block_begin = get_time()
                np_block = create_dgl_block(np_seeds, np_neighbors, fanout)
                np_seeds = np_block.srcdata[dgl.NID]
                np_build_block_end = get_time()
                # print(f"[Note]np_build_block_time:{np_build_block_end - np_build_block_begin:4f}")
                self.time_list[1] += np_build_block_end - np_build_block_begin
                # SP
                sp_all_time_begin = get_time()
                if self.model == "GAT":
                    num_dst = seeds.numel()
                    sp_unique_neigh, sp_arange_src = torch.unique(neighbors, return_inverse=True)
                    sp_arange_dst = torch.arange(num_dst, device=seeds.device).repeat_interleave(fanout)
                    # sp_block = create_block_from_coo(sp_arange_src, sp_arange_dst, sp_unique_neigh.numel(), num_dst)
                    if not self.shuffle_with_dst:
                        (
                            sp_shuffled_neigh,
                            sp_perm,
                            sp_send_offset,
                            sp_recv_offset,
                        ) = sp_sample_shuffle_src(sp_unique_neigh)

                        sp_sampling_result = (sp_send_offset, sp_recv_offset)

                        # seeds contains original dst nodes and recv src nodes
                        sp_seeds = torch.cat((seeds, sp_shuffled_neigh))
                    else:
                        sp_shuffled_seeds_and_neigh, sp_perm, sp_send_offset, sp_recv_offset = sp_sample_shuffle_src(
                            torch.cat((seeds, sp_unique_neigh))
                        )
                        sp_sampling_result = (sp_send_offset, sp_recv_offset)
                        sp_seeds = sp_shuffled_seeds_and_neigh
                else:
                    device = seeds.device
                    num_dst = seeds.numel()

                    if self.shuffle_with_dst:
                        # map src&dst to vir
                        # rules: [is_src, belong, dst_idx]
                        # map = is_src * base1 + belong * base2 + dst_idx
                        # base2 = num_seeds
                        # base1 = num_seeds * world_size
                        sp_map_allnodes = srcdst_to_vir(fanout, seeds, neighbors)
                        sp_sorted_allnodes, sp_perm_allnodes = torch.sort(sp_map_allnodes)
                        sp_map_src = sp_sorted_allnodes[num_dst:]
                    else:
                        # map src to vir
                        # rules: [belong, dst_idx]
                        # map = belong * num_dst + dst_idx
                        sp_map_src = src_to_vir(fanout, num_dst, neighbors)
                        sp_sorted_mapsrc, sp_perm_mapsrc = torch.sort(sp_map_src)

                    sp_unique_frontier, sp_arange_src = torch.unique(sp_map_src, return_inverse=True)
                    sp_arange_dst = sp_unique_frontier % num_dst  # [0, num_dst)
                    sp_arange_src = torch.arange(0, sp_unique_frontier.numel(), device=device)  # [0, #unique_frontier)
                    # sp_block1 = create_block_from_coo(sp_arange_src, sp_arange_dst, sp_unique_frontier.numel(), num_dst)
                    # blocks.insert(0, block1)

                    if self.shuffle_with_dst:
                        # send_frontier = (perm_dst, (pack virtual node and original))
                        # perm_dst = seeds[perm_allnodes[:num_dst]]
                        # pack virtual and original nodes = [from_rank, dst_id, ori_src]
                        # packed = from_rank * (sp_base * num_total_nodes) + dst_id * (num_total_nodes) + ori_src

                        sp_send_frontiers = torch.cat(
                            (
                                seeds[sp_perm_allnodes[:num_dst]],
                                self.rank * (self.sp_base * self.num_total_nodes)
                                + (sp_map_src % num_dst) * self.num_total_nodes
                                + neighbors[sp_perm_allnodes[num_dst:] - num_dst],
                            )
                        )
                        (
                            sp_recv_dst,
                            sp_recv_seeds,
                            sp_recv_neighbors,
                            sp_send_sizes,
                            sp_recv_sizes,
                        ) = sp_sample_and_shuffle(
                            num_dst,  # num_dst
                            sp_send_frontiers,  # send_frontiers
                            sp_sorted_allnodes,  # sorted_allnodes
                            sp_unique_frontier,  # unique_frontier
                            self.shuffle_with_dst,
                        )
                    else:
                        # send_frontier = (pack virtual nodes(with global id) and original)
                        # [from_rank, dst_id, ori_src]
                        # rules of send_frontier: from_rank * (sp_base * num_total_nodes) + perm_st * num_total_nodes + neighbors[perm_mapsrc]
                        sp_perm_dst = sp_sorted_mapsrc % num_dst
                        sp_send_frontier = (
                            self.rank * (self.sp_base * self.num_total_nodes) + sp_perm_dst * self.num_total_nodes + neighbors[sp_perm_mapsrc]
                        )

                        (
                            sp_recv_seeds,
                            sp_recv_neighbors,
                            sp_send_sizes,
                            sp_recv_sizes,
                        ) = sp_sample_and_shuffle(
                            num_dst,  # num_dst
                            sp_send_frontier,  # send_frontier
                            sp_sorted_mapsrc,  # sorted_mapsrc
                            sp_unique_frontier,  # unique_frontier
                            self.shuffle_with_dst,  # shuffle_with_dst
                        )

                    # build block2 by dgl.to_block
                    sp_unique_src, sp_arange_src = torch.unique(sp_recv_neighbors, return_inverse=True)
                    sp_unique_dst, sp_arange_dst = torch.unique(sp_recv_seeds, return_inverse=True)
                    # block2 = create_block_from_coo(sp_arange_src, sp_arange_dst, sp_unique_src.numel(), sp_unique_dst.numel())

                    # blocks.insert(0, block2)
                    sp_sampling_result = (sp_send_sizes, sp_recv_sizes)

                    # seeds contains original dst nodes and recv src nodes
                    if self.shuffle_with_dst:
                        sp_seeds = torch.cat((sp_recv_dst, sp_unique_src))
                    else:
                        if self.model == "GCN":
                            sp_seeds = sp_unique_src
                        else:
                            sp_seeds = torch.cat((seeds, sp_unique_src))

                sp_all_time_end = get_time()
                self.time_list[2] += sp_all_time_end - sp_all_time_begin

                mp_all_time_begin = get_time()
                # MP
                if self.model == "GAT":
                    mp_block, (mp_coo_row, mp_coo_col) = create_dgl_block(seeds, neighbors, fanout, True)
                    mp_unique_frontier = mp_block.srcdata["_ID"]
                    mp_send_frontier_size = torch.tensor([mp_unique_frontier.numel()])
                else:
                    mp_unique_frontier, mp_coo_row = tensor_relabel_csc(seeds, neighbors)

                (
                    mp_all_frontier,
                    mp_all_coo_row,
                    mp_send_size,
                    mp_recv_size,
                    mp_recv_frontier_size,
                    mp_recv_coo_size,
                ) = mp_sample_shuffle(seeds, mp_unique_frontier, mp_coo_row)

                if self.model == "GAT":
                    mp_sampling_result = (mp_send_frontier_size, mp_recv_frontier_size)
                else:
                    all_coo_col = torch.cat([torch.arange(0, i, device=mp_all_coo_row.device).repeat_interleave(fanout) for i in mp_recv_size])
                    mp_block = (mp_all_coo_row, all_coo_col, mp_recv_frontier_size, mp_recv_coo_size, mp_recv_size)
                    mp_sampling_result = (mp_send_size, mp_recv_size)
                mp_seeds = mp_all_frontier

                mp_all_time_end = get_time()
                self.time_list[3] += mp_all_time_end - mp_all_time_begin

        # return sample result for four PS
        return (
            (dp_seeds,),
            (np_seeds, np_send_offset, np_recv_offset),
            (sp_seeds,) + (sp_sampling_result),
            (mp_seeds,) + (mp_sampling_result),
        )
