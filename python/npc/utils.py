import torch
import dgl
from dgl.heterograph import DGLBlock


def create_block_from_csc(indptr, indices, e_ids, num_src, num_dst):
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


def create_block_from_coo(row, col, num_src, num_dst):
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


def creat_dgl_block(seeds, neighbors, fanout):
    unique_tensor, indices = torch.ops.npc.relabel_csc(seeds, neighbors)
    indptr = torch.arange(0, seeds.numel() + 1) * fanout

    block = create_block_from_csc(
        indptr,
        indices,
        torch.tensor([]),
        num_src=unique_tensor.numel(),
        num_dst=seeds.numel(),
    )

    return block
