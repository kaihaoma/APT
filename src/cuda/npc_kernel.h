#ifndef NPC_KERNEL_H_
#define NPC_KERNEL_H_

#include <string>

#include "../utils.h"
#include "glog/logging.h"

namespace npc {
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 8 * WARP_SIZE;
const int BLOCK_NUM = 2;
const int MAX_NUM_DEVICES = 33;
const int NUM_THREADS = 1024;

void IndexSelectAll(
    IdType size, IdType feat_dim, IdType feat_dim_offset, torch::Tensor index,
    torch::Tensor feat_pos_map, torch::Tensor input_table_dev,
    torch::Tensor input_table_uva, torch::Tensor output_table);

torch::Tensor LocalSampleNeighbors(torch::Tensor frontier, int fanout);

// cluster seeds with global nid into $world_size$ bluckets
// return torch::Tensor bucket_size, bucket_offset, sorted_idx, permutation,
// rev_permutation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ClusterAndPermute(
    int world_size, torch::Tensor seeds, IdType shuffle_id_offset,
    torch::Tensor shuffle_min_vids, int rank, int node_beg, int node_end);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MultiMachinesClusterAndPermute(torch::Tensor seeds);

// map dst&src to vir
torch::Tensor MapSrcDsttoVir(
    IdType world_size, IdType fanout, torch::Tensor dst, torch::Tensor src,
    IdType shuffle_id_offset, torch::Tensor shuffle_min_vids, int rank, int node_beg, int node_end);

    

// map src to vir
torch::Tensor MapSrctoVir(
    IdType world_size, IdType fanout, IdType num_dst, torch::Tensor src,
    IdType shuffle_id_offset, torch::Tensor shuffle_min_vids, int rank, int node_beg, int node_end);

// lower_bound i*base for torch.cat((sorted_nodes, unique_nodes)
torch::Tensor GetVirSendOffset(
    int world_size, int base, torch::Tensor sorted_nodes,
    torch::Tensor unique_nodes);

// lower_bound i*base for torch.cat((sorted_nodes, unique_nodes)
torch::Tensor GetVirSendOffsetWithDst(
    int world_size, int base, torch::Tensor sorted_nodes,
    torch::Tensor unique_nodes);

std::pair<torch::Tensor, torch::Tensor> TensorRelabelCSC(
    torch::Tensor seeds, torch::Tensor neighbors);

void CopyUSumCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor output, torch::Tensor coo_offset, torch::Tensor input_offset,
    torch::Tensor output_offset);

void CopyESumCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor output, torch::Tensor coo_offset,
    torch::Tensor output_offset);

void UMulESumCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor edata, torch::Tensor output, torch::Tensor coo_offset,
    torch::Tensor input_offset, torch::Tensor output_offset);

void UAddVCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor lhs,
    torch::Tensor rhs, torch::Tensor output, torch::Tensor coo_offset,
    torch::Tensor lhs_offset, torch::Tensor rhs_offset);

void UMulVCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor lhs,
    torch::Tensor rhs, torch::Tensor output, torch::Tensor coo_offset,
    torch::Tensor lhs_offset, torch::Tensor rhs_offset);
}  // namespace npc

#endif