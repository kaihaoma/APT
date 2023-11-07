#ifndef NPC_KERNEL_H_
#define NPC_KERNEL_H_

#include <string>

#include "../utils.h"
#include "glog/logging.h"

namespace npc {
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 8 * WARP_SIZE;
const int BLOCK_NUM = 2;
const int MAX_NUM_DEVICES = 17;
const int NUM_THREADS = 1024;

void IndexSelectAll(
    IdType size, IdType feat_dim, IdType feat_dim_offset, torch::Tensor index,
    torch::Tensor feat_pos_map, torch::Tensor input_table_dev,
    torch::Tensor input_table_uva, torch::Tensor output_table);

torch::Tensor LocalSampleNeighbors(
    torch::Tensor frontier, int fanout, bool to_virtual = false);

// cluster seeds with global nid into $world_size$ bluckets
// return torch::Tensor bucket_size, bucket_offset, sorted_idx, permutation,
// rev_permutation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ClusterAndPermute(int world_size, torch::Tensor seeds, torch::Tensor min_vids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MultiMachinesClusterAndPermute(torch::Tensor seeds);

// map src, dst to vir
torch::Tensor MapSrcDsttoVir(
    IdType world_size, IdType fanout, torch::Tensor dst, torch::Tensor src,
    torch::Tensor min_vids);

// map src to vir
torch::Tensor MapSrctoVir(
    IdType world_size, IdType fanout, IdType num_dst, torch::Tensor src,
    torch::Tensor min_vids);

// lower_bound i*base for torch.cat((sorted_nodes, unique_nodes)
torch::Tensor GetVirSendOffset(
    int world_size, int base, torch::Tensor sorted_nodes,
    torch::Tensor unique_nodes);

// lower_bound i*base for torch.cat((sorted_nodes, unique_nodes)
torch::Tensor GetVirSendOffsetV2(
    int world_size, int base, torch::Tensor sorted_nodes,
    torch::Tensor unique_nodes);

std::pair<torch::Tensor, torch::Tensor> TensorRelabelCSC(
    torch::Tensor seeds, torch::Tensor neighbors);

void CopyUSumCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor output, torch::Tensor coo_offset, torch::Tensor input_offset,
    torch::Tensor output_offset);

}  // namespace npc

#endif