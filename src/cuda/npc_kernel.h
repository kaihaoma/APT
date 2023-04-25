#ifndef NPC_KERNEL_H_
#define NPC_KERNEL_H_

#include <string>

#include "../utils.h"

namespace npc {
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 8 * WARP_SIZE;
const int BLOCK_NUM = 2;
const int MAX_NUM_DEVICES = 9;
const int NUM_THREADS = 1024;

void IndexSelectAll(
    IdType size, IdType feat_dim, torch::Tensor index,
    torch::Tensor feat_pos_map, torch::Tensor input_table_dev,
    torch::Tensor input_table_uva, torch::Tensor output_table,
    cudaStream_t stream = 0);

torch::Tensor LocalSampleNeighbors(
    torch::Tensor frontier, int fanout, cudaStream_t stream = 0);

// cluster seeds with global nid into $world_size$ bluckets
// return torch::Tensor bucket_size, bucket_offset, sorted_idx, permutation,
// rev_permutation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ClusterAndPermute(
    int rank, int world_size, torch::Tensor seeds, torch::Tensor min_vids,
    cudaStream_t stream = 0);
}  // namespace npc

#endif