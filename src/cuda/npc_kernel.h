#ifndef NPC_KERNEL_H_
#define NPC_KERNEL_H_

#include <torch/script.h>

#include <string>

#include "../utils.h"

namespace npc {
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 8 * WARP_SIZE;

void IndexSelectAll(
    IdType size, IdType feat_dim, torch::Tensor index,
    torch::Tensor feat_pos_map, torch::Tensor input_table_dev,
    torch::Tensor input_table_uva, torch::Tensor output_table,
    cudaStream_t stream = 0);
}  // namespace npc

#endif