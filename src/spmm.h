#ifndef NPC_SPMM_H_
#define NPC_SPMM_H_

#include <torch/torch.h>

#include "./utils.h"

namespace npc {
torch::Tensor CopyUSum(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor coo_offset, torch::Tensor input_offset,
    torch::Tensor output_offset);

torch::Tensor CopyESum(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor coo_offset, torch::Tensor output_offset);

torch::Tensor UMulESum(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor edata, torch::Tensor coo_offset, torch::Tensor input_offset,
    torch::Tensor output_offset);
}  // namespace npc

#endif