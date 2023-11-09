#include "./sddmm.h"

#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>

#include "./cuda/npc_kernel.h"

namespace npc {
torch::Tensor UAddV(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor lhs,
    torch::Tensor rhs, torch::Tensor coo_offset, torch::Tensor lhs_offset,
    torch::Tensor rhs_offset) {
  int64_t out_num = coo_row.numel();
  auto output = torch::zeros({out_num, lhs.size(1)}, lhs.options());
  UAddVCUDA(
      coo_row, coo_col, lhs, rhs, output, coo_offset, lhs_offset, rhs_offset);

  return output;
}

torch::Tensor UMulV(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor lhs,
    torch::Tensor rhs, torch::Tensor coo_offset, torch::Tensor lhs_offset,
    torch::Tensor rhs_offset) {
  int64_t out_num = coo_row.numel();
  auto output = torch::zeros({out_num, lhs.size(1)}, lhs.options());
  UMulVCUDA(
      coo_row, coo_col, lhs, rhs, output, coo_offset, lhs_offset, rhs_offset);

  return output;
}
}  // namespace npc