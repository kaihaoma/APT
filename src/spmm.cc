#include "./spmm.h"

#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>

#include "./cuda/npc_kernel.h"

namespace npc {
torch::Tensor CopyUSum(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor coo_offset, torch::Tensor input_offset,
    torch::Tensor output_offset) {
  int64_t out_num = output_offset[-1].item<int64_t>();
  auto output = torch::zeros({out_num, input.size(1)}, input.options());
  CopyUSumCUDA(
      coo_row, coo_col, input, output, coo_offset, input_offset, output_offset);

  return output;
}

torch::Tensor CopyESum(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor coo_offset, torch::Tensor output_offset) {
  int64_t out_num = output_offset[-1].item<int64_t>();
  auto output = torch::zeros({out_num, input.size(1)}, input.options());
  CopyESumCUDA(coo_row, coo_col, input, output, coo_offset, output_offset);

  return output;
}

torch::Tensor UMulESum(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor edata, torch::Tensor coo_offset, torch::Tensor input_offset,
    torch::Tensor output_offset) {
  int64_t out_num = output_offset[-1].item<int64_t>();
  auto output = torch::zeros({out_num, input.size(1)}, input.options());
  UMulESumCUDA(
      coo_row, coo_col, input, edata, output, coo_offset, input_offset,
      output_offset);

  return output;
}
}  // namespace npc