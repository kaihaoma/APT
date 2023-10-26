#include "./relabel.h"

#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>

#include "./cuda/npc_kernel.h"

namespace npc {
std::vector<torch::Tensor> RelabelCSC(
    torch::Tensor seeds, torch::Tensor neighbors) {
  torch::Tensor all_unique, relabeled_indices;
  std::tie(all_unique, relabeled_indices) = TensorRelabelCSC(seeds, neighbors);

  return {all_unique, relabeled_indices};
}
}  // namespace npc