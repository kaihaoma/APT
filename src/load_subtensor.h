#ifndef NPC_LOAD_SUBTENSOR_H_
#define NPC_LOAD_SUBTENSOR_H_

#include <torch/script.h>

#include <string>

namespace npc {
// Single machine cpu load subtensor by torch.index_select
torch::Tensor CPULoadSubtensor(torch::Tensor node_id);
// Single machine load subtensor by CUDA kernel
torch::Tensor LoadSubtensor(torch::Tensor node_id);

// Cross machine load subtensor
torch::Tensor CrossMachineLoadSubtensor(torch::Tensor node_id);

// Multiple machine cluster reqs into local and remote reqs
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ClusterReqs(
    torch::Tensor node_id);

}  // namespace npc

#endif