#ifndef NPC_CORE_H_
#define NPC_CORE_H_

#include <torch/script.h>

#include <string>

namespace npc {

torch::Tensor NCCLGetUniqueId();

void Initialize(int64_t rank, int64_t world_size, torch::Tensor nccl_id_tensor);
void Test(torch::Tensor test_tensor, int64_t idx, double val);
}  // namespace npc

#endif