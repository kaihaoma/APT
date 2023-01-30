#ifndef NPC_CORE_H_
#define NPC_CORE_H_

#include <string>
#include <torch/script.h>

namespace npc {

torch::Tensor NCCLGetUniqueId();

void Initialize(int64_t rank, int64_t world_size, torch::Tensor nccl_id_tensor);

}

#endif