#ifndef NPC_CORE_H_
#define NPC_CORE_H_

#include <torch/script.h>

#include <string>

#include "utils.h"

namespace npc {

torch::Tensor NCCLGetUniqueId();

void Initialize(
    IdType rank, IdType local_rank, IdType world_size,
    torch::Tensor nccl_id_tensor_list);

void RegisterMinVids(torch::Tensor min_vids);

void RegisterMultiMachinesScheme(
    torch::Tensor remote_worker_map, torch::Tensor remote_worker_id);
}  // namespace npc

#endif