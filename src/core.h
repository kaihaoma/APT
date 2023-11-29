#ifndef NPC_CORE_H_
#define NPC_CORE_H_

#include <torch/script.h>

#include <string>

#include "utils.h"

namespace npc {

torch::Tensor NCCLGetUniqueId();

void Initialize(
    IdType rank, IdType local_rank, IdType world_size,
    torch::Tensor nccl_id_tensor_list, IdType node_size);

void RegisterMinVids(torch::Tensor shuffle_min_vids, IdType shuffle_id_offset);

void RegisterMultiMachinesScheme(
    torch::Tensor remote_worker_map, torch::Tensor remote_worker_id);
}  // namespace npc

#endif