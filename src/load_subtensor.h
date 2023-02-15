#ifndef NPC_LOAD_SUBTENSOR_H_
#define NPC_LOAD_SUBTENSOR_H_

#include <torch/script.h>

#include <string>

namespace npc {

torch::Tensor LoadSubtensor(torch::Tensor node_id);
}  // namespace npc

#endif