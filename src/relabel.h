#ifndef NPC_RELABEL_H_
#define NPC_RELABEL_H_

#include <torch/torch.h>

#include "./utils.h"
#include "glog/logging.h"

namespace npc {
std::vector<torch::Tensor> RelabelCSC(
    torch::Tensor seeds, torch::Tensor neighbors);
}  // namespace npc

#endif