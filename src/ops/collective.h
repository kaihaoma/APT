#ifndef NPC_OPS_COLLECTIVE_H_
#define NPC_OPS_COLLECTIVE_H_

#include <torch/script.h>
#include <torch/custom_class.h>

#include <vector>

namespace npc {

torch::Tensor Allreduce(torch::Tensor t);

}  // namespace npc

#endif