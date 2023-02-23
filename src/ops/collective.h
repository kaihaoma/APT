#ifndef NPC_OPS_COLLECTIVE_H_
#define NPC_OPS_COLLECTIVE_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include <vector>

namespace npc {

void AlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_offset,
    torch::Tensor recv_offset, int expand = 1);
torch::Tensor Allreduce(torch::Tensor t);

}  // namespace npc

#endif