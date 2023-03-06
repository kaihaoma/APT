#ifndef NPC_FEAT_SHUFFLE_H_
#define NPC_FEAT_SHUFFLE_H_

#include <torch/script.h>

#include <string>

#include "./utils.h"

namespace npc {
torch::Tensor FeatShuffle(
    torch::Tensor inputs, torch::Tensor send_offset, torch::Tensor recv_offset,
    torch::Tensor permutation, IdType feat_dim, IdType fwd_flag);
}  // namespace npc

#endif