#ifndef NPC_SAMPLING_H_
#define NPC_SAMPLING_H_

#include <torch/script.h>

#include <string>

#include "./utils.h"

namespace npc {
// return value:
// 1. seeds (original order)
// 2. neighbors (original order)
// 3. permutation (seeds[perm] = sorted_idx)
// 4. reverse_permutation(sorted_idx[perm])
// 5. input_size_per_rank
// 6. output_size_per_rank
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SamplingNeighbors(torch::Tensor min_vids, torch::Tensor seeds, IdType fanout);

// shuffle seeds based on min_vids
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ShuffleSeeds(torch::Tensor min_vids, torch::Tensor seeds);
}  // namespace npc

#endif