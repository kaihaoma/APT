#ifndef NPC_SAMPLING_H_
#define NPC_SAMPLING_H_

#include <torch/script.h>
#include <torch/serialize.h>

#include <string>

#include "./utils.h"
#include "glog/logging.h"

namespace npc {

std::vector<torch::Tensor> LocalSamplingNeibhorsOneLayer(
    torch::Tensor seeds, IdType fanout);
// return value:
// 1. seeds (original order)
// 2. neighbors (original order)
// 3. permutation (seeds[perm] = sorted_idx)
// 4. send_offset
// 5. recv_offset
std::vector<torch::Tensor> NPSampleAndShuffle(
    torch::Tensor seeds, IdType fanout);

// shuffle seeds based on min_vids
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ShuffleSeeds(torch::Tensor seeds);

torch::Tensor SrcToVir(IdType fanout, IdType num_dst, torch::Tensor src);

torch::Tensor SrcDsttoVir(IdType fanout, torch::Tensor dst, torch::Tensor src);

// SP sample shuffle
//  input: seeds, frontier, seeds
std::vector<torch::Tensor> SPSampleAndShuffle(
    IdType num_seeds, torch::Tensor send_frontier,
    torch::Tensor sorted_allnodes, torch::Tensor unique_frontier,
    IdType shuffle_with_dst);

// SP sample shuffle src nodes
// input: uniqued src nodes
std::vector<torch::Tensor> SPSampleShuffleSrc(torch::Tensor unique_src);

// MP sample shuffle
std::vector<torch::Tensor> MPSampleShuffle(
    torch::Tensor seeds, torch::Tensor unique_frontier, torch::Tensor coo_row);

}  // namespace npc

#endif