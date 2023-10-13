#ifndef NPC_SAMPLING_H_
#define NPC_SAMPLING_H_

#include <torch/script.h>
#include <torch/serialize.h>

#include <string>

#include "./utils.h"
#include "glog/logging.h"

namespace npc {

const int SP_BITWISE_SHIFT = 20;

std::tuple<torch::Tensor, torch::Tensor> LocalSamplingNeibhorsOneLayer(
    torch::Tensor seeds, IdType fanout, IdType to_virtual = 0);
// return value:
// 1. seeds (original order)
// 2. neighbors (original order)
// 3. permutation (seeds[perm] = sorted_idx)
// 4. send_offset
// 5. recv_offset
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
NPSampleAndShuffle(torch::Tensor seeds, IdType fanout);

// shuffle seeds based on min_vids
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ShuffleSeeds(torch::Tensor seeds, IdType base1 = 1);

// SP sample shuffle
//  input: seeds, map_virtual_nodes, fanout
//  output:
//  1.shuffled_virtual_nodes
//  2.replicated_shuffled_virtual_nodes
//  3.original_neighbors
//  4.permutation (map_virtual_nodes)
//  5.send_offset (map_virtual_nodes)
//  6.recv_offset (map_virtual_nodes)
//  7.shuffled_seeds
//  8.permutation (seeds)
//  9.send_offset (seeds)
//  10.recv_offset (seeds)
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SPSampleAndShuffle2(
    torch::Tensor seeds, torch::Tensor map_virtual_nodes, IdType fanout);

torch::Tensor SrcDsttoVir(IdType fanout, torch::Tensor dst, torch::Tensor src);

// SP sample shuffle
//  input: seeds, frontier, seeds
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SPSampleAndShuffle(
    IdType num_seeds, torch::Tensor send_frontier,
    torch::Tensor sorted_allnodes, torch::Tensor unique_frontier);
// MP sample shuffle
// input: seeds, neighs
// output: gather_seeds, gather_neighs, send_size, recv_size
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
MPSampleShuffle(torch::Tensor seeds, torch::Tensor neighs);

}  // namespace npc

#endif