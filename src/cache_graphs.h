#ifndef NPC_CACHE_GRAPHS_H_
#define NPC_CACHE_GRAPHS_H_

#include <torch/script.h>

#include <string>

#include "./utils.h"

namespace npc {

void CacheGraphs(
    IdType num_local_nodes, IdType num_graph_nodes, IdType num_cached_nodes,
    torch::Tensor sorted_idx, torch::Tensor indptr, torch::Tensor local_indices,
    torch::Tensor global_indices);

void MixCacheGraphs(
    IdType num_cached_nodes, torch::Tensor cached_node_idx,
    torch::Tensor cached_indptr, torch::Tensor cached_indices,
    torch::Tensor global_indptr, torch::Tensor global_indices);
}  // namespace npc

#endif