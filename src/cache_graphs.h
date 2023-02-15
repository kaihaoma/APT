#ifndef NPC_CACHE_GRAPHS_H_
#define NPC_CACHE_GRAPHS_H_

#include <torch/script.h>

#include <string>

#include "./utils.h"

namespace npc {

void CacheGraphs(
    IdType num_local_nodes, IdType num_graph_nodes, IdType num_cached_nodes,
    torch::Tensor sorted_idx, torch::Tensor indptr, torch::Tensor indices);
}

#endif