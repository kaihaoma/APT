#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>

#include "./state.h"
#include "./utils.h"
#include "glog/logging.h"

namespace npc {

void CacheGraphs(
    IdType num_local_nodes, IdType num_graph_nodes, IdType num_cached_nodes,
    torch::Tensor sorted_idx, torch::Tensor indptr, torch::Tensor local_indices,
    torch::Tensor global_indices) {
  num_cached_nodes = num_graph_nodes;
  auto* state = NPCState::Global();
  auto rank = state->rank;
  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  state->graph_storage.num_graph_nodes = num_graph_nodes;
  state->graph_storage.dev_indptr = indptr.to(options);
  state->graph_storage.dev_local_indices = local_indices.to(options);
  state->graph_storage.dev_global_indices = global_indices.to(options);
  state->graph_storage.adj_pos_map = torch::arange(num_graph_nodes, options);
}
}  // namespace npc
