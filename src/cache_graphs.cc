#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>

#include "./state.h"
#include "./utils.h"
#include "glog/logging.h"

namespace npc {

void CacheGraphs(
    IdType num_local_nodes, IdType num_graph_nodes, IdType num_cached_nodes,
    torch::Tensor sorted_idx, torch::Tensor indptr, torch::Tensor indices) {
  LOG(INFO) << "To simplify the sampling process, cache all graphs into UVA";
  num_cached_nodes = num_graph_nodes;
  auto* state = NPCState::Global();
  auto rank = state->rank;
  state->graph_storage.num_graph_nodes = num_graph_nodes;
  state->graph_storage.dev_indptr = indptr;
  state->graph_storage.dev_indices = indices;
  state->graph_storage.adj_pos_map = torch::arange(
      num_graph_nodes,
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank));
}
}  // namespace npc
