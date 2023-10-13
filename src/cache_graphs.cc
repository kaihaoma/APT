#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>

#include "./state.h"
#include "./utils.h"
#include "glog/logging.h"

namespace npc {

void MixCacheGraphs(
    IdType num_cached_nodes, torch::Tensor cached_node_idx,
    torch::Tensor cached_indptr, torch::Tensor cached_indices,
    torch::Tensor global_indptr, torch::Tensor global_indices) {
  auto* state = NPCState::Global();
  auto local_rank = state->local_rank;

  CHECK(num_cached_nodes == cached_node_idx.numel());
  auto num_total_nodes = global_indptr.numel() - 1;
  LOG(INFO) << "[Note] Mixed Cached graph #dev: " << num_cached_nodes
            << "\t #uva: " << num_total_nodes;
  LOG(INFO) << "[Note] cached_indptr: " << cached_indptr.sizes()
            << "\t indices: " << cached_indices.sizes();
  LOG(INFO) << "[Note] global_indptr: " << global_indptr.sizes()
            << "\t indices: " << global_indices.sizes();
  state->graph_storage.num_cached_nodes = num_cached_nodes;
  state->graph_storage.num_total_nodes = num_total_nodes;

  state->graph_storage.dev_indptr = cached_indptr;
  state->graph_storage.dev_indices = cached_indices;
  state->graph_storage.uva_indptr = global_indptr;
  state->graph_storage.uva_indices = global_indices;
  std::vector<IdType> vec_cached_node_idx(
      cached_node_idx.data_ptr<IdType>(),
      cached_node_idx.data_ptr<IdType>() + num_cached_nodes);
  std::vector<IdType> adj_pos_map(num_total_nodes);
  for (IdType i = 0; i < num_total_nodes; ++i) {
    adj_pos_map[i] = ENCODE_ID(i);
  }
  for (IdType i = 0; i < num_cached_nodes; ++i) {
    adj_pos_map[vec_cached_node_idx[i]] = i;
  }
  state->graph_storage.adj_pos_map =
      torch::from_blob(
          adj_pos_map.data(), {num_total_nodes},
          torch::TensorOptions().dtype(torch::kInt64))
          .to(torch::kCUDA, local_rank);
}
}  // namespace npc
