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
  // state->graph_storage.dev_local_indices = local_indices.to(options);
  // state->graph_storage.dev_global_indices = global_indices.to(options);
  state->graph_storage.adj_pos_map = torch::arange(num_graph_nodes, options);
}
void MixCacheGraphs(
    IdType num_cached_nodes, torch::Tensor cached_node_idx,
    torch::Tensor cached_indptr, torch::Tensor cached_indices,
    torch::Tensor global_indptr, torch::Tensor global_indices) {
  auto* state = NPCState::Global();
  auto rank = state->rank;

  CHECK(num_cached_nodes == cached_node_idx.numel());
  auto num_total_nodes = global_indptr.numel() - 1;
  LOG(INFO) << "[NOTE] Mixed Cached graph #dev: " << num_cached_nodes
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
  // auto options =
  // torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  // state->graph_storage.dev_indptr = cached_indptr.to(options);
  // state->graph_storage.dev_indices = cached_indices.to(options);
  // state->graph_storage.uva_indptr = global_indptr.pin_memory();
  // state->graph_storage.uva_indices = global_indices.pin_memory();

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
          .to(torch::kCUDA, rank);
}
}  // namespace npc
