#include "./cache_feats.h"

#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>

#include "./state.h"
#include "./utils.h"
#include "glog/logging.h"

namespace npc {

void CacheFeats(
    torch::Tensor node_feats, torch::Tensor sorted_idx, double cache_ratio,
    IdType num_total_nodes) {
  auto node_feat_sizes = node_feats.sizes();
  IdType num_graph_nodes = node_feat_sizes[0];
  IdType input_dim = node_feat_sizes[1];
  auto* state = NPCState::Global();
  int rank = state->rank;
  IdType num_cached_nodes =
      std::min(IdType(cache_ratio * num_graph_nodes), num_graph_nodes);
  LOG(INFO) << "Rank#" << rank << ": cache " << num_cached_nodes << " of "
            << num_graph_nodes << ", ratio: " << cache_ratio;
  state->feat_storage.num_total_nodes = num_total_nodes;
  state->feat_storage.num_graph_nodes = num_graph_nodes;
  state->feat_storage.num_dev_nodes = num_cached_nodes;
  state->feat_storage.num_uva_nodes = num_graph_nodes - num_cached_nodes;
  state->feat_storage.input_dim = input_dim;
  state->feat_storage.dev_feats =
      node_feats.index({torch::indexing::Slice(0, num_cached_nodes, 1)})
          .to(torch::kCUDA, rank);
  state->feat_storage.uva_feats =
      node_feats
          .index({torch::indexing::Slice(
              num_cached_nodes, torch::indexing::None, torch::indexing::None)})
          .pin_memory();
  std::vector<IdType> vec_sorted_idx(
      sorted_idx.data_ptr<IdType>(),
      sorted_idx.data_ptr<IdType>() + sorted_idx.numel());
  std::vector<IdType> host_feat_pos_map(num_total_nodes, -1);
  for (int i = 0; i < num_cached_nodes; ++i) {
    host_feat_pos_map[vec_sorted_idx[i]] = i;
  }

  for (int i = num_cached_nodes; i < num_graph_nodes; ++i) {
    host_feat_pos_map[vec_sorted_idx[i]] = ENCODE_ID(i - num_cached_nodes);
  }

  state->feat_storage.feat_pos_map =
      torch::from_blob(
          host_feat_pos_map.data(), {num_total_nodes},
          torch::TensorOptions().dtype(torch::kInt64))
          .to(torch::kCUDA, rank);
  //.pin_memory();
}
}  // namespace npc