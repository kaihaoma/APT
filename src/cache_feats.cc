#include "./cache_feats.h"

#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>

#include "./state.h"
#include "./utils.h"
#include "glog/logging.h"

namespace npc {

void CacheFeats(
    torch::Tensor node_feats, torch::Tensor sorted_idx,
    int64_t num_cached_nodes) {
  std::vector<DataType> vec_node_feats(
      node_feats.data_ptr<DataType>(),
      node_feats.data_ptr<DataType>() + node_feats.numel());

  std::vector<IdType> vec_sorted_idx(
      sorted_idx.data_ptr<IdType>(),
      sorted_idx.data_ptr<IdType>() + sorted_idx.numel());

  int64_t num_nodes = sorted_idx.numel();
  int64_t num_elements = node_feats.numel();
  CHECK(num_elements % num_nodes == 0);
  int64_t feat_dim = num_elements / num_nodes;

  DataType *dev_ptr, *host_ptr;
  std::vector<DataType> dev_feats(
      vec_node_feats.begin(),
      vec_node_feats.begin() + num_cached_nodes * feat_dim);
  NPCCudaMallocAndCopy(&dev_ptr, dev_feats);

  std::vector<DataType> host_feats(
      vec_node_feats.begin() + num_cached_nodes * feat_dim,
      vec_node_feats.end());
  NPCHostMallocAndCopy(&host_ptr, host_feats);

  auto* state = NPCState::Global();
  state->feat_pos_map.resize(num_nodes, 0);
  for (int i = 0; i < num_cached_nodes; ++i) {
    state->feat_pos_map[vec_sorted_idx[i]] = i;
  }
  for (int i = num_cached_nodes; i < num_nodes; ++i) {
    state->feat_pos_map[vec_sorted_idx[i]] = -(i - num_cached_nodes) - 2;
  }
  state->dev_feats = dev_ptr;
  state->num_dev_nodes = num_cached_nodes;
  state->host_feats = host_ptr;
  state->num_host_nodes = num_nodes - num_cached_nodes;
  // LOG

  LOG(INFO) << "#dev: " << state->num_dev_nodes
            << "\t #host: " << state->num_host_nodes;

  LOG(INFO) << "FPM: " << VecToString(state->feat_pos_map);
  LOG(INFO) << "host feats: "
            << ArrToString(state->host_feats, state->num_host_nodes);
  LOG(INFO) << "dev feats: "
            << DevArrToString(state->dev_feats, state->num_dev_nodes);
}
}  // namespace npc