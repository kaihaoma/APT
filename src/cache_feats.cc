#include "./cache_feats.h"

#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>

#include "./state.h"
#include "./utils.h"
#include "glog/logging.h"

namespace npc {

void CacheFeatsShared(
    IdType num_total_nodes, torch::Tensor localnode_feats,
    torch::Tensor cached_feats, torch::Tensor cached_idx,
    torch::Tensor localnode_idx, IdType feat_dim_offset) {
  auto localnode_feat_sizes = localnode_feats.sizes();
  auto cached_feat_sizes = cached_feats.sizes();
  IdType num_localnode_nodes = localnode_feat_sizes[0];
  IdType feat_dim = localnode_feat_sizes[1];
  IdType num_cached_nodes = cached_feat_sizes[0];
  IdType rank_feat_dim = cached_feat_sizes[1];
  auto* state = NPCState::Global();

  int local_rank = state->rank;
  LOG(INFO) << "Cache #" << num_cached_nodes << "\t of uva"
            << num_localnode_nodes << "\t #total: " << num_total_nodes << "\n";
  LOG(INFO) << "cached_idx: " << cached_idx.numel()
            << "\tinput_dim: " << rank_feat_dim << "\n";
  LOG(INFO) << "localnode shape: " << localnode_feat_sizes[0] << ", "
            << localnode_feat_sizes[1]
            << "\t feat_dim_offset: " << feat_dim_offset << "\n";
  state->feat_storage.num_total_nodes = num_total_nodes;
  state->feat_storage.num_uva_nodes = num_localnode_nodes;
  state->feat_storage.num_dev_nodes = num_cached_nodes;
  state->feat_storage.dev_feats = cached_feats;
  state->feat_storage.uva_feats = localnode_feats;
  state->feat_storage.feat_dim = feat_dim;
  state->feat_storage.rank_feat_dim = rank_feat_dim;
  state->feat_storage.feat_dim_offset = feat_dim_offset;
  std::vector<IdType> vec_cached_idx(
      cached_idx.data_ptr<IdType>(),
      cached_idx.data_ptr<IdType>() + cached_idx.numel());
  std::vector<IdType> vec_localnode_idx(
      localnode_idx.data_ptr<IdType>(),
      localnode_idx.data_ptr<IdType>() + localnode_idx.numel());
  std::vector<IdType> host_feat_pos_map(num_total_nodes, FEAT_NOT_EXIST);
  for (int i = 0; i < num_localnode_nodes; ++i) {
    host_feat_pos_map[vec_localnode_idx[i]] = ENCODE_ID(i);
  }
  state->feat_storage.cpu_feat_pos_map = torch::tensor(host_feat_pos_map);
  for (int i = 0; i < num_cached_nodes; ++i) {
    host_feat_pos_map[vec_cached_idx[i]] = i;
  }
  state->feat_storage.feat_pos_map =
      torch::from_blob(
          host_feat_pos_map.data(), {num_total_nodes},
          torch::TensorOptions().dtype(torch::kInt64))
          .to(torch::kCUDA, local_rank);
}

}  // namespace npc