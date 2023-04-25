#include "./load_subtensor.h"

#include "./cuda/npc_kernel.h"
#include "./state.h"

namespace npc {

torch::Tensor LoadSubtensor(torch::Tensor node_id) {
  int num_nodes = node_id.numel();
  auto* state = NPCState::Global();
  int rank = state->rank;

  auto input_dim = state->feat_storage.input_dim;
  auto feat_pos_map = state->feat_storage.feat_pos_map;
  auto dev_feats = state->feat_storage.dev_feats;
  auto uva_feats = state->feat_storage.uva_feats;

  // ret tensor
  torch::Tensor ret = torch::empty(
      {num_nodes, input_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rank));
  // load dev and uva node feat in one function
  IndexSelectAll(
      num_nodes, input_dim, node_id, feat_pos_map, dev_feats, uva_feats, ret);
  return ret;
}

}  // namespace npc