#include "./load_subtensor.h"

#include <c10/cuda/CUDAStream.h>

#include "./cuda/npc_kernel.h"
#include "./ops/collective.h"
#include "./state.h"

namespace npc {
torch::Tensor CPULoadSubtensor(torch::Tensor node_id) {
  auto* state = NPCState::Global();
  auto cpu_feat_pos_map = state->feat_storage.cpu_feat_pos_map;
  auto uva_feats = state->feat_storage.uva_feats;
  auto node_id_pos = ENCODE_ID(cpu_feat_pos_map.index_select(0, node_id));
  auto ret = uva_feats.index_select(0, node_id_pos);
  return ret;
}

torch::Tensor LoadSubtensor(torch::Tensor node_id) {
  int num_nodes = node_id.numel();
  auto* state = NPCState::Global();
  int local_rank = state->local_rank;

  auto feat_dim = state->feat_storage.feat_dim;
  auto rank_feat_dim = state->feat_storage.rank_feat_dim;
  auto feat_pos_map = state->feat_storage.feat_pos_map;
  auto dev_feats = state->feat_storage.dev_feats;
  auto uva_feats = state->feat_storage.uva_feats;
  auto feat_dim_offset = state->feat_storage.feat_dim_offset;
  // ret tensor
  torch::Tensor ret = torch::empty(
      {num_nodes, rank_feat_dim}, torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(torch::kCUDA, local_rank));
  // load dev and uva node feat in one function
  IndexSelectAll(
      num_nodes, feat_dim, feat_dim_offset, node_id, feat_pos_map, dev_feats,
      uva_feats, ret);
  return ret;
}
torch::Tensor CrossMachineLoadSubtensor(torch::Tensor node_id) {
  torch::Tensor bucket_size, sorted_idx, permutation;
  auto* state = NPCState::Global();
  int local_rank = state->local_rank;
  int rank = state->rank;
  auto num_remote_workers = state->num_remote_workers;
  auto cuda_options = node_id.options();
  auto cpu_options = torch::TensorOptions().dtype(torch::kInt64);
  auto feat_dim = state->feat_storage.feat_dim;
  auto num_nodes = node_id.numel();

  std::tie(bucket_size, sorted_idx, permutation) =
      MultiMachinesClusterAndPermute(node_id);

  //  local_req: sorted_idx[:bucket_size[0]]
  //  remote_req: sorted_idx[bucket_size[0]:]
  //  Custom alltoall size
  auto recv_size = torch::empty({num_remote_workers}, cuda_options);
  auto arange = torch::ones({num_remote_workers}, cpu_options);
  CrossMachineAlltoAll(bucket_size, recv_size, arange, arange);

  auto cpu_recv_size = recv_size.to(torch::kCPU);
  auto total_recv_size = cpu_recv_size.sum().item<int>();
  // LOG(INFO) << "total_recv_size: " << total_recv_size;
  auto all_req = torch::empty({total_recv_size}, cuda_options);
  auto cpu_bucket_size = bucket_size.to(torch::kCPU);
  // custom alltoall remote req
  CrossMachineAlltoAll(sorted_idx, all_req, cpu_bucket_size, cpu_recv_size);

  //  local uva feature loading in ONE KERNEL
  auto feats = LoadSubtensor(all_req);

  auto recv_feats = torch::empty({num_nodes * feat_dim}, feats.options());
  // custom alltoall send back feats to remote
  CrossMachineAlltoAll(
      feats, recv_feats, cpu_recv_size, cpu_bucket_size, feat_dim);

  recv_feats = recv_feats.reshape({num_nodes, feat_dim}).index({permutation});

  return recv_feats;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ClusterReqs(
    torch::Tensor node_id) {
  return MultiMachinesClusterAndPermute(node_id);
}

}  // namespace npc