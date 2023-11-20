#include "./feat_shuffle.h"

#include "./cuda/npc_kernel.h"
#include "./ops/collective.h"
#include "./state.h"

namespace npc {

torch::Tensor FeatShuffle(
    torch::Tensor inputs, torch::Tensor send_offset, torch::Tensor recv_offset,
    torch::Tensor permutation, IdType feat_dim, IdType fwd_flag) {
  auto* state = NPCState::Global();
  auto flatten_inputs = fwd_flag ? inputs.flatten()
                                 : torch::index_select_backward(
                                       inputs, inputs.sizes(), 0, permutation)
                                       .flatten();
  auto outputs_size = recv_offset[-1].item<IdType>();
  auto flatten_outputs =
      torch::empty(outputs_size * feat_dim, inputs.options());

  AlltoAll(
      flatten_inputs, flatten_outputs, send_offset, recv_offset, feat_dim,
      state->trainer_id);

  auto outputs =
      fwd_flag
          ? flatten_outputs.reshape({-1, feat_dim}).index_select(0, permutation)
          : flatten_outputs.reshape({-1, feat_dim});
  return outputs;
}

torch::Tensor SPFeatShuffle(
    torch::Tensor inputs, torch::Tensor send_sizes, torch::Tensor recv_sizes,
    IdType total_recv_size, IdType expand, IdType shuffle_with_dst) {
  auto* state = NPCState::Global();
  auto world_size = state->world_size;
  auto flatten_inputs = inputs.flatten();
  auto flatten_outputs =
      torch::empty(total_recv_size * expand, inputs.options());

  if (shuffle_with_dst) {
    SPFeatureAlltoAllWithDst(
        flatten_inputs, flatten_outputs, send_sizes, recv_sizes, expand,
        state->trainer_id);
  } else {
    SPFeatureAlltoAll(
        flatten_inputs, flatten_outputs, send_sizes, recv_sizes, expand,
        state->trainer_id);
  }
  auto outputs = flatten_outputs.reshape({-1, expand});
  return outputs;
}

torch::Tensor MPFeatShuffleFwd(
    torch::Tensor inputs, torch::Tensor send_size, torch::Tensor recv_size,
    IdType feat_dim) {
  auto* state = NPCState::Global();
  auto flatten_inputs = inputs.flatten();
  auto flatten_outputs =
      torch::empty(recv_size.item<IdType>() * feat_dim, inputs.options());
  AllReduce(flatten_inputs, flatten_outputs, send_size, feat_dim);
  auto outputs = flatten_outputs.reshape({-1, feat_dim});
  return outputs;
}

torch::Tensor MPFeatShuffleBwd(
    torch::Tensor inputs, torch::Tensor send_size, torch::Tensor recv_size,
    IdType feat_dim) {
  auto* state = NPCState::Global();
  auto world_size = state->world_size;
  auto total_recv_size = torch::sum(recv_size).item<IdType>();
  auto flatten_inputs = inputs.flatten();
  auto flatten_outputs =
      torch::empty(total_recv_size * feat_dim, inputs.options());
  AllBroadcastV2(
      flatten_inputs, flatten_outputs, send_size, recv_size, feat_dim);
  auto outputs = flatten_outputs.reshape({-1, feat_dim});
  return outputs;
}

}  // namespace npc