#include "./feat_shuffle.h"

#include "./cuda/npc_kernel.h"
#include "./ops/collective.h"
#include "./state.h"

namespace npc {
torch::Tensor FeatShuffle(
    torch::Tensor inputs, torch::Tensor send_offset, torch::Tensor recv_offset,
    torch::Tensor permutation, IdType feat_dim, IdType fwd_flag) {
  auto flatten_inputs = fwd_flag ? inputs.flatten()
                                 : torch::index_select_backward(
                                       inputs, inputs.sizes(), 0, permutation)
                                       .flatten();
  auto outputs_size = recv_offset[-1].item<IdType>();
  auto flatten_outputs =
      torch::empty(outputs_size * feat_dim, inputs.options());

  AlltoAll(flatten_inputs, flatten_outputs, send_offset, recv_offset, feat_dim);

  auto outputs =
      fwd_flag
          ? flatten_outputs.reshape({-1, feat_dim}).index_select(0, permutation)
          : flatten_outputs.reshape({-1, feat_dim});

  return outputs;
}
}  // namespace npc