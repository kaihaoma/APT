#include "./sampling.h"

#include "./cuda/npc_kernel.h"
#include "./ops/collective.h"
#include "./state.h"

namespace npc {

std::tuple<torch::Tensor, torch::Tensor> LocalSamplingNeibhorsOneLayer(
    torch::Tensor seeds, IdType fanout) {
  auto local_neighbors = LocalSampleNeighbors(seeds, fanout);
  return {seeds, local_neighbors};
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SamplingNeighbors(torch::Tensor min_vids, torch::Tensor seeds, IdType fanout) {
  auto* state = NPCState::Global();
  int rank = state->rank;
  int world_size = state->world_size;
  auto cuda_tensor_option = seeds.options();

  torch::Tensor dev_size, dev_offset, sorted_idx, permutation;
  std::tie(dev_size, dev_offset, sorted_idx, permutation) =
      ClusterAndPermute(rank, world_size, seeds, min_vids);

  // All-to-all send sizes
  auto arange = torch::arange(1, world_size + 1);
  auto recv_sizes = torch::empty(world_size, cuda_tensor_option);
  AlltoAll(dev_size, recv_sizes, arange, arange);

  recv_sizes = recv_sizes.to(torch::kCPU);
  dev_size = dev_size.to(torch::kCPU);
  auto recv_offset = recv_sizes.cumsum(0);
  dev_offset = dev_offset.to(torch::kCPU);
  auto recv_size = recv_offset[world_size - 1].item<IdType>();
  auto recv_frontier = torch::empty(recv_size, cuda_tensor_option);

  AlltoAll(sorted_idx, recv_frontier, dev_offset, recv_offset);

  auto local_neighbors = LocalSampleNeighbors(recv_frontier, fanout);

  return {recv_frontier, local_neighbors, permutation, recv_offset, dev_offset};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ShuffleSeeds(torch::Tensor min_vids, torch::Tensor seeds) {
  auto* state = NPCState::Global();
  int rank = state->rank;
  int world_size = state->world_size;
  auto cuda_tensor_option = seeds.options();

  torch::Tensor dev_size, dev_offset, sorted_idx, permutation;
  std::tie(dev_size, dev_offset, sorted_idx, permutation) =
      ClusterAndPermute(rank, world_size, seeds, min_vids);

  // All-to-all send sizes
  auto arange = torch::arange(1, world_size + 1);

  auto recv_sizes = torch::empty(world_size, cuda_tensor_option);
  AlltoAll(dev_size, recv_sizes, arange, arange);

  recv_sizes = recv_sizes.to(torch::kCPU);
  dev_size = dev_size.to(torch::kCPU);
  auto recv_offset = recv_sizes.cumsum(0);
  dev_offset = dev_offset.to(torch::kCPU);
  auto recv_size = recv_offset[world_size - 1].item<IdType>();
  auto recv_frontier = torch::empty(recv_size, cuda_tensor_option);

  AlltoAll(sorted_idx, recv_frontier, dev_offset, recv_offset);

  return {recv_frontier, permutation, recv_offset, dev_offset};
}

}  // namespace npc
