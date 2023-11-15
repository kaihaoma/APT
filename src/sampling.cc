#include "./sampling.h"

#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>

#include "./cuda/npc_kernel.h"
#include "./ops/collective.h"
#include "./state.h"
#include "glog/logging.h"

namespace npc {

std::vector<torch::Tensor> LocalSamplingNeibhorsOneLayer(
    torch::Tensor seeds, IdType fanout) {
  auto local_neighbors = LocalSampleNeighbors(seeds, fanout);
  return {seeds, local_neighbors};
}

torch::Tensor SrcToVir(IdType fanout, IdType num_dst, torch::Tensor src) {
  auto* state = NPCState::Global();
  auto world_size = state->world_size;
  auto shuffle_id_offset = state->shuffle_id_offset;
  auto shuffle_min_vids = state->shuffle_min_vids;
  auto map_src = MapSrctoVir(
      world_size, fanout, num_dst, src, shuffle_id_offset, shuffle_min_vids);
  return map_src;
}

std::vector<torch::Tensor> NPSampleAndShuffle(
    torch::Tensor seeds, IdType fanout) {
  torch::Tensor shuffled_frontier, permutation, recv_offset, dev_offset;
  std::tie(shuffled_frontier, permutation, recv_offset, dev_offset) =
      ShuffleSeeds(seeds);
  auto local_neighbors = LocalSampleNeighbors(shuffled_frontier, fanout);

  return {
      shuffled_frontier, local_neighbors, permutation, recv_offset, dev_offset};
}

std::vector<torch::Tensor> SPSampleAndShuffle(
    IdType num_seeds, torch::Tensor send_frontier,
    torch::Tensor sorted_allnodes, torch::Tensor unique_frontier) {
  auto* state = NPCState::Global();
  auto rank = state->rank;
  auto world_size = state->world_size;
  auto num_total_nodes = state->graph_storage.num_total_nodes;

  // map src to vir
  // rules: [belong, dst_idx]
  // map = belong * base2 + dst_idx
  auto base2 = num_seeds;

  auto cuda_options = sorted_allnodes.options();

  // send_offset contains two parts sorted_allnodes idx:[0:world_size+1) and
  // unique_frontier idx:[world_size+1, 2*world_size+1)
  auto send_offset =
      GetVirSendOffset(world_size, base2, sorted_allnodes, unique_frontier);

  auto fir_uni = send_offset[world_size + 1].item<IdType>();
  auto send_sizes = send_offset.diff();
  send_sizes.index_put_({world_size}, fir_uni);

  send_sizes = send_sizes.index({state->sp_alltoall_size_permute}).contiguous();

  auto arange = torch::arange(1, world_size + 1) * 2;
  auto recv_sizes = torch::empty_like(send_sizes);

  AlltoAll(send_sizes, recv_sizes, arange, arange);
  // all-to-all packed (map_src & frontier)
  send_sizes = send_sizes.to(torch::kCPU);
  recv_sizes = recv_sizes.to(torch::kCPU);

  auto feat_recv_sizes = torch::sum(recv_sizes.index({torch::indexing::Slice(
                                        1, torch::indexing::None, 2)}))
                             .item<IdType>();

  auto recv_size = torch::sum(recv_sizes).item<IdType>() - feat_recv_sizes;

  auto recv_src = torch::empty(recv_size, cuda_options);

  SPSampleAlltoAll(send_frontier, recv_src, send_sizes, recv_sizes);

  auto ori_src = recv_src % num_total_nodes;
  auto vir_src = recv_src.div(num_total_nodes, "trunc");

  return {vir_src, ori_src, send_sizes, recv_sizes};
}

std::vector<torch::Tensor> SPSampleShuffleSrc(torch::Tensor unique_src) {
  torch::Tensor shuffled_src, permutation, recv_offset, dev_offset;
  std::tie(shuffled_src, permutation, recv_offset, dev_offset) =
      ShuffleSeeds(unique_src);

  return {shuffled_src, permutation, recv_offset, dev_offset};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ShuffleSeeds(torch::Tensor seeds) {
  auto* state = NPCState::Global();
  int rank = state->rank;
  int world_size = state->world_size;
  auto cuda_tensor_option = seeds.options();
  auto shuffle_id_offset = state->shuffle_id_offset;
  auto shuffle_min_vids = state->shuffle_min_vids;

  torch::Tensor dev_size, dev_offset, sorted_idx, permutation;
  std::tie(dev_size, dev_offset, sorted_idx, permutation) =
      ClusterAndPermute(world_size, seeds, shuffle_id_offset, shuffle_min_vids);

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

std::vector<torch::Tensor> MPSampleShuffle(
    torch::Tensor seeds, torch::Tensor unique_frontier, torch::Tensor coo_row) {
  auto* state = NPCState::Global();
  auto world_size = state->world_size;
  auto rank = state->rank;
  auto seeds_size = seeds.numel();
  auto frontier_size = unique_frontier.numel();
  auto coo_size = coo_row.numel();
  auto cuda_tensor_option = unique_frontier.options();
  // all gather send size
  auto arange = torch::arange(1, world_size + 1) * 2;

  auto send_size =
      torch::tensor({seeds_size, frontier_size, coo_size}, cuda_tensor_option);
  auto recv_size = AllGather(send_size).to(torch::kCPU);

  auto recv_seeds_size =
      recv_size.index({torch::indexing::Slice(0, torch::indexing::None, 3)})
          .contiguous();

  auto recv_frontier_size =
      recv_size.index({torch::indexing::Slice(1, torch::indexing::None, 3)})
          .contiguous();

  auto recv_coo_size =
      recv_size.index({torch::indexing::Slice(2, torch::indexing::None, 3)})
          .contiguous();

  // all boardcast seeds and neighbors
  auto recv_frontier_total_size = torch::sum(recv_frontier_size).item<IdType>();
  auto recv_coo_total_size = torch::sum(recv_coo_size).item<IdType>();

  auto recv_frontier =
      torch::empty(recv_frontier_total_size, cuda_tensor_option);
  auto recv_coo_row = torch::empty(recv_coo_total_size, cuda_tensor_option);

  AllBroadcastV2(
      unique_frontier, recv_frontier, torch::tensor({frontier_size}),
      recv_frontier_size);
  AllBroadcastV2(
      coo_row, recv_coo_row, torch::tensor({coo_size}), recv_coo_size);

  return {recv_frontier,   recv_coo_row,       torch::tensor({seeds_size}),
          recv_seeds_size, recv_frontier_size, recv_coo_size};
}

}  // namespace npc
