#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include "sampling.h"

TEST(ShuffleSeeds, ShuffleSeeds1) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto seeds = torch::arange(world_size, options).flip(0);
  auto min_vids = torch::arange(world_size + 1, options);
  auto res = torch::ones(world_size, options) * rank;
  torch::Tensor recv_frontier, permutation, recv_offset, dev_offset;
  std::tie(recv_frontier, permutation, recv_offset, dev_offset) =
      npc::ShuffleSeeds(min_vids, seeds);
  EXPECT_TRUE(recv_frontier.equal(res));
}

TEST(ShuffleSeeds, ShuffleSeeds2) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto seeds = torch::cat({torch::arange(world_size, options).flip(0), torch::arange(world_size, options)});
  auto min_vids = torch::arange(world_size + 1, options);
  auto res = torch::ones(world_size * 2, options) * rank;
  torch::Tensor recv_frontier, permutation, recv_offset, dev_offset;
  std::tie(recv_frontier, permutation, recv_offset, dev_offset) =
      npc::ShuffleSeeds(min_vids, seeds);
  EXPECT_TRUE(recv_frontier.equal(res));
}