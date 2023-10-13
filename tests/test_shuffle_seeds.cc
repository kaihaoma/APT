#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include "core.h"
#include "sampling.h"

TEST(ShuffleSeeds, ShuffleSeeds1) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto gpu_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);

  auto cpu_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

  auto base = 100;
  torch::Tensor recv_frontier, permutation, recv_offset, dev_offset;
  // register min_vids
  auto min_vids = base * torch::arange(world_size + 1, gpu_options);

  auto seeds = world_size * torch::arange((base - 1)) + rank;
  seeds = seeds.to(gpu_options);
  npc::RegisterMinVids(min_vids);
  std::tie(recv_frontier, permutation, recv_offset, dev_offset) =
      npc::ShuffleSeeds(seeds);
  LOG(INFO) << "Rk#" << rank
            << "recv_frontier: " << npc::TensorToString(recv_frontier) << "\n";

  LOG(INFO) << "Rk#" << rank
            << "permutation: " << npc::TensorToString(permutation) << "\n";
  LOG(INFO) << "Rk#" << rank
            << "recv_offset: " << npc::TensorToString(recv_offset) << "\n";

  LOG(INFO) << "Rk#" << rank
            << "dev_offset: " << npc::TensorToString(dev_offset) << "\n";

  //  EXPECT_TRUE(recv_frontier.equal(res));
}

TEST(ShuffleSeeds, ShuffleSeeds2) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto seeds = torch::cat(
      {torch::arange(world_size, options).flip(0),
       torch::arange(world_size, options)});
  auto min_vids = torch::arange(world_size + 1, options);
  auto res = torch::ones(world_size * 2, options) * rank;
  torch::Tensor recv_frontier, permutation, recv_offset, dev_offset;
  // std::tie(recv_frontier, permutation, recv_offset, dev_offset) =
  // npc::ShuffleSeeds(min_vids, seeds); EXPECT_TRUE(recv_frontier.equal(res));
}