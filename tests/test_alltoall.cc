#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include "ops/collective.h"

TEST(AllToAll, AlltoallNCCL1) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto send = torch::arange(world_size, options);
  auto recv = torch::empty(world_size, options);
  auto arange = torch::arange(1, world_size + 1);
  auto res = torch::ones(world_size, options) * rank;
  npc::AlltoAll(send, recv, arange, arange);
  EXPECT_TRUE(recv.equal(res));
}