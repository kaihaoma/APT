#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include "ops/collective.h"
#include "utils.h"

// expand = 1
TEST(AllReduce, AllReduce1) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rank);

  auto send_size = torch::arange(1, world_size + 1);
  auto total_send_size = torch::sum(send_size).item<IdType>();
  auto send = rank * torch::ones(total_send_size, options);

  auto reduce_val = 1. * (world_size - 1) * world_size / 2;
  auto res = torch::full(rank + 1, reduce_val, options);
  auto recv = torch::empty(rank + 1, options);
  npc::AllReduce(send, recv, send_size);
  EXPECT_TRUE(recv.equal(res))
      << npc::TensorToString(recv) << "!=" << npc::TensorToString(res);
}

// expand = 2
TEST(AllReduce, AllReduce2) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rank);

  auto send_size = torch::arange(1, world_size + 1);
  auto total_send_size = torch::sum(send_size).item<IdType>();
  auto send = rank * torch::ones({total_send_size, 2}, options);

  auto reduce_val = 1. * (world_size - 1) * world_size / 2;
  auto res = torch::full({rank + 1, 2}, reduce_val, options);
  auto recv = torch::empty({rank + 1, 2}, options);
  npc::AllReduce(send, recv, send_size, 2);
  EXPECT_TRUE(recv.equal(res))
      << npc::TensorToString(recv) << "!=" << npc::TensorToString(res);
}