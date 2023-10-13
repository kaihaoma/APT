#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include "ops/collective.h"
#include "utils.h"

TEST(AllBroadcast, AllBroadcast1) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto seeds = torch::arange(0, rank + 1, options);
  int send_size = seeds.numel();
  auto tensor_send_size = torch::tensor({send_size});
  auto recv_size = torch::arange(1, world_size + 1);
  // total sizes
  int total_sizes = world_size * (world_size + 1) / 2;
  auto recvs = torch::empty(total_sizes, options);
  npc::AllBroadcast(seeds, recvs, tensor_send_size, recv_size);
  // self make output
  std::vector<int> vecs;
  for (int r = 0; r < world_size; ++r) {
    for (int i = 0; i <= r; ++i) {
      vecs.emplace_back(i);
    }
  }
  auto make_recvs = torch::tensor(vecs, options);
  EXPECT_TRUE(recvs.equal(make_recvs))
      << npc::TensorToString(recvs) << "!=" << npc::TensorToString(make_recvs);
}