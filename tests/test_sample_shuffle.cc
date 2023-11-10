#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include "sampling.h"
#include "utils.h"

TEST(AllBroadcast, MPSampleShuffle) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto seeds = torch::arange(0, rank + 1, options);
  auto unique_frontier = torch::arange(0, rank + 1, options);
  auto coo_row = torch::arange(0, rank + 1, options);
  auto ret = npc::MPSampleShuffle(seeds, unique_frontier, coo_row);
  auto recv_frontier = ret[0];
  auto recv_coo_row = ret[1];

  // self make output
  std::vector<int> vecs;
  for (int r = 0; r < world_size; ++r) {
    for (int i = 0; i <= r; ++i) {
      vecs.emplace_back(i);
    }
  }
  auto make_recvs = torch::tensor(vecs, options);
  EXPECT_TRUE(recv_frontier.equal(make_recvs))
      << npc::TensorToString(recv_frontier)
      << "!=" << npc::TensorToString(make_recvs);
  EXPECT_TRUE(recv_coo_row.equal(make_recvs))
      << npc::TensorToString(recv_coo_row)
      << "!=" << npc::TensorToString(make_recvs);
}