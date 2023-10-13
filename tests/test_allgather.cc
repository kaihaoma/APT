#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include "ops/collective.h"
#include "utils.h"

TEST(AllGather, AllGather1) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);

  auto send = torch::tensor({rank, rank * 10}, options);
  std::cerr << "send: " << npc::TensorToString(send) << "\n";
  auto recv = npc::AllGather(send);
  std::vector<int> vec;
  for (int r = 0; r < world_size; ++r) {
    vec.push_back(r);
    vec.push_back(r * 10);
  }
  auto res = torch::tensor(vec, options);

  EXPECT_TRUE(recv.equal(res))
      << npc::TensorToString(recv) << "!=" << npc::TensorToString(res);
}