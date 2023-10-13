#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include "ops/collective.h"
#include "utils.h"

TEST(AllToAll, AlltoallNCCL1) {
  // Alltoall with expand 1
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

TEST(AllToAll, AlltoallNCCL2) {
  // Alltoall with expand 1, different send/recv size
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto gpu_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto cpu_options = torch::TensorOptions().dtype(torch::kInt64);
  // para
  int k = 7, b = 100;
  std::vector<IdType> send_vecs(world_size), recv_vecs(world_size);
  for (int r = 0; r < world_size; ++r) {
    send_vecs[r] = k * (rank - r) * (rank - r) + b;
    recv_vecs[r] = k * (r - rank) * (r - rank) + b;
  }
  auto send_size = torch::tensor(send_vecs, cpu_options);
  auto recv_size = torch::tensor(recv_vecs, cpu_options);
  auto send_offset = send_size.cumsum(0);
  auto recv_offset = recv_size.cumsum(0);
  auto num_send = send_offset[-1].item<IdType>();
  auto num_recv = recv_offset[-1].item<IdType>();
  auto send_val = torch::full(num_send, rank, gpu_options);
  auto recv_val = torch::empty(num_recv, gpu_options);
  std::vector<IdType> recv_val_vecs(num_recv);
  int offset = 0;
  for (int r = 0; r < world_size; ++r) {
    for (int i = 0; i < recv_vecs[r]; ++i) {
      recv_val_vecs[offset++] = r;
    }
  }

  LOG(INFO) << "Rk#" << rank << "send_size: " << npc::TensorToString(send_size)
            << "\t recv_size: " << npc::TensorToString(recv_size) << "\n";

  npc::AlltoAll(send_val, recv_val, send_offset, recv_offset);
  auto check_val = torch::tensor(recv_val_vecs, gpu_options);
  EXPECT_TRUE(recv_val.equal(check_val));
}

TEST(AlltoAll, AlltoallNCCL3) {
  // Alltoall with expand 16
  int expand = 16;
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto gpu_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto cpu_options = torch::TensorOptions().dtype(torch::kInt64);
  // para
  int k = 7, b = 100;
  std::vector<IdType> send_vecs(world_size), recv_vecs(world_size);
  for (int r = 0; r < world_size; ++r) {
    send_vecs[r] = k * (rank - r) * (rank - r) + b;
    recv_vecs[r] = k * (r - rank) * (r - rank) + b;
  }
  auto send_size = torch::tensor(send_vecs, cpu_options);
  auto recv_size = torch::tensor(recv_vecs, cpu_options);
  auto send_offset = send_size.cumsum(0);
  auto recv_offset = recv_size.cumsum(0);
  auto num_send = send_offset[-1].item<IdType>();
  auto num_recv = recv_offset[-1].item<IdType>();
  auto send_val = torch::full(num_send * expand, rank, gpu_options);
  auto recv_val = torch::empty(num_recv * expand, gpu_options);
  std::vector<IdType> recv_val_vecs(num_recv * expand);
  int offset = 0;
  for (int r = 0; r < world_size; ++r) {
    for (int i = 0; i < recv_vecs[r] * expand; ++i) {
      recv_val_vecs[offset++] = r;
    }
  }
  auto check_val = torch::tensor(recv_val_vecs, gpu_options);

  auto arange = torch::arange(1, world_size + 1);
  npc::AlltoAll(send_val, recv_val, send_offset, recv_offset, expand);
  EXPECT_TRUE(recv_val.equal(check_val));
}
