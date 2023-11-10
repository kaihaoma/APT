#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include "spmm.h"
#include "utils.h"

TEST(TestSpMM, CopyUSum1) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto id_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto data_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rank);
  auto input = torch::arange(6, data_options).view({6, 1});
  auto coo_row = torch::tensor({0, 1, 2, 2, 1, 0}, id_options);
  auto coo_col = torch::tensor({0, 1, 1, 0, 0, 1}, id_options);
  auto coo_offset = torch::tensor({0, 3, 6}, id_options);
  auto input_offset = torch::tensor({0, 3, 6}, id_options);
  auto output_offset = torch::tensor({0, 2, 4}, id_options);
  auto result = torch::tensor({0.0, 3.0, 9.0, 3.0}, data_options).view({4, 1});
  auto output = npc::CopyUSum(
      coo_row, coo_col, input, coo_offset, input_offset, output_offset);
  EXPECT_TRUE(output.equal(result))
      << npc::TensorToString(output) << "!=" << npc::TensorToString(result);
}

TEST(TestSpMM, CopyUSum2) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  auto id_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, rank);
  auto data_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rank);
  auto input = torch::arange(18, data_options).view({6, 3});
  auto coo_row = torch::tensor({0, 1, 2, 2, 1, 0}, id_options);
  auto coo_col = torch::tensor({0, 1, 1, 0, 0, 1}, id_options);
  auto coo_offset = torch::tensor({0, 3, 6}, id_options);
  auto input_offset = torch::tensor({0, 3, 6}, id_options);
  auto output_offset = torch::tensor({0, 2, 4}, id_options);
  auto result =
      torch::tensor(
          {{0, 1, 2}, {9, 11, 13}, {27, 29, 31}, {9, 10, 11}}, data_options)
          .view({4, 3});
  auto output = npc::CopyUSum(
      coo_row, coo_col, input, coo_offset, input_offset, output_offset);
  EXPECT_TRUE(output.equal(result))
      << npc::TensorToString(output) << "!=" << npc::TensorToString(result);
}