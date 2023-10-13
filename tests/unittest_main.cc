#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>
#include <torch/torch.h>

#include "core.h"
#include "glog/logging.h"
#include "utils.h"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  MPI_Status status;
  torch::Tensor nccl_id;
  int64_t nccl_id_len = NCCL_UNIQUE_ID_BYTES / 8;
  MPI_Init(&argc, &argv);
  LOG(INFO) << "MPI test is enabled argc: " << argc << " argv: " << argv[0];

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  LOG(INFO) << rank << " : " << world_size;

  setenv("MASTER_ADDR", "localhost", 1);
  setenv("MASTER_PORT", "29500", 1);
  cudaSetDevice(rank);
  std::vector<torch::Tensor> nccl_id_list;
  for (int i = 0; i < 2; ++i) {
    if (rank == 0) {
      nccl_id = npc::NCCLGetUniqueId();
      for (int r = 1; r < world_size; r++) {
        MPI_Send(
            nccl_id.data_ptr(), nccl_id_len, MPI_LONG_LONG, r, 10,
            MPI_COMM_WORLD);
      }
    } else {
      nccl_id = torch::empty(
          nccl_id_len, torch::dtype(torch::kInt64).device(torch::kCPU));
      MPI_Recv(
          nccl_id.data_ptr(), nccl_id_len, MPI_LONG_LONG, 0, 10, MPI_COMM_WORLD,
          &status);
    }
    nccl_id_list.emplace_back(nccl_id);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  auto nccl_id_list_tensor = torch::vstack(nccl_id_list);
  npc::Initialize(rank, rank, world_size, nccl_id_list_tensor);

  int result = RUN_ALL_TESTS();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return result;
}
