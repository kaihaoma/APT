#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>
#include <torch/torch.h>

#include "core.h"
#include "glog/logging.h"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  MPI_Status status;
  torch::Tensor nccl_id;
  int64_t nccl_id_len = NCCL_UNIQUE_ID_BYTES / 8;
  MPI_Init(&argc, &argv);
  LOG(INFO) << "MPI test is enabled";

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  LOG(INFO) << rank << " : " << world_size;

  setenv("MASTER_ADDR", "localhost", 1);
  setenv("MASTER_PORT", "29500", 1);
  cudaSetDevice(rank);

  if (rank == 0) {
    nccl_id = npc::NCCLGetUniqueId();
    for (int i = 1; i < world_size; i++) {
      MPI_Send(
          nccl_id.data_ptr(), nccl_id_len, MPI_LONG_LONG, i, 10,
          MPI_COMM_WORLD);
    }
  } else {
    nccl_id = torch::empty(
        nccl_id_len, torch::dtype(torch::kInt64).device(torch::kCPU));
    MPI_Recv(
        nccl_id.data_ptr(), nccl_id_len, MPI_LONG_LONG, 0, 10, MPI_COMM_WORLD,
        &status);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  npc::Initialize(rank, world_size, nccl_id);

  int result = RUN_ALL_TESTS();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return result;
}
