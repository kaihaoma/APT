#include "./core.h"

#include <nccl.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>

#include "./state.h"
#include "./utils.h"
#include "glog/logging.h"

namespace npc {

std::string NCCLIdToString(ncclUniqueId id) {
  return std::string(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

ncclUniqueId StringToNCCLId(const std::string& str) {
  ncclUniqueId ret;
  memcpy(ret.internal, str.data(), NCCL_UNIQUE_ID_BYTES);
  return ret;
}

torch::Tensor NCCLIdToTensor(ncclUniqueId id) {
  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  torch::Tensor t = torch::empty(NCCL_UNIQUE_ID_BYTES / 8, options);
  memcpy(t.data_ptr(), id.internal, NCCL_UNIQUE_ID_BYTES);
  return t;
}

ncclUniqueId TensorToNCCLId(torch::Tensor t) {
  ncclUniqueId ret;
  memcpy(ret.internal, t.data_ptr(), NCCL_UNIQUE_ID_BYTES);
  return ret;
}

torch::Tensor NCCLGetUniqueId() {
  ncclUniqueId nccl_id;
  ncclGetUniqueId(&nccl_id);
  return NCCLIdToTensor(nccl_id);
}

ncclComm_t BuildNCCLComm(
    int64_t rank, int64_t world_size, torch::Tensor nccl_id_tensor) {
  auto nccl_id = TensorToNCCLId(nccl_id_tensor);
  ncclComm_t ret;
  NCCLCHECK(ncclCommInitRank(&ret, world_size, nccl_id, rank));
  return ret;
}

void Initialize(
    int64_t rank, int64_t world_size, torch::Tensor nccl_id_tensor) {
  auto* state = NPCState::Global();
  state->rank = rank;
  state->world_size = world_size;
  state->nccl_comm = BuildNCCLComm(rank, world_size, nccl_id_tensor);
  CUDACHECK(cudaSetDevice(rank));
}

void Test(torch::Tensor test_tensor, int64_t idx, double val) {
  LOG(INFO) << "Hello from glog";
  LOG(INFO) << "Tensor to Vector: " << TensorToString(test_tensor)
            << "\t idx: " << idx << "\t val: " << val;
}
}  // namespace npc