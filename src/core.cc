#include "./core.h"

#include <c10/cuda/CUDAStream.h>
#include <nccl.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>

#include "./state.h"
// #include "./utils.h"
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
    IdType rank, IdType world_size, torch::Tensor nccl_id_tensor) {
  auto nccl_id = TensorToNCCLId(nccl_id_tensor);
  ncclComm_t ret;
  NCCLCHECK(ncclCommInitRank(&ret, world_size, nccl_id, rank));
  return ret;
}

void Initialize(
    IdType rank, IdType local_rank, IdType world_size,
    torch::Tensor nccl_id_tensor_list) {
  auto* state = NPCState::Global();
  state->rank = rank;
  state->local_rank = local_rank;
  state->world_size = world_size;
  auto nccl_id_sizes = nccl_id_tensor_list.sizes();
  state->num_threads = nccl_id_sizes[0];
  state->nccl_comm_list.resize(state->num_threads);
  LOG(INFO) << "#nccl comm: " << state->num_threads << "\n";
  for (IdType i = 0; i < state->num_threads; ++i) {
    state->nccl_comm_list[i] =
        BuildNCCLComm(rank, world_size, nccl_id_tensor_list[i]);
  }
  CHECK(state->num_threads >= 1 && state->num_threads);
  if (state->num_threads == 1) {
    state->sampler_id = 0;
    state->trainer_id = 0;
  } else if (state->num_threads == 2) {
    state->sampler_id = 0;
    state->trainer_id = 1;
  }
  CUDACHECK(cudaSetDevice(local_rank));
  // CUDACHECK(cudaStreamCreate(&state->nccl_stream));
  // CUDACHECK(cudaStreamCreate(&state->cuda_copy_stream));
  //  init sp_alltoall_size_permute
  std::vector<int> vec_sp_alltoall_size_permute(world_size * 3);
  for (int r = 0; r < world_size; ++r) {
    for (int i = 0; i < 3; ++i) {
      vec_sp_alltoall_size_permute[r * 3 + i] = i * world_size + r;
    }
  }
  state->sp_alltoall_size_permute = torch::tensor(vec_sp_alltoall_size_permute);
  LOG(INFO) << "sp_alltoall_size_permute: "
            << TensorToString(state->sp_alltoall_size_permute);
  state->cross_machine_flag = false;
}

void RegisterMinVids(torch::Tensor min_vids) {
  auto* state = NPCState::Global();
  state->min_vids = min_vids;
  LOG(INFO) << "Register min_vids: " << TensorToString(min_vids);
}

void RegisterMultiMachinesScheme(
    torch::Tensor remote_worker_map, torch::Tensor remote_worker_id) {
  auto* state = NPCState::Global();
  state->cross_machine_flag = true;
  state->remote_worker_map = remote_worker_map;
  std::vector<IdType> tmp_vec1(
      remote_worker_id.data_ptr<IdType>(),
      remote_worker_id.data_ptr<IdType>() + remote_worker_id.numel());
  state->vec_remote_worker_id = std::move(tmp_vec1);
  state->num_remote_workers = state->vec_remote_worker_id.size();

  auto cpu_remote_worker_map = remote_worker_map.to(torch::kCPU);
  std::vector<IdType> temp_vec2(
      cpu_remote_worker_map.data_ptr<IdType>(),
      cpu_remote_worker_map.data_ptr<IdType>() + cpu_remote_worker_map.numel());
  state->vec_remote_worker_map = std::move(temp_vec2);
  LOG(INFO) << "Register remote_worker_map: "
            << TensorToString(remote_worker_map) << "\t remote_worker_id: "
            << VecToString(state->vec_remote_worker_id)
            << "\t #vec_remote_worker_map: "
            << VecToString(state->vec_remote_worker_map);
}

}  // namespace npc