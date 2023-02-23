#include "./collective.h"

#include <cuda_runtime.h>
#include <state.h>

#include "../utils.h"

namespace npc {

void _Allreduce(void* input, void* output, IdType len) {
  auto* state = NPCState::Global();
  NCCLCHECK(ncclAllReduce(
      input, output, len, ncclInt64, ncclSum, state->nccl_comm, 0));
  CUDACHECK(cudaStreamSynchronize(0));
}

torch::Tensor Allreduce(torch::Tensor t) {
  auto ret = torch::empty_like(t);
  IdType len = t.numel();
  _Allreduce(t.data_ptr(), ret.data_ptr(), len);
  return ret;
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _AlltoAll(
    T* input, T* output, IdType* send_offset, IdType* recv_offset, int expand) {
  CUDACHECK(cudaStreamSynchronize(0));
  auto* state = NPCState::Global();
  int world_size = state->world_size;
  IdType send_off = 0;
  IdType recv_off = 0;
  ncclGroupStart();
  for (int r = 0; r < world_size; ++r) {
    IdType send_size = send_offset[r] - send_off;
    IdType recv_size = recv_offset[r] - recv_off;

    NCCLCHECK(ncclSend(
        input + send_off * expand, send_size * expand, NCCL_DATA_TYPE, r,
        state->nccl_comm, 0));
    NCCLCHECK(ncclRecv(
        output + recv_off * expand, recv_size * expand, NCCL_DATA_TYPE, r,
        state->nccl_comm, 0));

    send_off = send_offset[r];
    recv_off = recv_offset[r];
  }
  ncclGroupEnd();
  CUDACHECK(cudaStreamSynchronize(0));
}

void AlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_offset,
    torch::Tensor recv_offset, int expand) {
  if (input.dtype() == torch::kLong) {
    _AlltoAll<IdType, ncclInt64>(
        input.data_ptr<IdType>(), output.data_ptr<IdType>(),
        send_offset.data_ptr<IdType>(), recv_offset.data_ptr<IdType>(), expand);
  } else if (input.dtype() == torch::kFloat32) {
    _AlltoAll<DataType, ncclFloat32>(
        input.data_ptr<DataType>(), output.data_ptr<DataType>(),
        send_offset.data_ptr<IdType>(), recv_offset.data_ptr<IdType>(), expand);
  }
}

}  // namespace npc