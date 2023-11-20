#include "./collective.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <state.h>

#include "../utils.h"

namespace npc {
template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _Allgather(T* input, T* output, IdType size, IdType comm_type) {
  auto* state = NPCState::Global();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  NCCLCHECK(
      ncclAllGather(input, output, size, NCCL_DATA_TYPE, nccl_comm, stream));
}

torch::Tensor AllGather(torch::Tensor input, IdType comm_type) {
  auto* state = NPCState::Global();
  int rank = state->rank;
  int world_size = state->world_size;
  auto input_size = input.numel();

  auto ret = torch::empty(world_size * input_size, input.options());
  if (input.dtype() == torch::kLong) {
    _Allgather<IdType, ncclInt64>(
        input.data_ptr<IdType>(), ret.data_ptr<IdType>(), input_size,
        comm_type);
  } else if (input.dtype() == torch::kFloat32) {
    _Allgather<DataType, ncclFloat32>(
        input.data_ptr<DataType>(), ret.data_ptr<DataType>(), input_size,
        comm_type);
  }
  return ret;
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _SPAlltoAll(
    T* input, T* output, IdType* send_sizes, IdType* recv_sizes, IdType expand,
    IdType comm_type) {
  auto* state = NPCState::Global();
  int world_size = state->world_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  ncclGroupStart();
  IdType send_src_off = 0;
  IdType recv_src_off = 0;
  for (int r = 0; r < world_size; ++r) {
    // send_src
    IdType send_src_size = send_sizes[2 * r];
    IdType recv_src_size = recv_sizes[2 * r];
    NCCLCHECK(ncclSend(
        input + send_src_off * expand, send_src_size * expand, NCCL_DATA_TYPE,
        r, nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_src_off * expand, recv_src_size * expand, NCCL_DATA_TYPE,
        r, nccl_comm, stream));

    send_src_off += send_src_size;
    recv_src_off += recv_src_size;
  }
  ncclGroupEnd();
  cudaStreamSynchronize(stream);
}

// Split Para. custom all-to-all
void SPSampleAlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_sizes,
    torch::Tensor recv_sizes, IdType expand, IdType comm_type) {
  _SPAlltoAll<IdType, ncclInt64>(
      input.data_ptr<IdType>(), output.data_ptr<IdType>(),
      send_sizes.data_ptr<IdType>(), recv_sizes.data_ptr<IdType>(), expand,
      comm_type);
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
IdType _SPAlltoAllWithDst(
    T* input, T* output, IdType* send_sizes, IdType* recv_sizes, IdType expand,
    IdType comm_type) {
  auto* state = NPCState::Global();
  int world_size = state->world_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  IdType send_dst_off = 0;
  IdType recv_dst_off = 0;
  ncclGroupStart();
  for (int r = 0; r < world_size; ++r) {
    // send_dst
    IdType send_dst_size = send_sizes[3 * r];
    IdType recv_dst_size = recv_sizes[3 * r];
    NCCLCHECK(ncclSend(
        input + send_dst_off * expand, send_dst_size * expand, NCCL_DATA_TYPE,
        r, nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_dst_off * expand, recv_dst_size * expand, NCCL_DATA_TYPE,
        r, nccl_comm, stream));
    send_dst_off += send_dst_size;
    recv_dst_off += recv_dst_size;
  }
  IdType send_src_off = send_dst_off;
  IdType recv_src_off = recv_dst_off;
  for (int r = 0; r < world_size; ++r) {
    // send_src
    IdType send_src_size = send_sizes[3 * r + 1];
    IdType recv_src_size = recv_sizes[3 * r + 1];
    NCCLCHECK(ncclSend(
        input + send_src_off * expand, send_src_size * expand, NCCL_DATA_TYPE,
        r, nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_src_off * expand, recv_src_size * expand, NCCL_DATA_TYPE,
        r, nccl_comm, stream));

    send_src_off += send_src_size;
    recv_src_off += recv_src_size;
  }
  ncclGroupEnd();
  return recv_dst_off;
}

// Split Para. custom all-to-all
IdType SPSampleAlltoAllWithDst(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_sizes,
    torch::Tensor recv_sizes, IdType expand, IdType comm_type) {
  return _SPAlltoAllWithDst<IdType, ncclInt64>(
      input.data_ptr<IdType>(), output.data_ptr<IdType>(),
      send_sizes.data_ptr<IdType>(), recv_sizes.data_ptr<IdType>(), expand,
      comm_type);
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _SPFeatureAlltoAll(
    T* input, T* output, IdType* send_sizes, IdType* recv_sizes, IdType expand,
    IdType comm_type) {
  auto* state = NPCState::Global();
  int world_size = state->world_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  IdType send_off = 0;
  IdType recv_off = 0;
  ncclGroupStart();
  // all-to-all src
  for (int r = 0; r < world_size; ++r) {
    IdType send_size = send_sizes[2 * r + 1];
    IdType recv_size = recv_sizes[2 * r + 1];
    NCCLCHECK(ncclSend(
        input + send_off * expand, send_size * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_off * expand, recv_size * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));

    send_off += send_size;
    recv_off += recv_size;
  }
  ncclGroupEnd();
  // CUDACHECK(cudaStreamSynchronize(stream));
}

void SPFeatureAlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_sizes,
    torch::Tensor recv_sizes, IdType expand, IdType comm_type) {
  _SPFeatureAlltoAll<DataType, ncclFloat32>(
      input.data_ptr<DataType>(), output.data_ptr<DataType>(),
      send_sizes.data_ptr<IdType>(), recv_sizes.data_ptr<IdType>(), expand,
      comm_type);
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _SPFeatureAlltoAllWithDst(
    T* input, T* output, IdType* send_sizes, IdType* recv_sizes, IdType expand,
    IdType comm_type) {
  auto* state = NPCState::Global();
  int world_size = state->world_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  IdType send_off = 0;
  IdType recv_off = 0;
  ncclGroupStart();
  // all-to-all dst
  for (int r = 0; r < world_size; ++r) {
    IdType send_size = send_sizes[3 * r];
    IdType recv_size = recv_sizes[3 * r];

    NCCLCHECK(ncclSend(
        input + send_off * expand, send_size * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_off * expand, recv_size * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));

    send_off += send_size;
    recv_off += recv_size;
  }
  // all-to-all src
  for (int r = 0; r < world_size; ++r) {
    IdType send_size = send_sizes[3 * r + 2];
    IdType recv_size = recv_sizes[3 * r + 2];

    NCCLCHECK(ncclSend(
        input + send_off * expand, send_size * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_off * expand, recv_size * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));

    send_off += send_size;
    recv_off += recv_size;
  }
  ncclGroupEnd();
  // CUDACHECK(cudaStreamSynchronize(stream));
}

void SPFeatureAlltoAllWithDst(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_sizes,
    torch::Tensor recv_sizes, IdType expand, IdType comm_type) {
  _SPFeatureAlltoAllWithDst<DataType, ncclFloat32>(
      input.data_ptr<DataType>(), output.data_ptr<DataType>(),
      send_sizes.data_ptr<IdType>(), recv_sizes.data_ptr<IdType>(), expand,
      comm_type);
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _AlltoAll(
    T* input, T* output, IdType* send_offset, IdType* recv_offset,
    IdType expand, IdType comm_type) {
  auto* state = NPCState::Global();
  int world_size = state->world_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  IdType send_off = 0;
  IdType recv_off = 0;
  ncclGroupStart();
  for (int r = 0; r < world_size; ++r) {
    IdType send_size = send_offset[r] - send_off;
    IdType recv_size = recv_offset[r] - recv_off;

    NCCLCHECK(ncclSend(
        input + send_off * expand, send_size * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_off * expand, recv_size * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));

    send_off = send_offset[r];
    recv_off = recv_offset[r];
  }
  ncclGroupEnd();
  // CUDACHECK(cudaStreamSynchronize(stream));
}

void AlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_offset,
    torch::Tensor recv_offset, IdType expand, IdType comm_type) {
  if (input.dtype() == torch::kLong) {
    _AlltoAll<IdType, ncclInt64>(
        input.data_ptr<IdType>(), output.data_ptr<IdType>(),
        send_offset.data_ptr<IdType>(), recv_offset.data_ptr<IdType>(), expand,
        comm_type);
  } else if (input.dtype() == torch::kFloat32) {
    _AlltoAll<DataType, ncclFloat32>(
        input.data_ptr<DataType>(), output.data_ptr<DataType>(),
        send_offset.data_ptr<IdType>(), recv_offset.data_ptr<IdType>(), expand,
        comm_type);
  }
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _CrossMachineAlltoAll(
    T* input, T* output, IdType* send_size, IdType* recv_size, IdType expand,
    IdType comm_type) {
  auto* state = NPCState::Global();
  int rank = state->rank;
  int world_size = state->world_size;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  // auto remote_worker_map = state->vec_remote_worker_map;
  auto num_remote_workers = state->num_remote_workers;
  IdType send_off = send_size[0];
  IdType recv_off = recv_size[0];

  ncclGroupStart();
  for (int r = 1; r < num_remote_workers; ++r) {
    NCCLCHECK(ncclSend(
        input + send_off * expand, send_size[r] * expand, NCCL_DATA_TYPE,
        state->vec_remote_worker_id[r], nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_off * expand, recv_size[r] * expand, NCCL_DATA_TYPE,
        state->vec_remote_worker_id[r], nccl_comm, stream));
    send_off += send_size[r];
    recv_off += recv_size[r];
  }
  ncclGroupEnd();
  // cudamemcpy on send_size[0]
  cudaMemcpyAsync(
      output, input, send_size[0] * expand * sizeof(T),
      cudaMemcpyDeviceToDevice, stream);
  // CUDACHECK(cudaStreamSynchronize(stream));
}

void CrossMachineAlltoAll(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_size,
    torch::Tensor recv_size, IdType expand, IdType comm_type) {
  if (input.dtype() == torch::kLong) {
    _CrossMachineAlltoAll<IdType, ncclInt64>(
        input.data_ptr<IdType>(), output.data_ptr<IdType>(),
        send_size.data_ptr<IdType>(), recv_size.data_ptr<IdType>(), expand,
        comm_type);

  } else if (input.dtype() == torch::kFloat32) {
    _CrossMachineAlltoAll<DataType, ncclFloat32>(
        input.data_ptr<DataType>(), output.data_ptr<DataType>(),
        send_size.data_ptr<IdType>(), recv_size.data_ptr<IdType>(), expand,
        comm_type);
  }
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _AllBroadcast(
    T* input, T* output, IdType send_size, IdType* recv_size, IdType expand,
    IdType comm_type) {
  auto* state = NPCState::Global();
  int rank = state->rank;
  int world_size = state->world_size;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  IdType recv_off = 0;
  ncclGroupStart();
  for (int r = 0; r < world_size; ++r) {
    NCCLCHECK(ncclSend(
        input, send_size * expand, NCCL_DATA_TYPE, r, nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_off * expand, recv_size[r] * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));
    recv_off += recv_size[r];
  }
  ncclGroupEnd();
  // CUDACHECK(cudaStreamSynchronize(stream));
}

void AllBroadcast(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_size,
    torch::Tensor recv_size, IdType expand, IdType comm_type) {
  if (input.dtype() == torch::kLong) {
    _AllBroadcast<IdType, ncclInt64>(
        input.data_ptr<IdType>(), output.data_ptr<IdType>(),
        send_size.item<IdType>(), recv_size.data_ptr<IdType>(), expand,
        comm_type);
  } else if (input.dtype() == torch::kFloat32) {
    _AllBroadcast<DataType, ncclFloat32>(
        input.data_ptr<DataType>(), output.data_ptr<DataType>(),
        send_size.item<IdType>(), recv_size.data_ptr<IdType>(), expand,
        comm_type);
  }
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _AllBroadcastV2(
    T* input, T* output, IdType send_size, IdType* recv_size, IdType expand,
    IdType comm_type) {
  auto* state = NPCState::Global();
  int rank = state->rank;
  int world_size = state->world_size;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  IdType send_off = 0;
  IdType recv_off = 0;
  ncclGroupStart();
  for (int r = 0; r < world_size; ++r) {
    NCCLCHECK(ncclSend(
        input + send_off * expand, send_size * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));
    NCCLCHECK(ncclRecv(
        output + recv_off * expand, recv_size[r] * expand, NCCL_DATA_TYPE, r,
        nccl_comm, stream));
    recv_off += recv_size[r];
    send_off += send_size;
  }
  ncclGroupEnd();
  // CUDACHECK(cudaStreamSynchronize(stream));
}

void AllBroadcastV2(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_size,
    torch::Tensor recv_size, IdType expand, IdType comm_type) {
  auto* state = NPCState::Global();
  int rank = state->rank;
  int world_size = state->world_size;
  auto repeat_input = input.repeat({world_size});

  if (input.dtype() == torch::kLong) {
    _AllBroadcastV2<IdType, ncclInt64>(
        repeat_input.data_ptr<IdType>(), output.data_ptr<IdType>(),
        send_size.item<IdType>(), recv_size.data_ptr<IdType>(), expand,
        comm_type);
  } else if (input.dtype() == torch::kFloat32) {
    _AllBroadcastV2<DataType, ncclFloat32>(
        repeat_input.data_ptr<DataType>(), output.data_ptr<DataType>(),
        send_size.item<IdType>(), recv_size.data_ptr<IdType>(), expand,
        comm_type);
  }
}

template <typename T, ncclDataType_t NCCL_DATA_TYPE>
void _AllReduce(
    T* input, T* output, IdType* send_size, IdType expand, IdType comm_type) {
  auto* state = NPCState::Global();
  int rank = state->rank;
  int world_size = state->world_size;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto nccl_comm = state->nccl_comm_list[comm_type];
  ncclGroupStart();
  IdType send_offset = 0;
  for (int r = 0; r < world_size; ++r) {
    NCCLCHECK(ncclReduce(
        input + send_offset * expand, output, send_size[r] * expand,
        NCCL_DATA_TYPE, ncclSum, r, nccl_comm, stream));
    send_offset += send_size[r];
  }
  ncclGroupEnd();
  // CUDACHECK(cudaStreamSynchronize(stream));
}

void AllReduce(
    torch::Tensor input, torch::Tensor output, torch::Tensor send_size,
    IdType expand, IdType comm_type) {
  if (input.dtype() == torch::kLong) {
    _AllReduce<IdType, ncclInt64>(
        input.data_ptr<IdType>(), output.data_ptr<IdType>(),
        send_size.data_ptr<IdType>(), expand, comm_type);
  } else if (input.dtype() == torch::kFloat32) {
    _AllReduce<DataType, ncclFloat32>(
        input.data_ptr<DataType>(), output.data_ptr<DataType>(),
        send_size.data_ptr<IdType>(), expand, comm_type);
  }
}

}  // namespace npc