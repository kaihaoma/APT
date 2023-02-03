#include "./collective.h"

#include <cuda_runtime.h>
#include <state.h>

#include "../utils.h"

namespace npc {

void _Allreduce(void* input, void* output, int64_t len) {
  auto* state = NPCState::Global();
  NCCLCHECK(ncclAllReduce(
      input, output, len, ncclInt64, ncclSum, state->nccl_comm, 0));
  CUDACHECK(cudaStreamSynchronize(0));
}

torch::Tensor Allreduce(torch::Tensor t) {
  auto ret = torch::empty_like(t);
  int64_t len = t.numel();
  _Allreduce(t.data_ptr(), ret.data_ptr(), len);
  return ret;
}

}  // namespace npc