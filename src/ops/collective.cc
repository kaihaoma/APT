#include "./collective.h"


#include <vector>
#include <state.h>
#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                      \
  do {                                                      \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
      LOG(FATAL) << "Cuda error " << cudaGetErrorString(e); \
    }                                                       \
  } while (false);

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

namespace npc {

void _Allreduce(void* input, void* output, int64_t len) {
  auto* state = NPCState::Global();
  NCCLCHECK(ncclAllReduce(input, output, len, ncclInt64, ncclSum, state->nccl_comm, 0));
  CUDACHECK(cudaStreamSynchronize(0));
}

torch::Tensor Allreduce(torch::Tensor t) {
  auto ret = torch::empty_like(t);
  int64_t len = t.numel();
  _Allreduce(t.data_ptr(), ret.data_ptr(), len);
  return ret;
}

}  // namespace npc