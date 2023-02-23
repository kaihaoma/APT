#ifndef NPC_UTILS_H_
#define NPC_UTILS_H_

#include <cuda_runtime.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>
#include <vector>

using IdType = int64_t;
using DataType = float;

#define ENCODE_ID(i) (-(i)-2)
const int FEAT_ON_UVA = 0;
const int FEAT_ON_DEV = 1;

#define CUDACHECK(cmd)                                      \
  do {                                                      \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
      LOG(FATAL) << "Cuda error " << cudaGetErrorString(e); \
    }                                                       \
  } while (false);

#define NCCLCHECK(cmd)                                           \
  do {                                                           \
    ncclResult_t res = cmd;                                      \
    if (res != ncclSuccess) {                                    \
      printf(                                                    \
          "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
          ncclGetErrorString(res));                              \
      exit(EXIT_FAILURE);                                        \
    }                                                            \
  } while (0)

namespace npc {

template <typename T>
std::string VecToString(const std::vector<T> &vec) {
  std::string ret = "[";
  for (int i = 0; i < vec.size(); ++i) {
    if (i > 0) ret += ", ";
    ret += std::to_string(vec[i]);
  }
  ret += "]";
  return ret;
}

inline std::string TensorToString(torch::Tensor t) {
  std::vector<IdType> vec_tensor(
      t.data_ptr<IdType>(), t.data_ptr<IdType>() + t.numel());
  return VecToString(vec_tensor);
}

template <typename T>
std::string ArrToString(const T *ptr, int num_elements) {
  std::vector<T> vec(ptr, ptr + num_elements);
  return VecToString(vec);
}

template <typename T>
std::string DevArrToString(const T *ptr, int num_elements) {
  std::vector<T> vec(num_elements);
  CUDACHECK(cudaMemcpy(
      vec.data(), ptr, sizeof(T) * num_elements, cudaMemcpyDeviceToHost));
  return VecToString(vec);
}

template <typename T>
void NPCCudaMalloc(T **ptr, int size) {
  CUDACHECK(cudaMalloc(ptr, sizeof(T) * size));
  CUDACHECK(cudaMemset(*ptr, 0, sizeof(T) * size));
}

template <typename T>
void NPCCudaMallocAndCopy(T **ret, const T *src, int size) {
  NPCCudaMalloc(ret, size);
  CUDACHECK(cudaMemcpy(*ret, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template <typename T>
void NPCCudaMallocAndCopy(T **ret, const std::vector<T> &src) {
  NPCCudaMallocAndCopy(ret, src.data(), src.size());
}

template <typename T>
static inline void NPCCudaHostAlloc(T **ptr, size_t size) {
  CUDACHECK(cudaHostAlloc(ptr, sizeof(T) * size, cudaHostAllocMapped));
  memset(*ptr, 0, size);
}

template <typename T>
void NPCHostMallocAndCopy(T **ret, const std::vector<T> &src) {
  NPCCudaHostAlloc(ret, src.size());
  memcpy(*ret, src.data(), sizeof(T) * src.size());
}

}  // namespace npc
#endif