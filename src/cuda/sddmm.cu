#include "./atomic.h"
#include "./cuda_common.h"
#include "./npc_kernel.h"
#include "./utils.h"

namespace npc {
template <typename Idx, typename DType>
__global__ void UAddVKernel(
    const DType* __restrict__ lhs, const DType* __restrict__ rhs,
    DType* __restrict__ outfeat, const Idx* __restrict__ row,
    const Idx* __restrict__ col, int64_t E, const int64_t* __restrict__ coo_off,
    const int64_t* __restrict__ lhs_off, const int64_t* __restrict__ rhs_off,
    int64_t feat_dim, int64_t coo_off_len) {
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = row[ty];
    const Idx dst = col[ty];
    int rank = cub::UpperBound(coo_off, coo_off_len, ty) - 1;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* lhsoff = lhs + (lhs_off[rank] + src) * feat_dim;
    const DType* rhsoff = rhs + (rhs_off[rank] + dst) * feat_dim;
    DType* outoff = outfeat + ty * feat_dim;
    while (tx < feat_dim) {
      outoff[tx] = (lhsoff[tx] + rhsoff[tx]);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType>
__global__ void UMulVKernel(
    const DType* __restrict__ lhs, const DType* __restrict__ rhs,
    DType* __restrict__ outfeat, const Idx* __restrict__ row,
    const Idx* __restrict__ col, int64_t E, const int64_t* __restrict__ coo_off,
    const int64_t* __restrict__ lhs_off, const int64_t* __restrict__ rhs_off,
    int64_t feat_dim, int64_t coo_off_len) {
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = row[ty];
    const Idx dst = col[ty];
    int rank = cub::UpperBound(coo_off, coo_off_len, ty) - 1;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* lhsoff = lhs + (lhs_off[rank] + src) * feat_dim;
    const DType* rhsoff = rhs + (rhs_off[rank] + dst) * feat_dim;
    DType* outoff = outfeat + ty * feat_dim;
    while (tx < feat_dim) {
      outoff[tx] = (lhsoff[tx] * rhsoff[tx]);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

void UAddVCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor lhs,
    torch::Tensor rhs, torch::Tensor output, torch::Tensor coo_offset,
    torch::Tensor lhs_offset, torch::Tensor rhs_offset) {
  int64_t E = coo_row.numel();
  int64_t len = lhs.size(1);

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL(
      (UAddVKernel<int64_t, float>), nblks, nthrs, lhs.data_ptr<float>(),
      rhs.data_ptr<float>(), output.data_ptr<float>(),
      coo_row.data_ptr<int64_t>(), coo_col.data_ptr<int64_t>(), E,
      coo_offset.data_ptr<int64_t>(), lhs_offset.data_ptr<int64_t>(),
      rhs_offset.data_ptr<int64_t>(), len, coo_offset.numel());
}

void UMulVCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor lhs,
    torch::Tensor rhs, torch::Tensor output, torch::Tensor coo_offset,
    torch::Tensor lhs_offset, torch::Tensor rhs_offset) {
  int64_t E = coo_row.numel();
  int64_t len = lhs.size(1);

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL(
      (UMulVKernel<int64_t, float>), nblks, nthrs, lhs.data_ptr<float>(),
      rhs.data_ptr<float>(), output.data_ptr<float>(),
      coo_row.data_ptr<int64_t>(), coo_col.data_ptr<int64_t>(), E,
      coo_offset.data_ptr<int64_t>(), lhs_offset.data_ptr<int64_t>(),
      rhs_offset.data_ptr<int64_t>(), len, coo_offset.numel());
}
}  // namespace npc