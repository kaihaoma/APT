#include "./atomic.h"
#include "./cuda_common.h"
#include "./npc_kernel.h"
#include "./utils.h"

namespace npc {
template <typename Idx, typename DType>
__global__ void CopyUSumKernel(
    const DType* __restrict__ infeat, DType* __restrict__ outfeat,
    const Idx* __restrict__ row, const Idx* __restrict__ col, int64_t E,
    const int64_t* __restrict__ coo_off, const int64_t* __restrict__ in_off,
    const int64_t* __restrict__ out_off, int64_t feat_dim,
    int64_t coo_off_len) {
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = row[ty];
    const Idx dst = col[ty];
    int rank = cub::UpperBound(coo_off, coo_off_len, ty) - 1;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* inoff = infeat + (in_off[rank] + src) * feat_dim;
    DType* outoff = outfeat + (out_off[rank] + dst) * feat_dim;
    while (tx < feat_dim) {
      AtomicAdd(outoff + tx, inoff[tx]);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType>
__global__ void CopyESumKernel(
    const DType* __restrict__ infeat, DType* __restrict__ outfeat,
    const Idx* __restrict__ row, const Idx* __restrict__ col, int64_t E,
    const int64_t* __restrict__ coo_off, const int64_t* __restrict__ out_off,
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
    const DType* inoff = infeat + ty * feat_dim;
    DType* outoff = outfeat + (out_off[rank] + dst) * feat_dim;
    while (tx < feat_dim) {
      AtomicAdd(outoff + tx, inoff[tx]);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType>
__global__ void UMulESumKernel(
    const DType* __restrict__ infeat, const DType* __restrict__ efeat,
    DType* __restrict__ outfeat, const Idx* __restrict__ row,
    const Idx* __restrict__ col, int64_t E, const int64_t* __restrict__ coo_off,
    const int64_t* __restrict__ in_off, const int64_t* __restrict__ out_off,
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
    const DType* inoff = infeat + (in_off[rank] + src) * feat_dim;
    const DType* eoff = efeat + ty * feat_dim;
    DType* outoff = outfeat + (out_off[rank] + dst) * feat_dim;
    while (tx < feat_dim) {
      AtomicAdd(outoff + tx, (inoff[tx] + eoff[tx]));
      tx += stride_x;
    }
    ty += stride_y;
  }
}

void CopyUSumCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor output, torch::Tensor coo_offset, torch::Tensor input_offset,
    torch::Tensor output_offset) {
  int64_t E = coo_row.numel();
  int64_t len = input.size(1);

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL(
      (CopyUSumKernel<int64_t, float>), nblks, nthrs, input.data_ptr<float>(),
      output.data_ptr<float>(), coo_row.data_ptr<int64_t>(),
      coo_col.data_ptr<int64_t>(), E, coo_offset.data_ptr<int64_t>(),
      input_offset.data_ptr<int64_t>(), output_offset.data_ptr<int64_t>(), len,
      coo_offset.numel());
}

void CopyESumCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor output, torch::Tensor coo_offset,
    torch::Tensor output_offset) {
  int64_t E = coo_row.numel();
  int64_t len = input.size(1);

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL(
      (CopyESumKernel<int64_t, float>), nblks, nthrs, input.data_ptr<float>(),
      output.data_ptr<float>(), coo_row.data_ptr<int64_t>(),
      coo_col.data_ptr<int64_t>(), E, coo_offset.data_ptr<int64_t>(),
      output_offset.data_ptr<int64_t>(), len, coo_offset.numel());
}

void UMulESumCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor input,
    torch::Tensor edata, torch::Tensor output, torch::Tensor coo_offset,
    torch::Tensor input_offset, torch::Tensor output_offset) {
  int64_t E = coo_row.numel();
  int64_t len = input.size(1);

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL(
      (UMulESumKernel<int64_t, float>), nblks, nthrs, input.data_ptr<float>(),
      edata.data_ptr<float>(), output.data_ptr<float>(),
      coo_row.data_ptr<int64_t>(), coo_col.data_ptr<int64_t>(), E,
      coo_offset.data_ptr<int64_t>(), input_offset.data_ptr<int64_t>(),
      output_offset.data_ptr<int64_t>(), len, coo_offset.numel());
}
}  // namespace npc