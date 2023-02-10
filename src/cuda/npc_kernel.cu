#include "../utils.h"
#include "./npc_kernel.h"

namespace npc {

// Load node feature from dev&uva in on time
// 1. classify feat type 0: on dev 1: on uva
__global__ void _CSRRowWiseLoadSubtensorAlignedAllKernel(
    IdType size, IdType dim, IdType *index, IdType *feat_pos_map,
    DataType *input_table_dev, DataType *input_table_uva,
    DataType *output_table) {
  IdType out_row = blockIdx.x * blockDim.y + threadIdx.y;
  while (out_row < size) {
    const IdType out_row_start = out_row * dim;
    IdType feat_type = -1;
    IdType origin_in_row_start = -1;

    if (feat_pos_map[index[out_row]] < -1) {
      // on uva
      feat_type = FEAT_ON_UVA;
      // origin_in_row_start = (-(feat_pos_map[index[out_row]]) - 2) * dim;
      origin_in_row_start = ENCODE_ID(feat_pos_map[index[out_row]]) * dim;
    } else {
      // on dev
      feat_type = FEAT_ON_DEV;
      origin_in_row_start = feat_pos_map[index[out_row]] * dim;
    }
    const IdType in_row_start = origin_in_row_start & ~0x1F;
    const IdType in_row_end = origin_in_row_start + dim;
    IdType idx = threadIdx.x + in_row_start;
    while (idx < in_row_end) {
      if (idx >= origin_in_row_start) {
        if (feat_type == FEAT_ON_UVA) {
          output_table[out_row_start + (idx - origin_in_row_start)] =
              input_table_uva[idx];
        } else {
          output_table[out_row_start + (idx - origin_in_row_start)] =
              input_table_dev[idx];
        }
      }
      idx += blockDim.x;
    }
    out_row += gridDim.x * blockDim.y;
  }
}

void IndexSelectAll(
    IdType size, IdType feat_dim, torch::Tensor index,
    torch::Tensor feat_pos_map, torch::Tensor input_table_dev,
    torch::Tensor input_table_uva, torch::Tensor output_table,
    cudaStream_t stream) {
  constexpr int BLOCK_X = 128;
  constexpr int BLOCK_ROWS = 4;
  constexpr int TILE_SIZE = BLOCK_ROWS * 16;
  dim3 block(BLOCK_X, BLOCK_ROWS);
  int BLOCK_NUM = (size + TILE_SIZE - 1) / TILE_SIZE;
  dim3 grid(BLOCK_NUM);
  _CSRRowWiseLoadSubtensorAlignedAllKernel<<<grid, block, 0, stream>>>(
      size, feat_dim, index.data_ptr<IdType>(), feat_pos_map.data_ptr<IdType>(),
      input_table_dev.data_ptr<DataType>(),
      input_table_uva.data_ptr<DataType>(), output_table.data_ptr<DataType>());
}

}  // namespace npc