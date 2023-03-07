#include <curand_kernel.h>

#include "../state.h"
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
  CUDACHECK(cudaGetLastError());
  _CSRRowWiseLoadSubtensorAlignedAllKernel<<<grid, block, 0, stream>>>(
      size, feat_dim, index.data_ptr<IdType>(), feat_pos_map.data_ptr<IdType>(),
      input_table_dev.data_ptr<DataType>(),
      input_table_uva.data_ptr<DataType>(), output_table.data_ptr<DataType>());
  CUDACHECK(cudaStreamSynchronize(stream));
}

template <int BLOCK_WARPS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleReplaceKernel(
    const uint64_t random_seed, int fanout, IdType num_frontier,
    IdType *frontier, IdType *dev_indptr, IdType *dev_indices,
    IdType *adj_pos_map, IdType *out_indices) {
  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<IdType>(blockIdx.x + 1) * TILE_SIZE, num_frontier);

  curandState rng;
  curand_init(
      random_seed * gridDim.x + blockIdx.x,
      threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);
  while (out_row < last_row) {
    const IdType row = frontier[out_row];
    IdType pos = adj_pos_map[row];
    const int64_t in_row_start = dev_indptr[pos];
    const int64_t out_row_start = out_row * fanout;
    const int64_t deg = dev_indptr[pos + 1] - in_row_start;
    const int64_t *index = dev_indices + in_row_start;
    if (deg > 0) {
      for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_indices[out_idx] = index[edge];
      }
    }
    out_row += BLOCK_WARPS;
  }
}

torch::Tensor LocalSampleNeighbors(
    torch::Tensor frontier, int fanout, bool use_local_nid,
    cudaStream_t stream) {
  auto *state = NPCState::Global();
  auto num_frontier = frontier.numel();

  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  dim3 block(WARP_SIZE, BLOCK_WARPS);
  dim3 grid((num_frontier + TILE_SIZE - 1) / TILE_SIZE);
  const uint64_t random_seed = 7777777;

  auto adj_pos_map = state->graph_storage.adj_pos_map;
  auto dev_indptr = state->graph_storage.dev_indptr;

  auto dev_indices = use_local_nid ? state->graph_storage.dev_local_indices
                                   : state->graph_storage.dev_global_indices;
  auto num_graph_nodes = state->graph_storage.num_graph_nodes;

  auto options = frontier.options();
  auto out_indices = torch::empty({num_frontier * fanout}, options);

  _CSRRowWiseSampleReplaceKernel<BLOCK_WARPS, TILE_SIZE>
      <<<grid, block, 0, stream>>>(
          random_seed, fanout, num_frontier, frontier.data_ptr<IdType>(),
          dev_indptr.data_ptr<IdType>(), dev_indices.data_ptr<IdType>(),
          adj_pos_map.data_ptr<IdType>(), out_indices.data_ptr<IdType>());

  CUDACHECK(cudaStreamSynchronize(stream));
  return out_indices;
}

__global__ void _CountDeviceVerticesKernel(
    int world_size, IdType *min_vids, IdType num_seeds, IdType *seeds,
    IdType *bucket_size, IdType *belongs_to, IdType *inner_bucket_offset) {
  __shared__ IdType local_count[MAX_NUM_DEVICES];
  __shared__ IdType device_vid[MAX_NUM_DEVICES];
  IdType idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadIdx.x <= world_size) {
    device_vid[threadIdx.x] = min_vids[threadIdx.x];
    local_count[threadIdx.x] = 0;
  }

  __syncthreads();

  IdType device_id, vid;
  if (idx < num_seeds) {
    device_id = 0;
    vid = seeds[idx];
    while (device_id + 1 < world_size && device_vid[device_id + 1] <= vid) {
      ++device_id;
    }
    assert(false);
    belongs_to[idx] = device_id;
    // intra block offset
    inner_bucket_offset[idx] =
        atomicAdd((unsigned long long *)(local_count + device_id), 1);
  }

  __syncthreads();

  __shared__ IdType inter_block_offset[MAX_NUM_DEVICES];
  if (threadIdx.x < world_size) {
    inter_block_offset[threadIdx.x] = atomicAdd(
        (unsigned long long *)(bucket_size + threadIdx.x),
        local_count[threadIdx.x]);
  }
  __syncthreads();

  if (idx < num_seeds) {
    // inter block offset
    inner_bucket_offset[idx] += inter_block_offset[belongs_to[idx]];
  }
}

__global__ void _PermuteKernel(
    int world_size, IdType *min_vids, IdType num_seeds, IdType *seeds,
    IdType *bucket_offset, IdType *inner_bucket_offset, IdType *belongs_to,
    IdType *sorted_idx, IdType *permutation) {
  IdType idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < num_seeds) {
    auto rank = belongs_to[idx];
    auto offset =
        (rank == 0 ? 0 : bucket_offset[rank - 1]) + inner_bucket_offset[idx];
    // tranfer to local nid
    // sorted_idx[offset] = seeds[idx] - min_vids[rank];
    sorted_idx[offset] = seeds[idx];
    permutation[idx] = offset;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ClusterAndPermute(
    int rank, int world_size, torch::Tensor seeds, torch::Tensor min_vids,
    cudaStream_t stream) {
  IdType num_seeds = seeds.numel();

  auto dev_tensor_options = seeds.options();

  // return value
  auto bucket_size = torch::zeros({world_size}, dev_tensor_options);
  auto permutation = torch::empty({num_seeds}, dev_tensor_options);
  auto rev_permutation = torch::empty({num_seeds}, dev_tensor_options);
  auto belongs_to = torch::empty({num_seeds}, dev_tensor_options);
  auto sorted_idx = torch::empty({num_seeds}, dev_tensor_options);
  auto inner_bucket_offset = torch::empty({num_seeds}, dev_tensor_options);
  // 1.Count bucket size
  int num_blocks = (num_seeds + NUM_THREADS - 1) / NUM_THREADS;
  dim3 block(NUM_THREADS);
  dim3 grid(num_blocks);

  _CountDeviceVerticesKernel<<<grid, block, 0, stream>>>(
      world_size, min_vids.data_ptr<IdType>(), num_seeds,
      seeds.data_ptr<IdType>(), bucket_size.data_ptr<IdType>(),
      belongs_to.data_ptr<IdType>(), inner_bucket_offset.data_ptr<IdType>());
  CUDACHECK(cudaStreamSynchronize(stream));

  auto bucket_offset = bucket_size.cumsum(0);

  // 2.permutation
  _PermuteKernel<<<grid, block, 0, stream>>>(
      world_size, min_vids.data_ptr<IdType>(), num_seeds,
      seeds.data_ptr<IdType>(), bucket_offset.data_ptr<IdType>(),
      inner_bucket_offset.data_ptr<IdType>(), belongs_to.data_ptr<IdType>(),
      sorted_idx.data_ptr<IdType>(), permutation.data_ptr<IdType>());
  CUDACHECK(cudaStreamSynchronize(stream));

  return {bucket_size, bucket_offset, sorted_idx, permutation};
}

}  // namespace npc