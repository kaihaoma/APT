#include <c10/cuda/CUDAStream.h>
#include <curand_kernel.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

#include "../state.h"
#include "../utils.h"
#include "./npc_kernel.h"

namespace npc {

// map src and dst to vir_nodes
//[is_src, device_id, dst_idx]
// base1: world_size * num_dst base2: num_dst
__global__ void _MapSrcAndDst(
    int world_size, int fanout, IdType num_dst, IdType num_src, IdType num_vir,
    IdType shuffle_id_offset, IdType *shuffle_min_vids, IdType *dst,
    IdType *src, IdType *vir_nodes, int rank, int node_beg, int node_end) {
  __shared__ IdType device_vid[MAX_NUM_DEVICES];
  IdType idx = blockDim.x * blockIdx.x + threadIdx.x;
  IdType stride = gridDim.x * blockDim.x;
  IdType vid, device_id = shuffle_id_offset, offset;
  if (threadIdx.x <= world_size) {
    device_vid[threadIdx.x] = shuffle_min_vids[threadIdx.x];
  }
  __syncthreads();
  // offset = [is_src, dst_idx]
  while (idx < num_vir) {
    if (idx < num_dst) {
      // map dst to vir_nodes
      vid = dst[idx];
      offset = idx;
    } else {
      // map src to vir_nodes
      IdType src_idx = idx - num_dst;
      vid = src[src_idx];
      offset = (world_size * num_dst) + int(src_idx / fanout);
    }

    while (device_id + 1 < world_size && device_vid[device_id + 1] <= vid) {
      ++device_id;
    }

    if (device_id < node_beg || device_id >= node_end) {
      device_id = rank;
    }

    vir_nodes[idx] = device_id * num_dst + offset;
    idx += stride;
  }
}

torch::Tensor MapSrcDsttoVir(
    IdType world_size, IdType fanout, torch::Tensor dst, torch::Tensor src,
    IdType shuffle_id_offset, torch::Tensor shuffle_min_vids, int rank, int node_beg, int node_end) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto num_dst = dst.numel();
  auto num_src = src.numel();
  auto num_vir = num_dst + num_src;
  auto vir_nodes = torch::empty({num_vir}, dst.options());
  dim3 block(NUM_THREADS);
  dim3 grid((num_vir + NUM_THREADS - 1) / NUM_THREADS);
  _MapSrcAndDst<<<grid, block, 0, stream>>>(
      world_size, fanout, num_dst, num_src, num_vir, shuffle_id_offset,
      shuffle_min_vids.data_ptr<IdType>(), dst.data_ptr<IdType>(),
      src.data_ptr<IdType>(), vir_nodes.data_ptr<IdType>(), rank, node_beg, node_end);
  CUDACHECK(cudaStreamSynchronize(stream));
  return vir_nodes;
}
// map src to vir_nodes
//[device_id, dst_idx]
// vir_nodes = device_id * num_dst + dst_idx
// dst_idx = int(idx / fanout)
__global__ void _MapSrctoVir(
    int wolrd_size, int fanout, IdType num_dst, IdType num_src,
    IdType shuffle_id_offset, IdType *shuffle_min_vids, IdType *src,
    IdType *vir_nodes, int rank, int node_beg, int node_end) {
  __shared__ IdType device_vid[MAX_NUM_DEVICES];
  IdType idx = blockDim.x * blockIdx.x + threadIdx.x;
  IdType stride = gridDim.x * blockDim.x;
  IdType vid, device_id;
  if (threadIdx.x <= wolrd_size) {
    device_vid[threadIdx.x] = shuffle_min_vids[threadIdx.x];
  }
  __syncthreads();

  while (idx < num_src) {
    vid = src[idx];
    device_id = shuffle_id_offset;
    while (device_id + 1 < wolrd_size && device_vid[device_id + 1] <= vid) {
      ++device_id;
    }

    if (device_id < node_beg || device_id >= node_end) {
      device_id = rank;
    }

    vir_nodes[idx] = device_id * num_dst + int(idx / fanout);
    idx += stride;
  }
}

torch::Tensor MapSrctoVir(
    IdType world_size, IdType fanout, IdType num_dst, torch::Tensor src,
    IdType shuffle_id_offset, torch::Tensor shuffle_min_vids, int rank, int node_beg, int node_end) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto num_src = src.numel();
  auto vir_nodes = torch::empty({num_src}, src.options());
  dim3 block(NUM_THREADS);
  dim3 grid((num_src + NUM_THREADS - 1) / NUM_THREADS);

  _MapSrctoVir<<<grid, block, 0, stream>>>(
      world_size, fanout, num_dst, num_src, shuffle_id_offset,
      shuffle_min_vids.data_ptr<IdType>(), src.data_ptr<IdType>(),
      vir_nodes.data_ptr<IdType>(), rank, node_beg, node_end);
  CUDACHECK(cudaStreamSynchronize(stream));
  return vir_nodes;
}

torch::Tensor GetVirSendOffset(
    int world_size, int base, torch::Tensor sorted_nodes,
    torch::Tensor unique_nodes) {
  auto cuda_options = sorted_nodes.options();
  thrust::device_vector<int> device_vids(world_size + 1);
  // init device_vids to base * i for in in [0, world_size]
  thrust::sequence(device_vids.begin(), device_vids.end(), 0, base);

  auto send_offset = torch::empty({world_size * 2 + 1}, cuda_options);
  auto num_sorted_nodes = sorted_nodes.numel();
  auto num_unique_nodes = unique_nodes.numel();
  thrust::device_ptr<IdType> sorted_nodes_start(
      sorted_nodes.data_ptr<IdType>()),
      sorted_nodes_end(sorted_nodes_start + num_sorted_nodes),
      unique_nodes_start(unique_nodes.data_ptr<IdType>()),
      unique_nodes_end(unique_nodes_start + num_unique_nodes),
      send_offset_start(send_offset.data_ptr<IdType>());

  // sorted_nodes[0:num_sorted_nodes]
  // search_val [0:world_size]
  thrust::lower_bound(
      sorted_nodes_start, sorted_nodes_end, device_vids.begin(),
      device_vids.end(), send_offset_start);

  // unique_node[0:num_unique_nodes]
  // search_val [1:world_size]
  thrust::lower_bound(
      unique_nodes_start, unique_nodes_end, device_vids.begin() + 1,
      device_vids.end(), send_offset_start + world_size + 1);

  return send_offset;
}

torch::Tensor GetVirSendOffsetWithDst(
    int world_size, int base, torch::Tensor sorted_nodes,
    torch::Tensor unique_nodes) {
  auto cuda_options = sorted_nodes.options();
  thrust::device_vector<int> device_vids(world_size * 2 + 1);
  // init device_vids to base * i for in in [0, world_size*2]
  thrust::sequence(device_vids.begin(), device_vids.end(), 0, base);

  auto send_offset = torch::empty({world_size * 3 + 1}, cuda_options);
  auto num_sorted_nodes = sorted_nodes.numel();
  auto num_unique_nodes = unique_nodes.numel();
  thrust::device_ptr<IdType> sorted_nodes_start(
      sorted_nodes.data_ptr<IdType>()),
      sorted_nodes_end(sorted_nodes_start + num_sorted_nodes),
      unique_nodes_start(unique_nodes.data_ptr<IdType>()),
      unique_nodes_end(unique_nodes_start + num_unique_nodes),
      send_offset_start(send_offset.data_ptr<IdType>());
  // sorted_nodes[0:num_sorted_nodes]
  // search_val [0:2*world_size+1]
  thrust::lower_bound(
      sorted_nodes_start, sorted_nodes_end, device_vids.begin(),
      device_vids.end(), send_offset_start);
  // unique_node[0:num_unique_nodes]
  // search_val [world_size+1:2*world_size+1]
  thrust::lower_bound(
      unique_nodes_start, unique_nodes_end,
      device_vids.begin() + world_size + 1, device_vids.end(),
      send_offset_start + world_size * 2 + 1);

  return send_offset;
}

// Load node feature from dev&uva in on time
// 1. classify feat type 0: on dev 1: on uva
__global__ void _CSRRowWiseLoadSubtensorAlignedAllKernel(
    IdType size, IdType dim, IdType rank_feat_dim, IdType feat_dim_offset,
    IdType *index, IdType *feat_pos_map, DataType *input_table_dev,
    DataType *input_table_uva, DataType *output_table) {
  IdType out_row = blockIdx.x * blockDim.y + threadIdx.y;
  while (out_row < size) {
    const IdType out_row_start = out_row * rank_feat_dim;
    IdType feat_type = -1;
    IdType origin_in_row_start = -1;
    if (feat_pos_map[index[out_row]] < -1) {
      // on uva, add feat_dim_offset to handle "Model Para"
      feat_type = FEAT_ON_UVA;
      origin_in_row_start =
          ENCODE_ID(feat_pos_map[index[out_row]]) * dim + feat_dim_offset;
    } else {
      // on dev
      feat_type = FEAT_ON_DEV;
      origin_in_row_start = feat_pos_map[index[out_row]] * rank_feat_dim;
    }

    const IdType in_row_start = origin_in_row_start & ~0x1F;
    const IdType in_row_end = origin_in_row_start + rank_feat_dim;
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
    IdType size, IdType feat_dim, IdType feat_dim_offset, torch::Tensor index,
    torch::Tensor feat_pos_map, torch::Tensor input_table_dev,
    torch::Tensor input_table_uva, torch::Tensor output_table) {
  auto stream = at::cuda::getCurrentCUDAStream();
  // constexpr int BLOCK_X = 128;
  auto rank_feat_dim = output_table.size(1);
  int BLOCK_X = min(128, (int)rank_feat_dim);
  constexpr int BLOCK_ROWS = 4;
  constexpr int TILE_SIZE = BLOCK_ROWS * 16;
  // constexpr int TILE_SIZE = 4192;
  dim3 block(BLOCK_X, BLOCK_ROWS);
  int BLOCK_NUM = (size + TILE_SIZE - 1) / TILE_SIZE;
  dim3 grid(BLOCK_NUM);

  _CSRRowWiseLoadSubtensorAlignedAllKernel<<<grid, block, 0, stream>>>(
      size, feat_dim, rank_feat_dim, feat_dim_offset, index.data_ptr<IdType>(),
      feat_pos_map.data_ptr<IdType>(), input_table_dev.data_ptr<DataType>(),
      input_table_uva.data_ptr<DataType>(), output_table.data_ptr<DataType>());
  CUDACHECK(cudaStreamSynchronize(stream));
}

template <int BLOCK_WARPS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleReplaceKernel(
    const uint64_t random_seed, int fanout, IdType num_frontier,
    IdType *frontier, IdType *dev_indptr, IdType *dev_indices,
    IdType *uva_indptr, IdType *uva_indices, IdType *adj_pos_map,
    IdType *out_indices) {
  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const IdType last_row =
      min(static_cast<IdType>(blockIdx.x + 1) * TILE_SIZE, num_frontier);

  curandState rng;
  curand_init(
      random_seed * gridDim.x + blockIdx.x,
      threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);
  while (out_row < last_row) {
    const IdType row = frontier[out_row];
    IdType pos = adj_pos_map[row];
    bool on_dev;
    if (pos >= 0) {
      on_dev = true;
    } else {
      pos = ENCODE_ID(pos);
      on_dev = false;
    }
    const IdType in_row_start = on_dev ? dev_indptr[pos] : uva_indptr[pos];
    const IdType out_row_start = out_row * fanout;
    const IdType deg =
        (on_dev ? dev_indptr[pos + 1] : uva_indptr[pos + 1]) - in_row_start;
    const IdType *index = (on_dev ? dev_indices : uva_indices) + in_row_start;
    if (deg > 0) {
      for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
        const IdType edge = curand(&rng) % deg;
        const IdType out_idx = out_row_start + idx;
        out_indices[out_idx] = index[edge];
      }
    }
    out_row += BLOCK_WARPS;
  }
}

torch::Tensor LocalSampleNeighbors(torch::Tensor frontier, int fanout) {
  auto *state = NPCState::Global();
  auto num_frontier = frontier.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint64_t random_seed = 7777777;

  auto adj_pos_map = state->graph_storage.adj_pos_map;
  auto dev_indptr = state->graph_storage.dev_indptr;
  auto dev_indices = state->graph_storage.dev_indices;
  auto uva_indptr = state->graph_storage.uva_indptr;
  auto uva_indices = state->graph_storage.uva_indices;

  auto options = frontier.options();
  auto out_indices = torch::empty({num_frontier * fanout}, options);

  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  dim3 block(WARP_SIZE, BLOCK_WARPS);
  dim3 grid((num_frontier + TILE_SIZE - 1) / TILE_SIZE);
  _CSRRowWiseSampleReplaceKernel<BLOCK_WARPS, TILE_SIZE>
      <<<grid, block, 0, stream>>>(
          random_seed, fanout, num_frontier, frontier.data_ptr<IdType>(),
          dev_indptr.data_ptr<IdType>(), dev_indices.data_ptr<IdType>(),
          uva_indptr.data_ptr<IdType>(), uva_indices.data_ptr<IdType>(),
          adj_pos_map.data_ptr<IdType>(), out_indices.data_ptr<IdType>());

  CUDACHECK(cudaStreamSynchronize(stream));
  return out_indices;
}

__global__ void _CountDeviceVerticesKernel(
    int world_size, IdType shuffle_id_offset, IdType *shuffle_min_vids,
    IdType num_seeds, IdType *seeds, IdType *bucket_size, IdType *belongs_to,
    IdType *inner_bucket_offset, int rank, int node_beg, int node_end) {
  __shared__ IdType local_count[MAX_NUM_DEVICES];
  __shared__ IdType device_vid[MAX_NUM_DEVICES];
  IdType idx = blockDim.x * blockIdx.x + threadIdx.x;
  IdType stride = gridDim.x * blockDim.x;
  if (threadIdx.x <= world_size) {
    device_vid[threadIdx.x] = shuffle_min_vids[threadIdx.x];
    local_count[threadIdx.x] = 0;
  }

  __syncthreads();

  IdType device_id, vid;
  while (idx < num_seeds) {
    device_id = shuffle_id_offset;
    vid = seeds[idx];
    while (device_id + 1 < world_size && device_vid[device_id + 1] <= vid) {
      ++device_id;
    }
    if (device_id < node_beg || device_id >= node_end) {
      device_id = rank;
    }
    belongs_to[idx] = device_id;
    // inner block offset
    inner_bucket_offset[idx] =
        atomicAdd((unsigned long long *)(local_count + device_id), 1);
    idx += stride;
  }

  __syncthreads();

  __shared__ IdType inter_block_offset[MAX_NUM_DEVICES];
  if (threadIdx.x < world_size) {
    inter_block_offset[threadIdx.x] = atomicAdd(
        (unsigned long long *)(bucket_size + threadIdx.x),
        local_count[threadIdx.x]);
  }
  __syncthreads();
  idx = blockDim.x * blockIdx.x + threadIdx.x;
  while (idx < num_seeds) {
    // inter block offset
    inner_bucket_offset[idx] += inter_block_offset[belongs_to[idx]];
    idx += stride;
  }
}

__global__ void _PermuteKernel(
    IdType num_seeds, IdType *seeds, IdType *bucket_offset,
    IdType *inner_bucket_offset, IdType *belongs_to, IdType *sorted_idx,
    IdType *permutation) {
  IdType idx = blockDim.x * blockIdx.x + threadIdx.x;
  IdType stride = gridDim.x * blockDim.x;
  while (idx < num_seeds) {
    auto rank = belongs_to[idx];
    auto offset =
        (rank == 0 ? 0 : bucket_offset[rank - 1]) + inner_bucket_offset[idx];

    sorted_idx[offset] = seeds[idx];
    permutation[idx] = offset;
    idx += stride;
  }
}

__global__ void _CountMultiMachinesDeviceKernel(
    int world_size, int num_remote_workers, IdType *feat_pos_map,
    IdType *remote_worker_map, IdType *min_vids, IdType num_seeds,
    IdType *seeds, IdType *bucket_size, IdType *belongs_to,
    IdType *inner_bucket_offset) {
  __shared__ IdType device_vid[MAX_NUM_DEVICES];
  __shared__ IdType local_count[MAX_NUM_DEVICES];
  IdType idx = blockDim.x * blockIdx.x + threadIdx.x;
  IdType stride = gridDim.x * blockDim.x;
  if (threadIdx.x <= world_size) {
    device_vid[threadIdx.x] = min_vids[threadIdx.x];
    local_count[threadIdx.x] = 0;
  }

  __syncthreads();
  IdType device_id, vid;
  while (idx < num_seeds) {
    device_id = FEAT_ON_LOCAL_DEV;
    vid = seeds[idx];
    if (feat_pos_map[vid] == FEAT_NOT_EXIST) {
      while (device_id + 1 < world_size && device_vid[device_id + 1] <= vid) {
        ++device_id;
      }
      device_id = remote_worker_map[device_id];
    }
    belongs_to[idx] = device_id;
    // intra block offset
    inner_bucket_offset[idx] =
        atomicAdd((unsigned long long *)(local_count + device_id), 1);
    idx += stride;
  }

  __syncthreads();

  __shared__ IdType inter_block_offset[MAX_NUM_DEVICES];
  if (threadIdx.x < num_remote_workers) {
    inter_block_offset[threadIdx.x] = atomicAdd(
        (unsigned long long *)(bucket_size + threadIdx.x),
        local_count[threadIdx.x]);
  }

  __syncthreads();
  idx = blockDim.x * blockIdx.x + threadIdx.x;
  while (idx < num_seeds) {
    // inter block offset
    inner_bucket_offset[idx] += inter_block_offset[belongs_to[idx]];
    idx += stride;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ClusterAndPermute(
    int world_size, torch::Tensor seeds, IdType shuffle_id_offset,
    torch::Tensor shuffle_min_vids, int rank, int node_beg, int node_end) {
  auto num_seeds = seeds.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto dev_tensor_options = seeds.options();

  auto bucket_size = torch::zeros({world_size}, dev_tensor_options);
  auto permutation = torch::empty({num_seeds}, dev_tensor_options);
  auto belongs_to = torch::empty({num_seeds}, dev_tensor_options);
  auto sorted_idx = torch::empty({num_seeds}, dev_tensor_options);
  auto inner_bucket_offset = torch::empty({num_seeds}, dev_tensor_options);
  // 1.Count bucket size
  int num_blocks = (num_seeds + NUM_THREADS - 1) / NUM_THREADS;

  dim3 block(NUM_THREADS);
  dim3 grid(num_blocks);

  _CountDeviceVerticesKernel<<<grid, block, 0, stream>>>(
      world_size, shuffle_id_offset, shuffle_min_vids.data_ptr<IdType>(),
      num_seeds, seeds.data_ptr<IdType>(), bucket_size.data_ptr<IdType>(),
      belongs_to.data_ptr<IdType>(), inner_bucket_offset.data_ptr<IdType>(),
      rank, node_beg, node_end);
  CUDACHECK(cudaStreamSynchronize(stream));

  auto bucket_offset = bucket_size.cumsum(0);

  // 2.permutation
  _PermuteKernel<<<grid, block, 0, stream>>>(
      num_seeds, seeds.data_ptr<IdType>(), bucket_offset.data_ptr<IdType>(),
      inner_bucket_offset.data_ptr<IdType>(), belongs_to.data_ptr<IdType>(),
      sorted_idx.data_ptr<IdType>(), permutation.data_ptr<IdType>());
  CUDACHECK(cudaStreamSynchronize(stream));

  return {bucket_size, bucket_offset, sorted_idx, permutation};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
MultiMachinesClusterAndPermute(torch::Tensor seeds) {
  auto *state = NPCState::Global();
  auto world_size = state->world_size;
  auto num_remote_workers = state->num_remote_workers;
  auto shuffle_min_vids = state->shuffle_min_vids;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto num_seeds = seeds.numel();
  auto cuda_options = seeds.options();
  auto remote_worker_map = state->remote_worker_map;
  auto bucket_size = torch::zeros({num_remote_workers}, cuda_options);
  auto sorted_idx = torch::empty({num_seeds}, cuda_options);
  auto inner_bucket_offset = torch::empty({num_seeds}, cuda_options);
  auto belongs_to = torch::empty({num_seeds}, cuda_options);
  auto permutation = torch::empty({num_seeds}, cuda_options);
  auto feat_pos_map = state->feat_storage.feat_pos_map;

  int num_blocks = (num_seeds + NUM_THREADS - 1) / NUM_THREADS;
  dim3 block(NUM_THREADS);
  dim3 grid(num_blocks);

  // cluster into local nodes and remote nodes(by min_vids)
  _CountMultiMachinesDeviceKernel<<<grid, block, 0, stream>>>(
      world_size, num_remote_workers, feat_pos_map.data_ptr<IdType>(),
      remote_worker_map.data_ptr<IdType>(), shuffle_min_vids.data_ptr<IdType>(),
      num_seeds, seeds.data_ptr<IdType>(), bucket_size.data_ptr<IdType>(),
      belongs_to.data_ptr<IdType>(), inner_bucket_offset.data_ptr<IdType>());
  CUDACHECK(cudaStreamSynchronize(stream));
  auto bucket_offset = bucket_size.cumsum(0);
  // permute
  _PermuteKernel<<<grid, block, 0, stream>>>(
      num_seeds, seeds.data_ptr<IdType>(), bucket_offset.data_ptr<IdType>(),
      inner_bucket_offset.data_ptr<IdType>(), belongs_to.data_ptr<IdType>(),
      sorted_idx.data_ptr<IdType>(), permutation.data_ptr<IdType>());
  CUDACHECK(cudaStreamSynchronize(stream));
  return {bucket_size, sorted_idx, permutation};
}

}  // namespace npc