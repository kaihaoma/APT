#ifndef NPC_STATE_H_
#define NPC_STATE_H_

#include <nccl.h>

#include "./utils.h"

namespace npc {

struct FeatStorage {
  torch::Tensor labels;
  torch::Tensor dev_feats, uva_feats, global_shared_feats;
  torch::Tensor feat_pos_map, cpu_feat_pos_map;
  IdType num_dev_nodes, num_uva_nodes, num_graph_nodes, num_total_nodes;
  IdType feat_dim, rank_feat_dim, feat_dim_offset;
};

struct GraphStorage {
  torch::Tensor dev_indptr, dev_indices;
  torch::Tensor uva_indptr, uva_indices;
  torch::Tensor adj_pos_map;
  IdType num_graph_nodes, num_total_nodes, num_cached_nodes;
};

struct Profiler {
  // graph storage
  IdType graph_num_cached_nodes, graph_num_total_nodes, graph_cache_bytes;
  // feat storage
  IdType feat_num_cached_nodes, node_num_total_nodes, feat_cache_bytes;
};

struct NPCState {
  // nccl communication
  int rank, local_rank, world_size, node_size;
  ncclComm_t nccl_comm;
  std::vector<ncclComm_t> nccl_comm_list;
  int sampler_id, trainer_id, num_threads;
  cudaStream_t nccl_stream, cuda_copy_stream;
  std::vector<cudaStream_t> vec_cuda_stream;
  // shuffle info
  torch::Tensor shuffle_min_vids;
  IdType shuffle_id_offset;

  // node feats
  GraphStorage graph_storage;
  // graph topology
  FeatStorage feat_storage;
  // SP alltoall permute
  torch::Tensor sp_alltoall_size_permute_step2, sp_alltoall_size_permute_step3;

  // cross_machine_flag
  bool cross_machine_flag;
  int num_remote_workers;
  torch::Tensor remote_worker_map;
  std::vector<IdType> vec_remote_worker_id, vec_remote_worker_map;

  static NPCState *Global() {
    static NPCState state;
    return &state;
  }
  // for logs
  std::string tag;
};

}  // namespace npc

#endif