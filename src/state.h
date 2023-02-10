#ifndef NPC_STATE_H_
#define NPC_STATE_H_

#include <nccl.h>

#include "./utils.h"

namespace npc {

struct FeatStorage {
  torch::Tensor dev_feats, uva_feats;
  torch::Tensor feat_pos_map;
  IdType num_dev_nodes, num_uva_nodes, num_total_nodes;
  IdType input_dim;
};

struct GraphStorage {
  torch::Tensor dev_indptr, dev_indices;
  torch::Tensor adj_pos_map;
  IdType num_graph_nodes;
};

struct NPCState {
  // nccl communication
  int rank, world_size;
  ncclComm_t nccl_comm;
  // node feats
  GraphStorage graph_storage;
  // graph topology
  FeatStorage feat_storage;

  static NPCState *Global() {
    static NPCState state;
    return &state;
  }
};

}  // namespace npc

#endif