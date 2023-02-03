#ifndef NPC_STATE_H_
#define NPC_STATE_H_

#include <nccl.h>

#include "./utils.h"

namespace npc {

struct NPCState {
  // nccl communication
  int rank, world_size;
  ncclComm_t nccl_comm;
  // node feats
  int num_dev_nodes, num_host_nodes;
  std::vector<IdType> feat_pos_map;
  DataType *dev_feats, *host_feats;

  static NPCState* Global() {
    static NPCState state;
    return &state;
  }
};

}  // namespace npc

#endif