#ifndef NPC_STATE_H_
#define NPC_STATE_H_

#include <nccl.h>

namespace npc {

struct NPCState {
  int rank, world_size;
  ncclComm_t nccl_comm;
  static NPCState* Global() {
    static NPCState state;
    return &state;
  }
};

}  // namespace npc

#endif