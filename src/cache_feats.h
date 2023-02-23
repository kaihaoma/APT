#ifndef NPC_CACHE_FEATS_H_
#define NPC_CACHE_FEATS_H_

#include <torch/script.h>

#include <string>

#include "./utils.h"

namespace npc {

void CacheFeats(
    torch::Tensor node_feats, torch::Tensor sorted_idx,
    IdType num_cached_nodes);
}

#endif