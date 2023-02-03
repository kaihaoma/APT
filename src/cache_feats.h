#ifndef CACHE_FEATS_H_
#define CACHE_FEATS_H_

#include <torch/script.h>

#include <string>

namespace npc {
void CacheFeats(
    torch::Tensor node_feats, torch::Tensor sorted_idx,
    int64_t num_cached_nodes);
}

#endif