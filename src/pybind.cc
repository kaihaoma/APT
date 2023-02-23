#include <torch/custom_class.h>
#include <torch/script.h>

#include "./cache_feats.h"
#include "./cache_graphs.h"
#include "./core.h"
#include "./feat_shuffle.h"
#include "./load_subtensor.h"
#include "./ops/collective.h"
#include "./sampling.h"
#include "./utils.h"

namespace npc {

TORCH_LIBRARY(npc, m) {
  m.def("nccl_get_unique_id", &NCCLGetUniqueId)
      .def("init", &Initialize)
      .def("allreduce", &Allreduce)
      .def("cache_feats", &CacheFeats)
      .def("cache_graphs", &CacheGraphs)
      .def("load_subtensor", &LoadSubtensor)
      .def("sample_neighbors", &SamplingNeighbors)
      .def("feat_shuffle", &FeatShuffle)
      .def("test", &Test);
}

}  // namespace npc