#include <torch/custom_class.h>
#include <torch/script.h>

#include "./cache_feats.h"
#include "./cache_graphs.h"
#include "./core.h"
#include "./feat_shuffle.h"
#include "./load_subtensor.h"
#include "./ops/collective.h"
#include "./relabel.h"
#include "./sampling.h"
#include "./sddmm.h"
#include "./spmm.h"
#include "./utils.h"

namespace npc {

TORCH_LIBRARY(npc, m) {
  m.def("nccl_get_unique_id", &NCCLGetUniqueId)
      .def("init", &Initialize)
      .def("alltoall", &AlltoAll)
      .def("cache_feats_shared", &CacheFeatsShared)
      .def("mix_cache_graphs", &MixCacheGraphs)
      .def("register_min_vids", &RegisterMinVids)
      .def("register_multi_machines_scheme", &RegisterMultiMachinesScheme)
      .def("cpu_load_subtensor", &CPULoadSubtensor)
      .def("load_subtensor", &LoadSubtensor)
      .def("crossmachine_load_subtensor", &CrossMachineLoadSubtensor)
      .def("cluster_reqs", &ClusterReqs)
      .def("local_sample_one_layer", &LocalSamplingNeibhorsOneLayer)
      .def("srcdst_to_vir", &SrcDsttoVir)
      .def("src_to_vir", &SrcToVir)
      .def("np_sample_and_shuffle", &NPSampleAndShuffle)
      .def("sp_sample_and_shuffle", &SPSampleAndShuffle)
      .def("sp_sample_shuffle_src", &SPSampleShuffleSrc)
      .def("mp_sample_shuffle", &MPSampleShuffle)
      .def("feat_shuffle", &FeatShuffle)
      .def("sp_feat_shuffle", &SPFeatShuffle)
      .def("mp_feat_shuffle_fwd", &MPFeatShuffleFwd)
      .def("mp_feat_shuffle_bwd", &MPFeatShuffleBwd)
      .def("shuffle_seeds", &ShuffleSeeds)
      .def("relabel_csc", &RelabelCSC)
      .def("spmm_copy_u_sum", &CopyUSum)
      .def("spmm_copy_e_sum", &CopyESum)
      .def("spmm_u_mul_e_sum", &UMulESum)
      .def("sddmm_u_add_v", &UAddV)
      .def("sddmm_u_mul_v", &UMulV);
}

}  // namespace npc