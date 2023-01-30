#include <torch/custom_class.h>
#include <torch/script.h>

#include "./core.h"
#include "./ops/collective.h"

namespace npc {

TORCH_LIBRARY(npc, m) {
  m.def("nccl_get_unique_id", &NCCLGetUniqueId)
      .def("init", &Initialize)
      .def("allreduce", &Allreduce);
}

}  // namespace npc