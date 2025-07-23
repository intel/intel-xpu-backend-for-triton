#include <torch/extension.h>

#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      auto msg = std::string("[") + __FILE__ +                                 \
                 "] Got cutlass error: " + cutlassGetStatusString(error) +     \
                 " at: " + std::to_string(__LINE__);                           \
      throw std::runtime_error(msg);                                           \
    }                                                                          \
  }

#include "attention/attention.hpp"
#include "gemm/gemm.hpp"

////////////////////////////////////////////////////////////////////////////////
// PYBIND MODULE
////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(cutlass_kernel, m) {
  m.def("gemm", &gemm_kernel, "gemm (CUTLASS)");
  m.def("attention", &attention_kernel, "attention (CUTLASS)");
}
