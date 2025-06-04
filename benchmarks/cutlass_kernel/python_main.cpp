#include <torch/extension.h>

#include "gemm/gemm.hpp"
// #include "attention/attention.hpp"

////////////////////////////////////////////////////////////////////////////////
// PYBIND MODULE
////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(cutlass_kernel, m) {
  m.def("gemm", &gemm, "gemm (CUTLASS)");
  // m.def("attention", &attention, "attention (CUTLASS)");
}
