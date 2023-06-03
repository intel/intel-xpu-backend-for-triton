#include <pybind11/pybind11.h>

void init_intel_xpu_backend_for_triton(pybind11::module &m);

PYBIND11_MODULE(libintel_xpu_backend_for_triton, m) {
  m.doc() = "Python bindings to the C++ Intel XPU Backend for Triton API";
  init_intel_xpu_backend_for_triton(m);
}
