// Minimal stub when LLD is not available: defines init_triton_amd so the main
// libtriton.so links. The AMD backend is registered but not functional.
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_amd(py::module &&m) {
  m.doc() = "AMD Triton backend (stub; LLD not available, full backend disabled)";
}
