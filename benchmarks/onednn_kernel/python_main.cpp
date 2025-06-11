#include "oneapi/dnnl/dnnl_common.hpp"
#include "softmax/softmax.h"
#include <ATen/record_function.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <torch/extension.h>

#define CHECK_XPU(x)                                                           \
  TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_XPU(x);                                                                \
  CHECK_CONTIGUOUS(x)

at::Tensor softmax(const int64_t M, const int64_t N, const at::Tensor &input,
                   const at::Tensor &output, const int64_t dim) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  RECORD_FUNCTION("onednn softmax", {});

  dnnl::engine engine(dnnl::engine::kind::gpu, 0);
  dnnl::stream engine_stream(engine);
  auto evt = softmax_example(M, N, dim, input.data_ptr(), output.data_ptr(),
                             engine, engine_stream);
  return output;
}

PYBIND11_MODULE(onednn_kernel, m) {
  // softmax
  m.def("onednn_softmax", &softmax, "softmax forward (oneDNN)");
  ;
}
