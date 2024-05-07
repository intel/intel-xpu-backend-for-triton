#include "softmax.h"
#include <ipex.h>
#include <torch/extension.h>
#include <vector>

sycl::queue get_current_sycl_queue() {
  // submit kernel
  c10::impl::VirtualGuardImpl impl(at::DeviceType::XPU);
  c10::Stream stream = impl.getStream(impl.getDevice());

  return xpu::get_queue_from_stream(stream);
}

#define CHECK_XPU(x)                                                           \
  TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_XPU(x);                                                                \
  CHECK_CONTIGUOUS(x)

at::Tensor softmax_shape_256_256(const at::Tensor &input, const int64_t dim) {
  CHECK_INPUT(input);

  auto output = at::empty_like(input);

  auto queue = get_current_sycl_queue();
  softmax_forward<mat1_256x256_bf16_cfg0>(input.data_ptr(), output.data_ptr(),
                                          queue);
  return output;
}

at::Tensor softmax_shape_1024_1024(const at::Tensor &input, const int64_t dim) {
  CHECK_INPUT(input);

  auto output = at::empty_like(input);

  auto queue = get_current_sycl_queue();
  softmax_forward<mat1_1024x1024_bf16_cfg0>(input.data_ptr(), output.data_ptr(),
                                            queue);
  return output;
}

at::Tensor softmax_shape_2048_2048(const at::Tensor &input, const int64_t dim) {
  CHECK_INPUT(input);

  auto output = at::empty_like(input);

  auto queue = get_current_sycl_queue();
  softmax_forward<mat1_2048x2048_bf16_cfg0>(input.data_ptr(), output.data_ptr(),
                                            queue);
  return output;
}

at::Tensor softmax_shape_4096_4096(const at::Tensor &input, const int64_t dim) {
  CHECK_INPUT(input);

  auto output = at::empty_like(input);

  auto queue = get_current_sycl_queue();
  softmax_forward<mat1_4096x4096_bf16_cfg0>(input.data_ptr(), output.data_ptr(),
                                            queue);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_shape_256_256", &softmax_shape_256_256,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_1024_1024", &softmax_shape_1024_1024,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_2048_2048", &softmax_shape_2048_2048,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_4096_4096", &softmax_shape_4096_4096,
        "softmax forward (XeTLA)");
}
