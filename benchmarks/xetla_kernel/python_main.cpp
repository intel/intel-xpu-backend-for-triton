#include "bgemm.h"
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

template <typename T>
at::Tensor softmax(const at::Tensor &input, const int64_t dim) {
  CHECK_INPUT(input);

  auto output = at::empty_like(input);

  auto queue = get_current_sycl_queue();
  softmax_forward<T>(input.data_ptr(), output.data_ptr(), queue);
  return output;
}

template <typename T>
at::Tensor bgemm(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c,
                 const at::Tensor &d, const at::Tensor &cnt) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  CHECK_INPUT(d);

  auto queue = get_current_sycl_queue();
  bgemm_run<T>(a.data_ptr(), b.data_ptr(), c.data_ptr(), d.data_ptr(),
               cnt.data_ptr(), queue);

  return d;
}

PYBIND11_MODULE(xetla_kernel, m) {
  m.def("softmax_shape_256_256", &softmax<mat1_256x256_bf16_cfg0>,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_1024_1024", &softmax<mat1_1024x1024_bf16_cfg0>,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_2048_2048", &softmax<mat1_2048x2048_bf16_cfg0>,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_4096_4096", &softmax<mat1_4096x4096_bf16_cfg0>,
        "softmax forward (XeTLA)");
  // bgemm: M=N=K [256, 512 ... 4096]
  m.def("bgemm_shape_256_256_256", &bgemm<Test_256x256x256_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_512_512_512", &bgemm<Test_512x512x512_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_768_768_768", &bgemm<Test_768x768x768_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_1024_1024_1024", &bgemm<Test_1024x1024x1024_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_1280_1280_1280", &bgemm<Test_1280x1280x1280_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_1536_1536_1536", &bgemm<Test_1536x1536x1536_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_1792_1792_1792", &bgemm<Test_1792x1792x1792_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_2048_2048_2048", &bgemm<Test_2048x2048x2048_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_2304_2304_2304", &bgemm<Test_2304x2304x2304_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_2560_2560_2560", &bgemm<Test_2560x2560x2560_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_2816_2816_2816", &bgemm<Test_2816x2816x2816_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_3072_3072_3072", &bgemm<Test_3072x3072x3072_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_3328_3328_3328", &bgemm<Test_3328x3328x3328_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_3584_3584_3584", &bgemm<Test_3584x3584x3584_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_3840_3840_3840", &bgemm<Test_3840x3840x3840_row_row>,
        "bgemm (XeTLA)");
  m.def("bgemm_shape_4096_4096_4096", &bgemm<Test_4096x4096x4096_row_row>,
        "bgemm (XeTLA)");
}
