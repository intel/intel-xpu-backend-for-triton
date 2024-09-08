#include "flash_attention/fmha_forward_v5.h"
#include "gemm/gemm.h"
#include "softmax/softmax.h"
#include "stream_k_gemm/stream_k_gemm.h"
#include <CL/sycl.hpp>
#include <c10/core/ScalarType.h>
#include <cstdint>

#ifdef USE_IPEX
#include <torch/extension.h>
#include <ipex.h>
#else
#include <c10/xpu/XPUStream.h>
#endif

sycl::queue get_current_sycl_queue() {
  // submit kernel
  c10::impl::VirtualGuardImpl impl(at::DeviceType::XPU);
  c10::Stream stream = impl.getStream(impl.getDevice());

#ifdef USE_IPEX
  auto queue = xpu::get_queue_from_stream(stream);
#else
  auto xpu_stream = c10::xpu::XPUStream(stream);
  auto queue = xpu_stream.queue();
#endif

  return queue;
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
  auto evt = softmax_forward<T>(input.data_ptr(), output.data_ptr(), queue);
  return output;
}

template <typename T>
at::Tensor bf16_gemm(const at::Tensor &a, const at::Tensor &b,
                     const at::Tensor &c, const at::Tensor &acc,
                     const at::Tensor &cnt) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  CHECK_INPUT(acc);

  auto queue = get_current_sycl_queue();
  auto evt = gemm_run<T>(a.data_ptr(), b.data_ptr(), c.data_ptr(),
                         acc.data_ptr(), cnt.data_ptr(), queue);
  return acc;
}

at::Tensor bf16_stream_k_gemm(const at::Tensor &a, const at::Tensor &b,
                              const at::Tensor &c, const at::Tensor &acc,
                              const at::Tensor &cnt) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  CHECK_INPUT(acc);

  auto queue = get_current_sycl_queue();
  auto evt = stream_k_gemm_run(a.data_ptr(), b.data_ptr(), c.data_ptr(),
                               acc.data_ptr(), cnt.data_ptr(), queue);
  return acc;
}

#define CALL_IMPL_ATTENTION_FUNC(P)                                            \
  fmha::fmha_forward_impl<P, T, use_mask, IsCausal, use_dropout>(              \
      queue, num_batches, num_heads, head_size, num_queries, num_keys)

template <bool use_mask = false, bool IsCausal = false,
          bool use_dropout = false>
void flash_attn(const int64_t num_batches, const int64_t num_heads,
                const int64_t head_size, const int64_t num_queries,
                const int64_t num_keys) {

  auto queue = get_current_sycl_queue();

  sycl::event evt;
  if (head_size <= 64) {
    evt = CALL_IMPL_ATTENTION_FUNC(fmha_policy_64x128x64);
  } else if (head_size <= 128) {
    evt = CALL_IMPL_ATTENTION_FUNC(fmha_policy_64x128x128);
  } else if (head_size <= 25) {
    if (num_keys <= 256) {
      evt = CALL_IMPL_ATTENTION_FUNC(fmha_policy_32x256x256);
    } else {
      evt = CALL_IMPL_ATTENTION_FUNC(fmha_policy_64x512x256);
    }
  } else {
    std::cout << "No policy available for current head_size " << head_size
              << "\n";
  }

  return;
}

PYBIND11_MODULE(xetla_kernel, m) {
  // softmax
  m.def("softmax_shape_4096_256", &softmax<mat1_4096x256_bf16_cfg0>,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_4096_1024", &softmax<mat1_4096x1024_bf16_cfg0>,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_4096_2048", &softmax<mat1_4096x2048_bf16_cfg0>,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_4096_4096", &softmax<mat1_4096x4096_bf16_cfg0>,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_4096_8192", &softmax<mat1_4096x8k_bf16_cfg0>,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_4096_16384", &softmax<mat1_4096x16k_bf16_cfg0>,
        "softmax forward (XeTLA)");
  m.def("softmax_shape_4096_32768", &softmax<mat1_4096x32k_bf16_cfg0>,
        "softmax forward (XeTLA)");
  // gemm
  m.def("gemm_shape_1_1024_1024_1024",
        &bf16_gemm<Test_1x1024x1024x1024_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_2048_2048_2048",
        &bf16_gemm<Test_1x2048x2048x2048_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_4096_4096_4096",
        &bf16_gemm<Test_1x4096x4096x4096_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_8192_8192_8192",
        &bf16_gemm<Test_1x8192x8192x8192_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_1_5120_13824", &bf16_gemm<Test_1x1x5120x13824_row_row>,
        "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_4_4096_12288", &bf16_gemm<Test_1x4x4096x12288_row_row>,
        "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_512_8192_8192", &bf16_gemm<Test_1x512x8192x8192_row_row>,
        "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_512_8192_32768",
        &bf16_gemm<Test_1x512x8192x32768_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_512_32768_8192",
        &bf16_gemm<Test_1x512x32768x8192_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_1024_16384_8192",
        &bf16_gemm<Test_1x1024x16384x8192_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_1024_28672_8192",
        &bf16_gemm<Test_1x1024x28672x8192_row_row>, "bf16_gemm (XeTLA)");
  // FIXME: Remove this case when gemm_streamk_benchmark works
  m.def("gemm_shape_1_3072_4096_3072",
        &bf16_gemm<Test_1x3072x4096x3072_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_4096_16384_8192",
        &bf16_gemm<Test_1x4096x16384x8192_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_8192_16384_1024",
        &bf16_gemm<Test_1x8192x16384x1024_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_8192_16384_4096",
        &bf16_gemm<Test_1x8192x16384x4096_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_16384_1024_8192",
        &bf16_gemm<Test_1x16384x1024x8192_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_16384_4096_8192",
        &bf16_gemm<Test_1x16384x4096x8192_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_16384_8192_1024",
        &bf16_gemm<Test_1x16384x8192x1024_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_1_16384_8192_4096",
        &bf16_gemm<Test_1x16384x8192x4096_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_4_32768_128_4096",
        &bf16_gemm<Test_4x32768x128x4096_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_4_32768_4096_128",
        &bf16_gemm<Test_4x32768x4096x128_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_32_4096_4096_128",
        &bf16_gemm<Test_32x4096x4096x128_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_4096_8_128_16384",
        &bf16_gemm<Test_4096x8x128x16384_row_row>, "bf16_gemm (XeTLA)");
  m.def("gemm_shape_4096_8_16384_128",
        &bf16_gemm<Test_4096x8x16384x128_row_row>, "bf16_gemm (XeTLA)");
  // flash_attn
  m.def("flash_attn", &flash_attn<false, false, false>, "flash attn (XeTLA)");
}
