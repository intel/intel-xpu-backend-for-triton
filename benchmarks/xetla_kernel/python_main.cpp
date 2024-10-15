#include "flash_attention/fmha_backward.h"
#include "flash_attention/fmha_forward_v5.h"
#include "gemm/gemm.h"
#include "softmax/softmax.h"
#include "stream_k_gemm/stream_k_gemm.h"
#include <CL/sycl.hpp>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <torch/extension.h>

#ifdef USE_IPEX
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
at::Tensor softmax(const at::Tensor &input, const at::Tensor &output,
                   const int64_t dim) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
#ifdef USE_IPEX
  RECORD_FUNCTION("xetla softmax", {});
#endif

  auto queue = get_current_sycl_queue();
  auto evt = softmax_forward<T>(input.data_ptr(), output.data_ptr(), queue);
#ifdef USE_IPEX
  xpu::profiler_record("xetla kernel", evt);
#endif
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
#ifdef USE_IPEX
  RECORD_FUNCTION("xetla gemm", {});
#endif

  auto queue = get_current_sycl_queue();
  auto evt = gemm_run<T>(a.data_ptr(), b.data_ptr(), c.data_ptr(),
                         acc.data_ptr(), cnt.data_ptr(), queue);
#ifdef USE_IPEX
  xpu::profiler_record("xetla kernel", evt);
#endif
  return acc;
}

at::Tensor bf16_stream_k_gemm(const at::Tensor &a, const at::Tensor &b,
                              const at::Tensor &c, const at::Tensor &acc,
                              const at::Tensor &cnt) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  CHECK_INPUT(acc);
#ifdef USE_IPEX
  RECORD_FUNCTION("xetla stream_k_gemm", {});
#endif

  auto queue = get_current_sycl_queue();
  auto evt = stream_k_gemm_run(a.data_ptr(), b.data_ptr(), c.data_ptr(),
                               acc.data_ptr(), cnt.data_ptr(), queue);
#ifdef USE_IPEX
  xpu::profiler_record("xetla kernel", evt);
#endif
  return acc;
}

#define CALL_IMPL_ATTENTION_FWD_FUNC(P)                                        \
  fmha::fmha_forward_impl<P, T, use_mask, IsCausal, use_dropout>(              \
      queue, q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),         \
      dropout_mask.data_ptr(), bias.data_ptr(), m.data_ptr(), l.data_ptr(),    \
      num_batches, num_heads, head_size, num_queries, num_keys, head_scale)

template <bool use_mask = false, bool IsCausal = false,
          bool use_dropout = false>
void flash_attn(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                const at::Tensor &out, const at::Tensor &dropout_mask,
                const at::Tensor &bias, const at::Tensor &m,
                const at::Tensor &l, const int64_t num_batches,
                const int64_t num_heads, const int64_t head_size,
                const int64_t num_queries, const int64_t num_keys,
                float head_scale) {

  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(out);
  CHECK_INPUT(dropout_mask);
  CHECK_INPUT(bias);
  CHECK_INPUT(m);
  CHECK_INPUT(l);
#ifdef USE_IPEX
  RECORD_FUNCTION("xetla fa", {});
#endif

  auto queue = get_current_sycl_queue();

  sycl::event evt;
  if (head_size <= 64) {
    evt = CALL_IMPL_ATTENTION_FWD_FUNC(fmha_policy_64x128x64);
  } else if (head_size <= 128) {
    evt = CALL_IMPL_ATTENTION_FWD_FUNC(fmha_policy_64x128x128);
  } else if (head_size <= 25) {
    if (num_keys <= 256) {
      evt = CALL_IMPL_ATTENTION_FWD_FUNC(fmha_policy_32x256x256);
    } else {
      evt = CALL_IMPL_ATTENTION_FWD_FUNC(fmha_policy_64x512x256);
    }
  } else {
    std::cout << "No policy available for current head_size " << head_size
              << "\n";
  }

#ifdef USE_IPEX
  xpu::profiler_record("xetla kernel", evt);
#endif
  return;
}

#define CALL_IMPL_ATTENTION_BWD_FUNC(P)                                        \
  fmha::xetla_fmha_backward_kernel<P, T, kUseBias, kIsCausal, kIsDropout>(     \
      queue, grad_out.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),    \
      bias.data_ptr(), dropout.data_ptr(), out.data_ptr(),                     \
      log_sumexp.data_ptr(), workspace.data_ptr(), grad_q_tmp.data_ptr(),      \
      alpha, dropout_prob, grad_query.data_ptr(), grad_key.data_ptr(),         \
      grad_value.data_ptr(), grad_bias.data_ptr(), num_batches, num_heads,     \
      head_size, num_queries, num_keys, bias_strideB, bias_strideN,            \
      bias_strideF, attn_mask_padding)

template <bool kUseBias = false, bool kIsCausal = false,
          bool kIsDropout = false>
void flash_attn_bwd(const at::Tensor &grad_out, const at::Tensor &q,
                    const at::Tensor &k, const at::Tensor &v,
                    const at::Tensor &bias, const at::Tensor &dropout,
                    const at::Tensor &out, const at::Tensor &log_sumexp,
                    const at::Tensor &workspace, const at::Tensor &grad_q_tmp,
                    float alpha, float dropout_prob,
                    const at::Tensor &grad_query, const at::Tensor &grad_key,
                    const at::Tensor &grad_value, const at::Tensor &grad_bias,
                    const int64_t num_batches, const int64_t num_heads,
                    const int64_t head_size, const int64_t num_queries,
                    const int64_t num_keys, const int64_t bias_strideB,
                    const int64_t bias_strideN, const int64_t bias_strideF,
                    const int64_t attn_mask_padding) {

  CHECK_INPUT(grad_out);
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(bias);
  CHECK_INPUT(dropout);
  CHECK_INPUT(out);
  CHECK_INPUT(log_sumexp);
  CHECK_INPUT(workspace);
  CHECK_INPUT(grad_q_tmp);
  CHECK_INPUT(grad_query);
  CHECK_INPUT(grad_key);
  CHECK_INPUT(grad_value);
  CHECK_INPUT(grad_bias);

#ifdef USE_IPEX
  RECORD_FUNCTION("xetla fa", {});
#endif

  auto queue = get_current_sycl_queue();

  sycl::event evt;
  if (head_size <= 64) {
    evt = CALL_IMPL_ATTENTION_BWD_FUNC(fmha_bwd_policy_128x128x64);
  } else if (head_size <= 128) {
    evt = CALL_IMPL_ATTENTION_BWD_FUNC(fmha_bwd_policy_128x128x128);
  } else if (head_size <= 256) {
    evt = CALL_IMPL_ATTENTION_BWD_FUNC(fmha_bwd_policy_128x128x256);
  } else if (head_size <= 512) {
    evt = CALL_IMPL_ATTENTION_BWD_FUNC(fmha_bwd_policy_64x128x512);
  } else {
    std::cout << "No policy available for current head_size " << head_size
              << "\n";
  }

#ifdef USE_IPEX
  xpu::profiler_record("xetla kernel", evt);
#endif
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
  // gemm stream k
  m.def("gemm_streamk_shape_3072_4096_3072", &bf16_stream_k_gemm,
        "bf16_gemm_streamk (XeTLA)");
  // flash_attn
  m.def("flash_attn_causal_false", &flash_attn<false, false, false>,
        "flash attn fwd (XeTLA)");
  m.def("flash_attn_causal_true", &flash_attn<false, true, false>,
        "flash attn fwd (XeTLA)");
  // flash_attn_bwd
  m.def("flash_attn_bwd_causal_false", &flash_attn_bwd<false, false, false>,
        "flash attn bwd (XeTLA)");
  m.def("flash_attn_bwd_causal_true", &flash_attn_bwd<false, true, false>,
        "flash attn bwd (XeTLA)");
}
