#include "flash_attention/fmha_forward_v5.h"
#include "gemm/gemm.h"
#include "softmax/softmax.h"
#include "stream_k_gemm/stream_k_gemm.h"
#include <CL/sycl.hpp>
#include <cstdint>

#include <ipex.h>
#include <torch/extension.h>

static constexpr float kNegInfinity = INFINITY * -1;

sycl::queue get_current_sycl_queue() {
  // submit kernel
  c10::impl::VirtualGuardImpl impl(at::DeviceType::XPU);
  c10::Stream stream = impl.getStream(impl.getDevice());

  return xpu::get_queue_from_stream(stream);
}

struct Shape {
  Shape(int B, int N, int F, int T, int H)
      : num_batches(B), num_heads(N), num_queries(F), num_keys(T),
        head_size(H) {}
  const int num_batches;
  const int num_heads;
  const int num_queries;
  const int num_keys;
  const int head_size;

  inline uint32_t get_query_size() const {
    return num_batches * num_heads * num_queries * head_size;
  }
  inline uint32_t get_key_size() const {
    return num_batches * num_heads * num_keys * head_size;
  }
  inline uint32_t get_score_size() const {
    return num_batches * num_heads * num_queries * num_keys;
  }
  inline uint32_t get_ml_size() const {
    return num_batches * num_heads * num_queries;
  }
  inline uint32_t get_attn_mask_size() const {
#if _BIAS_AS_INPUT
    return num_batches * num_heads * num_queries * num_keys;
#else
    return num_batches * num_queries * num_keys;
#endif
  }
};

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
  RECORD_FUNCTION("xetla softmax", {input});

  auto output = at::empty_like(input);

  auto queue = get_current_sycl_queue();
  auto evt = softmax_forward<T>(input.data_ptr(), output.data_ptr(), queue);
  xpu::profiler_record("xetla kernel", evt);
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
  RECORD_FUNCTION("xetla gemm", {a, b, c, acc});

  auto queue = get_current_sycl_queue();
  auto evt = gemm_run<T>(a.data_ptr(), b.data_ptr(), c.data_ptr(),
                         acc.data_ptr(), cnt.data_ptr(), queue);
  xpu::profiler_record("xetla kernel", evt);
  return acc;
}

at::Tensor bf16_stream_k_gemm(const at::Tensor &a, const at::Tensor &b,
                              const at::Tensor &c, const at::Tensor &acc,
                              const at::Tensor &cnt) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(c);
  CHECK_INPUT(acc);
  RECORD_FUNCTION("xetla stream_k_gemm", {a, b, c, acc});

  auto queue = get_current_sycl_queue();
  auto evt = stream_k_gemm_run(a.data_ptr(), b.data_ptr(), c.data_ptr(),
                               acc.data_ptr(), cnt.data_ptr(), queue);
  xpu::profiler_record("xetla kernel", evt);
  return acc;
}

// refers to test_fmha.cpp::test_fmha_forward()
template <typename T, bool IsCausal>
void flash_attn(const int64_t num_batches, const int64_t num_heads,
                const int64_t head_size, const int64_t num_queries,
                const int64_t num_keys) {
  Shape shape(num_batches, num_heads, num_queries, num_keys, head_size);
  auto queue = get_current_sycl_queue();

  constexpr bool use_mask = false;
  constexpr bool use_dropout = false;
  float dropout_prob = 0.0f;
  if constexpr (use_dropout)
    dropout_prob = 0.5f;
  const float scale = 1 / (1 - dropout_prob);
  const float head_scale = sycl::rsqrt(float(shape.head_size));

  uint32_t size_query = shape.get_query_size();
  uint32_t size_key = shape.get_key_size();
  uint32_t size_score = shape.get_score_size();
  uint32_t size_attn_mask = shape.get_attn_mask_size();
  uint32_t size_ml = shape.get_ml_size();

  // forward
  void *query_ptr = at::empty(size_query, at::kBFloat16).data_ptr();
  void *key_ptr = at::empty(size_key, at::kBFloat16).data_ptr();
  void *value_ptr = at::empty(size_key, at::kBFloat16).data_ptr();
  void *attn_mask_ptr = at::empty(size_attn_mask, at::kBFloat16).data_ptr();
  void *dropout_mask_ptr = at::empty(size_score, torch::kUInt8).data_ptr();

  void *output_ptr = at::empty(size_query, at::kBFloat16).data_ptr();
  void *m_ptr = at::empty(size_ml, at::kFloat).data_ptr();
  void *l_ptr = at::empty(size_ml, at::kFloat).data_ptr();

  // Type cast
  T *query = static_cast<T *>(query_ptr);
  T *key = static_cast<T *>(key_ptr);
  T *value = static_cast<T *>(value_ptr);
  T *attn_mask = static_cast<T *>(attn_mask_ptr);
  uint8_t *dropout_mask = static_cast<uint8_t *>(dropout_mask_ptr);

  T *output = static_cast<T *>(output_ptr);
  float *m = static_cast<float *>(m_ptr);
  float *l = static_cast<float *>(l_ptr);

  fmha_forward<T, use_mask, IsCausal, use_dropout>(
      queue, query, key, value, attn_mask, dropout_mask, dropout_prob, output,
      m, l, shape.num_batches, shape.num_heads, shape.head_size,
      shape.num_queries, shape.num_keys, head_scale);

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
  // stream_k_gemm
  m.def("stream_k_gemm_shape_1_3072_4096_3072", &bf16_stream_k_gemm,
        "bf16_stream_k_gemm (XeTLA)");
  // flash_attn_shape_$Z_$H_${N_CTX}_${D_HEAD}
  m.def(
      "flash_attn_shape_4_48_1024_64",
      [](const int64_t num_batches, const int64_t num_heads,
         const int64_t head_size, const int64_t num_queries,
         const int64_t num_keys) {
        return flash_attn<sycl::half, false>(num_batches, num_heads, head_size,
                                             num_queries, num_keys);
      },
      py::arg("num_batches"), py::arg("num_heads"), py::arg("head_size"),
      py::arg("num_queries"), py::arg("num_keys"), "flash attn (XeTLA)");
}
