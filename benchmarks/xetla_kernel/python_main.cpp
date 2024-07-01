#include "bgemm/bgemm.h"
#include "flash_attention/fmha_forward_v5.h"
#include "softmax/softmax.h"
#include <CL/sycl.hpp>
#include <cstdint>
#include <ipex.h>
#include <random>
#include <torch/extension.h>
#include <vector>

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

template <typename T> void init_random(T *arr, size_t nelems) {
  std::vector<float> vec(nelems);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> rg(-1.0f, 1.0f);
  std::generate(vec.begin(), vec.end(), [&] { return rg(gen); });

  for (uint32_t i = 0; i < nelems; ++i)
    arr[i] = static_cast<T>(vec[i]);
}

template <typename T>
void random_init_forward(const Shape &shape, T *query, T *key, T *value,
                         T *attn_mask, uint8_t *dropout_mask,
                         const bool use_mask, const bool use_dropout) {
  uint32_t size_query = shape.get_query_size();
  uint32_t size_key = shape.get_key_size();
  uint32_t size_score = shape.get_score_size();
  uint32_t size_attn_mask = shape.get_attn_mask_size();

  init_random(query, size_query);
  init_random(key, size_key);
  init_random(value, size_key);
  if (use_mask) {
    init_random(attn_mask, size_attn_mask);
  }
  if (use_dropout) {
    for (uint32_t i = 0; i < size_score; ++i)
      dropout_mask[i] = (i & 1) == 0;
  }
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
  RECORD_FUNCTION("xetla softmax", {input});

  auto output = at::empty_like(input);

  auto queue = get_current_sycl_queue();
  auto evt = softmax_forward<T>(input.data_ptr(), output.data_ptr(), queue);
  xpu::profiler_record("xetla kernel", evt);
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

// refers to test_fmha.cpp::test_fmha_forward()
template <typename T, bool IsCausal>
T *flash_attn(const int64_t num_batches, const int64_t num_heads,
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

  random_init_forward(shape, query, key, value, attn_mask, dropout_mask,
                      use_mask, use_dropout);

  fmha_forward<T, use_mask, IsCausal, use_dropout>(
      queue, query, key, value, attn_mask, dropout_mask, dropout_prob, output,
      m, l, shape.num_batches, shape.num_heads, shape.head_size,
      shape.num_queries, shape.num_keys, head_scale);

  return output;
}

PYBIND11_MODULE(xetla_kernel, m) {
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
