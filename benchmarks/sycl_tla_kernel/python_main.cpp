#include <torch/extension.h>

#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      auto msg = std::string("[") + __FILE__ +                                 \
                 "] Got cutlass error: " + cutlassGetStatusString(error) +     \
                 " at: " + std::to_string(__LINE__);                           \
      throw std::runtime_error(msg);                                           \
    }                                                                          \
  }

#include "attention/benchmark_runner.hpp"
#include "attention/fmha_configuration.hpp"
#include "gemm/gemm.hpp"
#include "gemm_splitk/gemm_splitk.hpp"

// Example FMHA configuration struct (user should adapt as needed)
using FMHAPrefill_HF16_HF16_HF16_FP32_RCR_h64_NonCausal_FixedLen =
    cutlass::flash_attention::FMHAConfigGen<
        /*Mode*/ cutlass::flash_attention::FMHAMode::Prefill,
        /*ElementQ*/ cutlass::half_t, /*ElementK*/ cutlass::half_t,
        /*ElementV*/ cutlass::half_t, /*ElementO*/ float,
        /*LayoutQ*/ cutlass::layout::RowMajor,
        /*LayoutK*/ cutlass::layout::ColumnMajor,
        /*LayoutV*/ cutlass::layout::RowMajor,
        /*LayoutO*/ cutlass::layout::RowMajor,
        /*Causal*/ false, /*VarLen*/ false, /*CachedKV*/ false,
        /*PagedKV*/ false, /*Persistent*/ false, /*HeadDim*/ 64>::type;

using FMHAPrefill_HF16_HF16_HF16_FP32_RCR_h128_NonCausal_FixedLen =
    cutlass::flash_attention::FMHAConfigGen<
        /*Mode*/ cutlass::flash_attention::FMHAMode::Prefill,
        /*ElementQ*/ cutlass::half_t, /*ElementK*/ cutlass::half_t,
        /*ElementV*/ cutlass::half_t, /*ElementO*/ float,
        /*LayoutQ*/ cutlass::layout::RowMajor,
        /*LayoutK*/ cutlass::layout::ColumnMajor,
        /*LayoutV*/ cutlass::layout::RowMajor,
        /*LayoutO*/ cutlass::layout::RowMajor,
        /*Causal*/ false, /*VarLen*/ false, /*CachedKV*/ false,
        /*PagedKV*/ false, /*Persistent*/ false, /*HeadDim*/ 128>::type;

using FMHAPrefill_HF16_HF16_HF16_FP32_RCR_h64_Causal_FixedLen =
    cutlass::flash_attention::FMHAConfigGen<
        /*Mode*/ cutlass::flash_attention::FMHAMode::Prefill,
        /*ElementQ*/ cutlass::half_t, /*ElementK*/ cutlass::half_t,
        /*ElementV*/ cutlass::half_t, /*ElementO*/ float,
        /*LayoutQ*/ cutlass::layout::RowMajor,
        /*LayoutK*/ cutlass::layout::ColumnMajor,
        /*LayoutV*/ cutlass::layout::RowMajor,
        /*LayoutO*/ cutlass::layout::RowMajor,
        /*Causal*/ true, /*VarLen*/ false, /*CachedKV*/ false,
        /*PagedKV*/ false, /*Persistent*/ false, /*HeadDim*/ 64>::type;

using FMHAPrefill_HF16_HF16_HF16_FP32_RCR_h128_Causal_FixedLen =
    cutlass::flash_attention::FMHAConfigGen<
        /*Mode*/ cutlass::flash_attention::FMHAMode::Prefill,
        /*ElementQ*/ cutlass::half_t, /*ElementK*/ cutlass::half_t,
        /*ElementV*/ cutlass::half_t, /*ElementO*/ float,
        /*LayoutQ*/ cutlass::layout::RowMajor,
        /*LayoutK*/ cutlass::layout::ColumnMajor,
        /*LayoutV*/ cutlass::layout::RowMajor,
        /*LayoutO*/ cutlass::layout::RowMajor,
        /*Causal*/ true, /*VarLen*/ false, /*CachedKV*/ false,
        /*PagedKV*/ false, /*Persistent*/ false, /*HeadDim*/ 128>::type;

auto attention_08(const at::Tensor &Q, const at::Tensor &K, const at::Tensor &V,
                  at::Tensor &O, int Batch, int NumHeadsQ, int NumHeadsKV,
                  int SeqLengthQO, int SeqLengthKV, int HeadSizeQK,
                  int HeadSizeVO, bool Causal, float sm_scale) -> int {
  ::FMHAOptions options;
  options.batch = Batch;
  options.num_heads_q = NumHeadsQ;
  options.num_heads_kv = NumHeadsKV;
  options.seq_len_qo = SeqLengthQO;
  options.seq_len_kv = SeqLengthKV;
  options.head_size_qk = HeadSizeQK;
  options.head_size_vo = HeadSizeVO;
  options.softmax_scale = sm_scale;

  cutlass::KernelHardwareInfo hw_info;
  if (hw_info.sm_count == 0) {
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    CUTLASS_TRACE_HOST(
        "Query result for SM count per device: " << hw_info.sm_count);
  }

  if (HeadSizeVO == 64 && Causal) {
    ::FMHARunner<FMHAPrefill_HF16_HF16_HF16_FP32_RCR_h64_Causal_FixedLen>().run(
        options, hw_info, Q, K, V, O);
  } else if (HeadSizeVO == 64 && !Causal) {
    ::FMHARunner<FMHAPrefill_HF16_HF16_HF16_FP32_RCR_h64_NonCausal_FixedLen>()
        .run(options, hw_info, Q, K, V, O);
  } else if (HeadSizeVO == 128 && Causal) {
    ::FMHARunner<FMHAPrefill_HF16_HF16_HF16_FP32_RCR_h128_Causal_FixedLen>()
        .run(options, hw_info, Q, K, V, O);
  } else if (HeadSizeVO == 128 && !Causal) {
    ::FMHARunner<FMHAPrefill_HF16_HF16_HF16_FP32_RCR_h128_NonCausal_FixedLen>()
        .run(options, hw_info, Q, K, V, O);
  } else {
    TORCH_CHECK(false,
                "sycl-tla attention: unsupported HeadSizeVO=", HeadSizeVO,
                " (supported: 64, 128)");
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// PYBIND MODULE
////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(sycl_tla_kernel, m) {
  m.def("gemm", &gemm_kernel, "gemm (SYCL-TLA)");
  m.def("gemm_splitk", &gemm_splitk_kernel, "gemm split-K (SYCL-TLA)");
  m.def("attention", &attention_08, "attention (SYCL-TLA)");
}
