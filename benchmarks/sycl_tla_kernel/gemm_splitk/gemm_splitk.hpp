#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/thread/activation.h"

#include <exception>
#include <iostream>

#define CUTLASS_CREATE_GEMM_BENCHMARK(x)
#define CUTLASS_BENCHMARK(x)
#include "benchmarks_splitk_sycl.hpp"

////////////////////////////////////////////////////////////////////////////////
// PRIVATE FUNCTION
////////////////////////////////////////////////////////////////////////////////

template <typename GemmConfig>
static auto gemm_splitk_run(const at::Tensor &A, const at::Tensor &B,
                            at::Tensor &C, const int M, const int N,
                            const int K, const int split_k) -> int {
  RECORD_FUNCTION("sycl-tla gemm_splitk", {});

  using ElementComputeEpilogue = float;

  try {
    using Gemm = GemmConfig::Gemm;
    typename Gemm::Arguments arguments;

    const cutlass::bfloat16_t *_A =
        static_cast<const cutlass::bfloat16_t *>(A.data_ptr());
    const cutlass::bfloat16_t *_B =
        static_cast<const cutlass::bfloat16_t *>(B.data_ptr());
    float *_C = static_cast<float *>(C.data_ptr());

    using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
    ProblemShapeType problem_size = ProblemShapeType{M, N, K, 1};

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    static cutlass::KernelHardwareInfo hw_info;
    if (hw_info.sm_count == 0) {
      hw_info.sm_count =
          cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    }

    using StreamKMode = cutlass::gemm::kernel::detail::
        PersistentTileSchedulerXeStreamKParams::DecompositionMode;

    arguments = GemmConfig::defaultArguments();
    arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    arguments.problem_shape = problem_size;
    arguments.mainloop = {_A, stride_A, _B, stride_B};
    arguments.epilogue = {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
        nullptr,
        stride_C,
        _C,
        stride_D};
    arguments.hw_info = hw_info;
    arguments.scheduler = {split_k, StreamKMode::SplitK};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_op.run());

    compat::wait();

  } catch (std::exception &e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "Unexpected error" << std::endl;
    return -1;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// PUBLIC FUNCTION
////////////////////////////////////////////////////////////////////////////////

using SplitKDim = std::tuple<int, int, int>;
using GemmSplitKRunPtr = int (*)(const at::Tensor &A, const at::Tensor &B,
                                 at::Tensor &C, const int M, const int N,
                                 const int K, const int split_k);

#include GEMM_SPLITK_CONFIG_HEADER

auto gemm_splitk_kernel(const at::Tensor &A, const at::Tensor &B, at::Tensor &C,
                        const int M, const int N, const int K,
                        const int split_k) -> int {
  const SplitKDim test_case{M, N, K};

  for (auto const &kv : gemm_splitk_config) {
    if (test_case == kv.first) {
      return kv.second(A, B, C, M, N, K, split_k);
    }
  }

  return gemm_splitk_run<PvcGemmSplitKBF16BF16FP32_RRR_1>(A, B, C, M, N, K,
                                                          split_k);
}
