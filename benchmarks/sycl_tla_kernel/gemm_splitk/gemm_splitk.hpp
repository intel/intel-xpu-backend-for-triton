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

// The split-K and stream-K references share one kernel: the Xe stream-K tile
// scheduler, which `mode` drives into either decomposition. `split_k` is only
// read in SplitK mode.
using StreamKMode = cutlass::gemm::kernel::detail::
    PersistentTileSchedulerXeStreamKParams::DecompositionMode;

template <typename GemmConfig>
static auto gemm_splitk_run(const at::Tensor &A, const at::Tensor &B,
                            at::Tensor &C, const int M, const int N,
                            const int K, const int split_k,
                            const StreamKMode mode) -> int {
  RECORD_FUNCTION(mode == StreamKMode::StreamK ? "sycl-tla gemm_streamk"
                                               : "sycl-tla gemm_splitk",
                  {});

  using ElementComputeEpilogue = float;
  using Gemm = typename GemmConfig::Gemm;

  if (mode == StreamKMode::StreamK) {
    // The scheduler silently falls back to a plain data-parallel decomposition
    // when the stream-K tiles do not hold min_iters_per_sk_unit_ k iterations
    // between them (k_tiles * sk_tiles < min_iters_per_sk_unit_). This
    // reference requires that many k tiles on its own, which is stricter than
    // the scheduler needs (in StreamK mode sk_tiles >= min(output_tiles,
    // wgs_per_wave), so a small K still runs stream-K once there are a few
    // output tiles) but rules the fallback out without replicating the
    // scheduler's wave arithmetic. Fail loudly instead of measuring a
    // data-parallel kernel as stream-K. Raised outside the try below so that
    // it reaches Python as an exception.
    using Params =
        cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams;
    constexpr int tile_k =
        cute::size<2>(typename Gemm::GemmKernel::TileShape{});
    const int k_tiles = (K + tile_k - 1) / tile_k;
    TORCH_CHECK(k_tiles >= static_cast<int>(Params::min_iters_per_sk_unit_),
                "sycl-tla gemm_streamk: K=", K, " spans only ", k_tiles,
                " k tiles of ", tile_k, "; this reference requires at least ",
                Params::min_iters_per_sk_unit_,
                " so that the scheduler cannot fall back to a data-parallel "
                "decomposition");
  }

  try {
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
    arguments.scheduler = {split_k, mode};

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
                                 const int K, const int split_k,
                                 const StreamKMode mode);

// Split-K and stream-K each have their own shape table (input_gemm_splitk.in
// and input_gemm_streamk.in), so that tuning one never retargets the other.
#include GEMM_SPLITK_CONFIG_HEADER
#include GEMM_STREAMK_CONFIG_HEADER

template <typename Table>
static auto gemm_splitk_dispatch(const Table &table, GemmSplitKRunPtr fallback,
                                 const at::Tensor &A, const at::Tensor &B,
                                 at::Tensor &C, const int M, const int N,
                                 const int K, const int split_k,
                                 const StreamKMode mode) -> int {
  const SplitKDim test_case{M, N, K};

  for (auto const &kv : table) {
    if (test_case == kv.first) {
      return kv.second(A, B, C, M, N, K, split_k, mode);
    }
  }

  return fallback(A, B, C, M, N, K, split_k, mode);
}

auto gemm_splitk_kernel(const at::Tensor &A, const at::Tensor &B, at::Tensor &C,
                        const int M, const int N, const int K,
                        const int split_k) -> int {
  return gemm_splitk_dispatch(gemm_splitk_config,
                              &gemm_splitk_run<PvcGemmSplitKBF16BF16FP32_RRR_1>,
                              A, B, C, M, N, K, split_k, StreamKMode::SplitK);
}

// Stream-K decomposition: the scheduler hands the last full wave of output
// tiles plus the partial one (every tile, when there are fewer than two full
// waves) to a single persistent wave that splits their K iterations, and runs
// the remaining full waves data-parallel. `splits` is not read in this mode;
// 1 matches SYCL-TLA's own stream-K benchmark config.
auto gemm_streamk_kernel(const at::Tensor &A, const at::Tensor &B,
                         at::Tensor &C, const int M, const int N, const int K)
    -> int {
  return gemm_splitk_dispatch(
      gemm_streamk_config, &gemm_splitk_run<PvcGemmStreamKBF16BF16FP32_RRR_1>,
      A, B, C, M, N, K, /*split_k=*/1, StreamKMode::StreamK);
}
