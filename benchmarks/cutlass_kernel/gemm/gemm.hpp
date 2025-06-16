#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/thread/activation.h"

#include <exception>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// PRIVATE FUNCTION
////////////////////////////////////////////////////////////////////////////////

template <typename TileShape>
static auto gemm_run(const at::Tensor &A, const at::Tensor &B, at::Tensor &C,
                     const int M, const int N, const int K, const int L)
    -> int {
  RECORD_FUNCTION("cutlass gemm", {});

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::bfloat16_t;
  using ElementOutput = float;

  using LayoutA = typename cutlass::layout::RowMajor;
  using LayoutB = typename cutlass::layout::RowMajor;
  using LayoutC = typename cutlass::layout::RowMajor;
  using LayoutD = typename cutlass::layout::RowMajor;

  constexpr int AlignmentA = sizeof(ElementInputA);
  constexpr int AlignmentB = sizeof(ElementInputB);
  constexpr int AlignmentC = sizeof(ElementAccumulator);
  constexpr int AlignmentD = sizeof(ElementOutput);

  /// MAIN LOOP ///

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp, ElementInputA,
          LayoutA, AlignmentA, ElementInputB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape,
          cute::Shape<cute::_1, cute::_1, cute::_1>,
          cutlass::gemm::collective::StageCountAuto,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  /// EPILOGUE LOOP ///

  using EpilogueOp = typename cutlass::epilogue::fusion::LinCombEltAct<
      cutlass::epilogue::thread::ReLu, ElementOutput, ElementComputeEpilogue,
      ElementAccumulator, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp, TileShape,
          cute::Shape<cute::_1, cute::_1, cute::_1>,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
          LayoutC, AlignmentC, ElementOutput, LayoutD, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          EpilogueOp>::CollectiveOp;

  /// GEMM ///

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  /// GEMM INVOCATION ///

  try {
    using Gemm =
        typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    typename Gemm::Arguments arguments;

    /// Buffer Initialization
    const cutlass::bfloat16_t *_A =
        static_cast<const cutlass::bfloat16_t *>(A.data_ptr());
    const cutlass::bfloat16_t *_B =
        static_cast<const cutlass::bfloat16_t *>(B.data_ptr());
    float *_C = static_cast<float *>(C.data_ptr());

    /// Problem size
    using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
    ProblemShapeType problem_size = ProblemShapeType{M, N, K, L};

    /// Stride
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    StrideB stride_B =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    StrideC stride_C =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    StrideD stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    static cutlass::KernelHardwareInfo hw_info;
    if (hw_info.sm_count == 0) {
      hw_info.sm_count =
          cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
      CUTLASS_TRACE_HOST(
          "Query result for SM count per device: " << hw_info.sm_count);
    }

    arguments = {cutlass::gemm::GemmUniversalMode::kGemm,
                 problem_size,
                 {_A, stride_A, _B, stride_B},
                 {{ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
                  nullptr,
                  stride_C,
                  _C,
                  stride_D},
                 hw_info};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_op.run());

    syclcompat::wait();

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

using Dim = std::tuple<int, int, int, int>;
using GemmRunPtr = int (*)(const at::Tensor &A, const at::Tensor &B,
                           at::Tensor &C, const int M, const int N, const int K,
                           const int L);

/// Each entry associates a specific problem dimension to their corresponding
/// tile shape. For more details, see:
/// https://github.com/codeplaysoftware/cutlass-sycl/tree/sycl-develop/benchmarks

// clang-format off
static constexpr std::array<std::pair<Dim, GemmRunPtr>, 18> tile_map = {{
  { { 1,    1024, 8192,   28672 }, &gemm_run<cute::Shape<cute::_128, cute::_512, cute::_32>> },
  { { 32,   4096, 128,    4096  }, &gemm_run<cute::Shape<cute::_256, cute::_128, cute::_32>> },
  { { 4096, 8,    16384,  128   }, &gemm_run<cute::Shape<cute::_128, cute::_256, cute::_16>> },
  { { 4096, 8,    128,    16384 }, &gemm_run<cute::Shape<cute::_8,   cute::_128, cute::_32>> },
  { { 1,    1,    1024,   4096  }, &gemm_run<cute::Shape<cute::_8,   cute::_64,  cute::_32>> },
  { { 1,    1,    4096,   4096  }, &gemm_run<cute::Shape<cute::_8,   cute::_128, cute::_32>> },
  { { 1,    1,    6144,   4096  }, &gemm_run<cute::Shape<cute::_8,   cute::_128, cute::_32>> },
  { { 1,    1,    14336,  4096  }, &gemm_run<cute::Shape<cute::_64,  cute::_256, cute::_32>> },
  { { 1,    1,    28672,  4096  }, &gemm_run<cute::Shape<cute::_32,  cute::_128, cute::_32>> },
  { { 1,    1,    128256, 4096  }, &gemm_run<cute::Shape<cute::_32,  cute::_512, cute::_32>> },
  { { 1,    1,    4096,   14336 }, &gemm_run<cute::Shape<cute::_8,   cute::_128, cute::_32>> },
  { { 1,    8,    1024,   4096  }, &gemm_run<cute::Shape<cute::_8,   cute::_64,  cute::_32>> },
  { { 1,    8,    4096,   4096  }, &gemm_run<cute::Shape<cute::_256, cute::_256, cute::_32>> },
  { { 1,    8,    6144,   4096  }, &gemm_run<cute::Shape<cute::_256, cute::_256, cute::_32>> },
  { { 1,    8,    14336,  4096  }, &gemm_run<cute::Shape<cute::_64,  cute::_256, cute::_32>> },
  { { 1,    8,    28672,  4096  }, &gemm_run<cute::Shape<cute::_32,  cute::_128, cute::_32>> },
  { { 1,    8,    128256, 4096  }, &gemm_run<cute::Shape<cute::_32,  cute::_512, cute::_32>> },
  { { 1,    8,    4096,   14336 }, &gemm_run<cute::Shape<cute::_256, cute::_256, cute::_32>> },
}};
// clang-format on

auto gemm(const at::Tensor &A, const at::Tensor &B, at::Tensor &C, const int M,
          const int N, const int K, const int L) -> int {
  const Dim test_case{L, M, N, K};

  for (auto const &kv : tile_map) {
    if (test_case == kv.first) {
      return kv.second(A, B, C, M, N, K, L);
    }
  }

  return gemm_run<cute::Shape<cute::_256, cute::_256, cute::_32>>(A, B, C, M, N,
                                                                  K, L);
}
