#include <torch/extension.h>

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include <exception>
#include <iostream>

#define CUTLASS_CHECK(status)                                                      \
{                                                                                  \
  cutlass::Status error = status;                                                  \
  if (error != cutlass::Status::kSuccess) {                                        \
    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \
        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \
    throw std::runtime_error(msg);                                                 \
  }                                                                                \
}

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

template <typename TileShape>
auto gemm(
  const at::Tensor &A,
  const at::Tensor &B,
  at::Tensor &C,
  const int M,
  const int N,
  const int K,
  const int L
) -> int {
  RECORD_FUNCTION("cutlass gemm", {});

  /// MAIN LOOP ///

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelPVC, cutlass::arch::OpClassTensorOp,
    ElementInputA, LayoutA, AlignmentA,
    ElementInputB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  /// EPILOGUE LOOP ///

  using EpilogueOp = typename cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::ReLu,
    ElementOutput, ElementComputeEpilogue, ElementAccumulator,
    ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest
  >;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::IntelPVC, cutlass::arch::OpClassTensorOp,
    TileShape,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementComputeEpilogue, ElementAccumulator,
    ElementAccumulator, LayoutC, AlignmentC,
    ElementOutput,      LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EpilogueOp
  >::CollectiveOp;

  /// GEMM ///

  using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;

  try {
    using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    typename Gemm::Arguments arguments;

    /// Buffer Initialization
    const cutlass::bfloat16_t *_A  = static_cast<const cutlass::bfloat16_t  *>(A.data_ptr());
    const cutlass::bfloat16_t *_B  = static_cast<const cutlass::bfloat16_t  *>(B.data_ptr());
    float *_C  = static_cast<float *>(C.data_ptr());

    /// Problem size
    using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
    ProblemShapeType problem_size = ProblemShapeType{M, N, K, L};

    /// Stride
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    static cutlass::KernelHardwareInfo hw_info;
    if (hw_info.sm_count == 0) {
      hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
      CUTLASS_TRACE_HOST("Query result for SM count per device: " << hw_info.sm_count);
    }

    arguments = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {_A, stride_A, _B, stride_B},
      {
        { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },
        nullptr,
        stride_C,
        _C,
        stride_D
      },
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    {
      CUTLASS_CHECK(gemm_op.can_implement(arguments));
    }
    {
      CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    }
    {
      CUTLASS_CHECK(gemm_op.run());
    }
  }
  catch (std::exception& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
    return -1;
  }
  catch (...) {
    return -1;
  }

  return 0;
}

// See https://github.com/codeplaysoftware/cutlass-fork/blob/sycl-develop/benchmarks/pvc/input.in
using TileShape_RRR_1 = typename cute::Shape<cute::_256, cute::_256, cute::_32>;
using TileShape_RRR_2 = typename cute::Shape<cute::_128, cute::_512, cute::_32>;
using TileShape_RRR_3 = typename cute::Shape<cute::_256, cute::_128, cute::_32>;
using TileShape_RRR_4 = typename cute::Shape<cute::_128, cute::_256, cute::_16>;
using TileShape_RRR_5 = typename cute::Shape<cute::_8, cute::_128, cute::_32>;

PYBIND11_MODULE(cutlass_kernel, m) {
  m.def("gemm_256_256_32", &gemm<TileShape_RRR_1>, "gemm (CUTLASS)");
  m.def("gemm_128_512_32", &gemm<TileShape_RRR_2>, "gemm (CUTLASS)");
  m.def("gemm_256_128_32", &gemm<TileShape_RRR_3>, "gemm (CUTLASS)");
  m.def("gemm_128_256_16", &gemm<TileShape_RRR_4>, "gemm (CUTLASS)");
  /// FIXME: Getting a compile time error for RRR_5
  // m.def("gemm_8_128_32", &gemm<TileShape_RRR_5>, "gemm (CUTLASS)");
}
