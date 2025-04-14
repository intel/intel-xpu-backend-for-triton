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

////////////////////////////////////////////////////////////////////////////////
/// PRIVATE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_CHECK(status)                                                      \
{                                                                                  \
  cutlass::Status error = status;                                                  \
  if (error != cutlass::Status::kSuccess) {                                        \
    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \
        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \
    throw std::runtime_error(msg);                                                 \
  }                                                                                \
}

#define CHECK_XPU(x)                                                           \
  TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_XPU(x);                                                                \
  CHECK_CONTIGUOUS(x)

using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementInputA = cutlass::bfloat16_t;
using ElementInputB = cutlass::bfloat16_t;
using ElementOutput = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = sizeof(ElementInputA);
constexpr int AlignmentB = sizeof(ElementInputB);
constexpr int AlignmentC = sizeof(ElementAccumulator);
constexpr int AlignmentD = sizeof(ElementOutput);

using TileShape = cute::Shape<cute::_256, cute::_256, cute::_32>;

/// MAIN LOOP ///

/// @brief PVC Collective Builder
using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
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

/// @brief PVC Collective Builder
using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
  cutlass::epilogue::thread::ReLu,
  ElementOutput, ElementComputeEpilogue, ElementAccumulator,
  ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest
>;
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
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

/// @brief PVC Collective Builder
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  cute::Shape<int, int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue
>;

////////////////////////////////////////////////////////////////////////////////
/// PUBLIC FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

auto test_sycl() -> void {
  sycl::queue q;

  std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
}

auto gemm(
  const at::Tensor &A,
  const at::Tensor &B,
  at::Tensor &C,
  const int M,
  const int N,
  const int K,
  const int L
) -> int {

  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);
  RECORD_FUNCTION("cutlass gemm", {});

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  try {
    /// Parameter Initialization ///

    Gemm::Arguments arguments;

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
      {
        _A,
        stride_A,
        _B,
        stride_B,
      },
      {
        { ElementComputeEpilogue(1), ElementComputeEpilogue(0) },
        nullptr,
        stride_C,
        _C,
        stride_D
      },
      hw_info
    };
    // arguments.scheduler.max_swizzle_size = 4;

    /// Gemm Invocation ///

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
    }
    {
      auto status = gemm_op.initialize(arguments, workspace.get());
      CUTLASS_CHECK(status);
    }
    {
      auto status = gemm_op();
      CUTLASS_CHECK(status);
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

// auto gemm(
//   const at::Tensor &A,
//   const at::Tensor &B,
//   at::Tensor &C,
//   const int M,
//   const int N,
//   const int K,
//   const int L,
//   const int lda,
//   const int ldb,
//   const int ldc,
//   const int ldd
// ) -> int {

//   using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

//   try {
//     /// Parameter Initialization ///
//     using coord_t = cutlass::gemm::GemmCoord::Index;

//     static cutlass::KernelHardwareInfo hw_info;
//     if (hw_info.sm_count == 0) {
//       hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
//       CUTLASS_TRACE_HOST("Query result for SM count per device: " << hw_info.sm_count);
//     }

//     Gemm::Arguments arguments;

//     const cutlass::bfloat16_t *_A  = static_cast<const cutlass::bfloat16_t  *>(A.data_ptr());
//     const cutlass::bfloat16_t *_B  = static_cast<const cutlass::bfloat16_t  *>(B.data_ptr());
//     float *_C  = static_cast<float *>(C.data_ptr());

//     // arguments = {
//     //   /// Kernel Mode
//     //   cutlass::gemm::GemmUniversalMode::kGemm,
//     //   /// Problem Size
//     //   {
//     //     static_cast<coord_t>(M),
//     //     static_cast<coord_t>(N),
//     //     static_cast<coord_t>(K),
//     //     static_cast<coord_t>(L)
//     //   },
//     //   {
//     //     /// Block A
//     //     _A,
//     //     /// Stride A
//     //     {
//     //       lda,
//     //       cute::Int<1>{},
//     //       0
//     //     },
//     //     /// Block B
//     //     _B,
//     //     /// Stride B
//     //     {
//     //       cute::Int<1>{},
//     //       ldb,
//     //       0
//     //     },
//     //   },
//     //   {
//     //     /// Alpha and Beta
//     //     {
//     //       ElementComputeEpilogue(1),
//     //       ElementComputeEpilogue(0)
//     //     },
//     //     /// Block C
//     //     nullptr,
//     //     /// Stride C
//     //     {
//     //       cute::Int<1>{},
//     //       cute::Int<1>{},
//     //       0
//     //     },
//     //     /// Block D
//     //     _C,
//     //     /// Stride D
//     //     {
//     //       ldd,
//     //       cute::Int<1>{},
//     //       0
//     //     },
//     //   },
//     //   hw_info
//     // };
//     // arguments.scheduler.max_swizzle_size = 4;

//     /// Problem size
//     using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
//     ProblemShapeType problem_size = ProblemShapeType{M, N, K, L};

//     /// Stride
//     using StrideA = typename Gemm::GemmKernel::StrideA;
//     using StrideB = typename Gemm::GemmKernel::StrideB;
//     using StrideC = typename Gemm::GemmKernel::StrideC;
//     using StrideD = typename Gemm::GemmKernel::StrideD;
//     StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
//     StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
//     StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
//     StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

//     arguments = {
//       cutlass::gemm::GemmUniversalMode::kGemm,
//       problem_size,
//       {
//         _A,
//         stride_A,
//         _B,
//         stride_B,
//       },
//       {
//         /// Alpha and Beta
//         {
//           ElementComputeEpilogue(1),
//           ElementComputeEpilogue(0)
//         },
//         nullptr,
//         stride_C,
//         _C,
//         stride_D
//       },
//       hw_info
//     };
//     arguments.scheduler.max_swizzle_size = 4;

//     /// Gemm Invocation ///

//     Gemm gemm_op;

//     size_t workspace_size = Gemm::get_workspace_size(arguments);
//     cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

//     {
//     auto status = gemm_op.can_implement(arguments);
//     CUTLASS_CHECK(status);
//     }
//     {
//       auto status = gemm_op.initialize(arguments, workspace.get());
//       CUTLASS_CHECK(status);
//     }
//     {
//       auto status = gemm_op();
//       CUTLASS_CHECK(status);
//     }
//   }
//   catch (std::exception& e) {
//     std::cerr << "Runtime error: " << e.what() << std::endl;
//     return -1;
//   }
//   catch (...) {
//     return -1;
//   }

//   return 0;
// }

////////////////////////////////////////////////////////////////////////////////
/// MODULE
////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(cutlass_kernel, m) {
    // gemm
    m.def("test_sycl", &test_sycl, "test_sycl (CUTLASS)");
    m.def("gemm", &gemm, "gemm (CUTLASS)");
}