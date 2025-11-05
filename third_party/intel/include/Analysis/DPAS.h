#ifndef TRITON_INTEL_ANALYSIS_DPAS_H
#define TRITON_INTEL_ANALYSIS_DPAS_H

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu::intel {

//===----------------------------------------------------------------------===//
// Intel DPAS Analysis
//===----------------------------------------------------------------------===//

class DPASAnalysis {
public:
  explicit DPASAnalysis(Operation *root);

  enum class Result { True, False, Maybe };

  enum class DPASEngineType : uint8_t {
    // data types for operands D,C,A,B.
    FP32_FP32_FP16_FP16 = 0, // default
    FP32_FP32_BF16_BF16,
    FP32_FP32_TF32_TF32,
    FP16_FP16_FP16_FP16,
    BF16_BF16_FP8_FP8,
    BF16_BF16_BF16_BF16,
    U32_U32_U8_U8,
    S32_S32_S8_S8,
    // data types for dot scaled.
    FP32_FP32_BF16_FP8,
    FP32_FP32_BF16_FP4,
    FP32_FP32_FP16_FP8,
    FP32_FP32_FP16_FP4,
    FP32_FP32_FP8_BF16,
    FP32_FP32_FP8_FP16,
    FP32_FP32_FP8_FP8,
    FP32_FP32_FP8_FP4,
    FP32_FP32_FP4_BF16,
    FP32_FP32_FP4_FP16,
    FP32_FP32_FP4_FP8,
    FP32_FP32_FP4_FP4,
    NOT_APPLICABLE
  };

  /// Analyze the 'dotToDPASEngineMap' for the given function and return:
  ///  - Result::True if the function associated with this analysis contains
  ///     DotOp operations that can be lowered to DPAS instructions,
  ///  - Result::False if it contains DotOp operations that cannot be lowered
  ///    to DPAS instructions, and
  ///  - Result::Maybe if it contains DotOp operations that could be lowered to
  ///    DPAS instructions if the module was executed with a different subgroup
  ///    (aka threads per warp) size.
  Result canUseDPAS(FunctionOpInterface funcOp) const;

  /// Given a 'DotOp' or 'ScaledDot' operation, return its DPAS engine type.
  static DPASEngineType getDPASType(Operation *op);

  // clang-format off
  template <typename OpTy>
  typename std::enable_if<llvm::is_one_of<OpTy, DotOp, DotScaledOp>::value,
                          DPASAnalysis::DPASEngineType>::type
  static getDPASType(OpTy);
  // clang-format on

private:
  mlir::ModuleOp mod;

  /// Tracks Dot/DotScaled operations and their DPAS engine type.
  std::map<Operation *, DPASEngineType> dotToDPASEngineMap;

  /// Tracks the Dot/DotScaled operations contained in a function.
  std::map<FunctionOpInterface, SmallVector<Operation *>> funcToDotMap;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_DPAS_H
