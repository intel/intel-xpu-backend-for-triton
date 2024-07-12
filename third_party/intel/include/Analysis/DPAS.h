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
    BF16_BF16_BF16_BF16,
    U32_U32_U8_U8,
    S32_S32_S8_S8,
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

  /// Given a DotOp operation, return its DPAS engine type.
  static DPASEngineType getDPASType(DotOp op);

private:
  mlir::ModuleOp mod;

  /// Tracks Dot operations and their DPAS engine type.
  std::map<DotOp, DPASEngineType> dotToDPASEngineMap;

  /// Tracks the Dot operations contained in a function.
  std::map<FunctionOpInterface, SmallVector<DotOp>> funcToDotMap;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_DPAS_H
