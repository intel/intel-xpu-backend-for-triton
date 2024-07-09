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
  DPASAnalysis(FunctionOpInterface func);

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

  /// Analyze the dpasMap and return:
  ///  - Result::True if the function associated with this analysis contains
  ///     DotOp operations that can be lowered to DPAS instructions,
  ///  - Result::False if it contains DotOp operations that cannot be lowered
  ///    to DPAS instructions, and
  ///  - Result::Maybe if it contains DotOp operations that could be lowered to
  ///    DPAS instructions if the module was executed with a different subgroup
  ///    (aka threads per warp) size.
  Result canUseDPAS() const;

  /// Return the threads per warp (aka subgroup size) supported by the DPAS
  /// instruction on the given device architecture.
  static unsigned supportedThreadsPerWarp(DeviceArch arch);

  /// Given a DotOp operation, return the DPAS engine type.
  static DPASEngineType getDPASType(DotOp op);

private:
  /// The module enclosing the function associated with the analysis.
  mlir::ModuleOp mod;

  /// The map of DotOp to DPAS type.
  std::map<DotOp, DPASEngineType> dpasMap;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_DPAS_H
