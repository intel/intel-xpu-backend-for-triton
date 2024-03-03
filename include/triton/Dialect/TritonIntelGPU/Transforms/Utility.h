#ifndef TRITON_DIALECT_TRITONINTELGPU_TRANSFORMS_UTILITY_H
#define TRITON_DIALECT_TRITONINTELGPU_TRANSFORMS_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

// data type for D_C_A_B.
enum class DPASEngineType : uint8_t {
  // floating-point XMX engine instr
  FP32_FP32_FP16_FP16 = 0, // default
  FP32_FP32_BF16_BF16,
  FP32_FP32_TF32_TF32,
  FP16_FP16_FP16_FP16,
  BF16_BF16_BF16_BF16,
  // integer XMX engine instr
  U32_U32_U8_U8,
  S32_S32_S8_S8,
  //
  NOT_APPLICABLE,
};

bool supportDPAS(DotOp op);
DPASEngineType getDPASType(DotOp op);

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONINTELGPU_TRANSFORMS_UTILITY_H
