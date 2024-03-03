#include "triton/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

bool supportDPAS(DotOp op) {
  return getDPASType(op) != DPASEngineType::NOT_APPLICABLE;
}

DPASEngineType getDPASType(DotOp op) {
  // d = a * b + c
  auto aTy = op.getA().getType().cast<RankedTensorType>();
  auto bTy = op.getB().getType().cast<RankedTensorType>();
  auto cTy = op.getC().getType().cast<RankedTensorType>();
  auto dTy = op.getD().getType().cast<RankedTensorType>();

  if (aTy.getElementType() != bTy.getElementType() ||
      cTy.getElementType() != dTy.getElementType())
    return DPASEngineType::NOT_APPLICABLE;

  // TODO: add more dpas supported data type.
  if (dTy.getElementType().isIntOrIndex()) {
    // Integer
    if (dTy.getElementType().getIntOrFloatBitWidth() == 32) {
      if (aTy.getElementType().getIntOrFloatBitWidth() == 8 &&
          bTy.getElementType().getIntOrFloatBitWidth() == 8)
        return dTy.getElementType().isSignedInteger()
                   ? DPASEngineType::S32_S32_S8_S8
                   : DPASEngineType::U32_U32_U8_U8;
    }
  } else {
    // floating.
    if (dTy.getElementType().isF32()) {
      if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
        return DPASEngineType::FP32_FP32_FP16_FP16;
      if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
        return DPASEngineType::FP32_FP32_BF16_BF16;
      if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
          op.getAllowTF32())
        return DPASEngineType::FP32_FP32_TF32_TF32;
    } else if (dTy.getElementType().isF16()) {
      if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
        return DPASEngineType::FP16_FP16_FP16_FP16;
    } else if (dTy.getElementType().isBF16()) {
      if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
        return DPASEngineType::BF16_BF16_BF16_BF16;
    }
  }

  return DPASEngineType::NOT_APPLICABLE;
}

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
