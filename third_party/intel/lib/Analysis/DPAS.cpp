#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Casting.h"
#include <iostream>
#include <type_traits>

namespace mlir::triton::gpu::intel {

DPASAnalysis::DPASAnalysis(Operation *root) {
  if (auto m = dyn_cast<ModuleOp>(root))
    mod = m;
  else
    mod = root->getParentOfType<ModuleOp>();

  bool supportDPAS =
      mod->hasAttr(TritonIntelGPUDialect::getSupportDPASAttrName());

  // Populate the maps.
  mod.walk([&](FunctionOpInterface funcOp) {
    auto it = funcToDotMap.find(funcOp);

    funcOp.walk([&](Operation *op) {
      if (!isa<DotOp, DotScaledOp>(op))
        return;

      if (it != funcToDotMap.end())
        it->second.push_back(op);
      else
        funcToDotMap[funcOp] = {op};

      DPASEngineType dpasEngineType = supportDPAS
                                          ? DPASAnalysis::getDPASType(op)
                                          : DPASEngineType::NOT_APPLICABLE;
      if (dpasEngineType == DPASEngineType::FP32_FP32_TF32_TF32 &&
          cast<DotOp>(op).getInputPrecision() != InputPrecision::TF32)
        dpasEngineType = DPASEngineType::NOT_APPLICABLE;
      dotToDPASEngineMap[op] = dpasEngineType;
    });
  });
}

DPASAnalysis::Result
DPASAnalysis::canUseDPAS(FunctionOpInterface funcOp) const {
  if (funcToDotMap.empty() || dotToDPASEngineMap.empty())
    return Result::False;

  auto it = funcToDotMap.find(funcOp);
  if (it == funcToDotMap.end())
    return Result::False;

  // Ensure all dot operations in the function can be lowered to DPAS
  // instructions.
  for (Operation *dotOp : it->second) {
    DPASEngineType dpasEngineType = dotToDPASEngineMap.at(dotOp);
    if (dpasEngineType == DPASEngineType::NOT_APPLICABLE)
      return Result::False;
  }

  // Verify whether the module has the correct number of threads per warp.
  // Note: if the module doesn't then return 'Result::Maybe' to allow the caller
  // to set warp size.
  Attribute threadsPerWarpAttr =
      mod->getDiscardableAttr(TritonGPUDialect::getThreadsPerWarpAttrName());
  if (!threadsPerWarpAttr)
    return Result::Maybe;

  unsigned threadsPerWarp = cast<IntegerAttr>(threadsPerWarpAttr).getInt();
  unsigned minSGSize = mod->getAttrOfType<IntegerAttr>(
                              TritonIntelGPUDialect::getMinSGSizeAttrName())
                           .getInt();
  return (threadsPerWarp == minSGSize) ? Result::True : Result::False;
}

DPASAnalysis::DPASEngineType DPASAnalysis::getDPASType(Operation *op) {
  if (auto dotOp = dyn_cast<DotOp>(op))
    return DPASAnalysis::getDPASType(dotOp);
  if (auto dotScaledOp = dyn_cast<DotScaledOp>(op))
    return DPASAnalysis::getDPASType(dotScaledOp);
  return DPASEngineType::NOT_APPLICABLE;
}

// This function determines the DPAS engine type for the given operation.
// It checks the element types of the tensors involved in the operation
// and returns the appropriate DPAS engine type based on the type combinations.
template <typename OpTy>
typename std::enable_if<llvm::is_one_of<OpTy, DotOp, DotScaledOp>::value,
                        DPASAnalysis::DPASEngineType>::type
DPASAnalysis::getDPASType(OpTy op) {
  auto cTy = cast<RankedTensorType>(op.getC().getType());
  auto dTy = cast<RankedTensorType>(op.getD().getType());
  Type cElemTy = cTy.getElementType();
  Type dElemTy = dTy.getElementType();

  assert(cElemTy == dElemTy && "Unexpected element type mismatch");

  RankedTensorType aTy, bTy;
  Type aElemTy, bElemTy;

  if constexpr (std::is_same_v<OpTy, DotOp>) {
    // d = a * b + c
    aTy = cast<RankedTensorType>(op.getA().getType());
    bTy = cast<RankedTensorType>(op.getB().getType());
    aElemTy = aTy.getElementType();
    bElemTy = bTy.getElementType();

    if (aElemTy != bElemTy)
      return DPASEngineType::NOT_APPLICABLE;

    if (dElemTy.isIntOrIndex()) {
      if (dElemTy.getIntOrFloatBitWidth() == 32 &&
          aElemTy.getIntOrFloatBitWidth() == 8)
        return dElemTy.isSignedInteger() ? DPASEngineType::S32_S32_S8_S8
                                         : DPASEngineType::U32_U32_U8_U8;
      return DPASEngineType::NOT_APPLICABLE;
    }

    if (isa<FloatType>(dElemTy)) {
      if (dElemTy.isF32()) {
        if (aElemTy.isF16())
          return DPASEngineType::FP32_FP32_FP16_FP16;
        if (aElemTy.isBF16())
          return DPASEngineType::FP32_FP32_BF16_BF16;
        if (aElemTy.isF32() && op.getInputPrecision() == InputPrecision::TF32)
          return DPASEngineType::FP32_FP32_TF32_TF32;
        // For FP8XFP8->FP32, upcast to FP16
        if (aElemTy.isFloat8E5M2())
          return DPASEngineType::FP32_FP32_FP16_FP16;
        if (aElemTy.isFloat8E4M3FN())
          return DPASEngineType::FP32_FP32_FP16_FP16;
      } else if (dElemTy.isF16()) {
        if (aElemTy.isF16())
          return DPASEngineType::FP16_FP16_FP16_FP16;
      } else if (dElemTy.isBF16()) {
        if (aElemTy.isBF16())
          return DPASEngineType::BF16_BF16_BF16_BF16;
      }
    }
  }

  if constexpr (std::is_same_v<OpTy, DotScaledOp>) {
    aTy = cast<RankedTensorType>(op.getLhs().getType());
    bTy = cast<RankedTensorType>(op.getRhs().getType());
    aElemTy = aTy.getElementType();
    bElemTy = bTy.getElementType();

    if (isa<FloatType>(dElemTy)) {
      if (dElemTy.isF32()) {
        if (aElemTy.isBF16() &&
            (bElemTy.isFloat8E4M3FN() || bElemTy.isFloat8E5M2()))
          return DPASEngineType::FP32_FP32_BF16_FP8;
        // 2 E2M1 are packed into 1 int8
        if (aElemTy.isBF16() && bElemTy.isInteger(8))
          return DPASEngineType::FP32_FP32_BF16_FP4;
        if ((aElemTy.isFloat8E4M3FN() || aElemTy.isFloat8E5M2()) &&
            bElemTy.isBF16())
          return DPASEngineType::FP32_FP32_FP8_BF16;
        if ((aElemTy.isFloat8E4M3FN() || aElemTy.isFloat8E5M2()) &&
            (bElemTy.isFloat8E4M3FN() || bElemTy.isFloat8E5M2()))
          return DPASEngineType::FP32_FP32_FP8_FP8;
        if ((aElemTy.isFloat8E4M3FN() || aElemTy.isFloat8E5M2()) &&
            bElemTy.isInteger(8))
          return DPASEngineType::FP32_FP32_FP8_FP4;
        if (aElemTy.isInteger(8) && bElemTy.isBF16())
          return DPASEngineType::FP32_FP32_FP4_BF16;
        if (aElemTy.isInteger(8) &&
            (bElemTy.isFloat8E4M3FN() || bElemTy.isFloat8E5M2()))
          return DPASEngineType::FP32_FP32_FP4_FP8;
      }
    }
  }
  return DPASEngineType::NOT_APPLICABLE;
}

// Explicit instantiations.
template DPASAnalysis::DPASEngineType
DPASAnalysis::getDPASType<DotOp>(DotOp op);
template DPASAnalysis::DPASEngineType
DPASAnalysis::getDPASType<DotScaledOp>(DotScaledOp op);

} // namespace mlir::triton::gpu::intel
