#ifndef TRITON_INTEL_ANALYSIS_DPAS_TPP
#define TRITON_INTEL_ANALYSIS_DPAS_TPP

#include "Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <triton/Tools/Sys/GetEnv.hpp>

namespace mlir::triton::gpu::intel {

template <typename DPASEngineType, typename Enable>
DPASAnalysis<DPASEngineType, Enable>::DPASAnalysis(Operation *root) {
  if (auto m = dyn_cast<ModuleOp>(root))
    mod = m;
  else
    mod = root->getParentOfType<ModuleOp>();

  bool supportDPAS =
      mod->hasAttr(TritonIntelGPUDialect::getSupportDPASAttrName());

  // Populate the maps.
  mod.walk([&](FunctionOpInterface funcOp) {
    if (funcToDotMap.find(funcOp) == funcToDotMap.end())
      funcToDotMap[funcOp] = {};

    auto it = funcToDotMap.find(funcOp);

    funcOp.walk([&](Operation *op) {
      if (!isa<DotOp, DotScaledOp>(op))
        return;

      it->second.push_back(op);

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

template <typename DPASEngineType, typename Enable>
DPASAnalysisResult DPASAnalysis<DPASEngineType, Enable>::canUseDPAS(
    FunctionOpInterface funcOp) const {
  if (funcToDotMap.empty() || dotToDPASEngineMap.empty())
    return DPASAnalysisResult::False;

  auto it = funcToDotMap.find(funcOp);
  if (it == funcToDotMap.end())
    return DPASAnalysisResult::False;

  // Ensure all dot operations in the function can be lowered to DPAS
  // instructions.
  for (Operation *op : it->second) {
    auto it = dotToDPASEngineMap.find(op);
    if (it == dotToDPASEngineMap.end()) {
      llvm::errs() << "DPASAnalysis: Operation not found in map.\n";
      return DPASAnalysisResult::False;
    }

    DPASEngineType dpasEngineType = dotToDPASEngineMap.at(op);
    if (dpasEngineType == DPASEngineType::NOT_APPLICABLE)
      return DPASAnalysisResult::False;
  }

  // Verify whether the module has the correct number of threads per warp.
  // Note: if the module doesn't then return 'Result::Maybe' to allow the caller
  // to set warp size.
  Attribute threadsPerWarpAttr = mod->getDiscardableAttr(AttrNumThreadsPerWarp);
  if (!threadsPerWarpAttr)
    return DPASAnalysisResult::Maybe;

  unsigned threadsPerWarp = cast<IntegerAttr>(threadsPerWarpAttr).getInt();
  unsigned minSGSize = mod->getAttrOfType<IntegerAttr>(
                              TritonIntelGPUDialect::getMinSGSizeAttrName())
                           .getInt();

  assert(minSGSize == 8 || minSGSize == 16 ||
         minSGSize == 32 && "Unexpected minimum subgroup size");

  if (minSGSize != 8) {
    // We can support threads_per_warp=16 or 32 on Xe+ and later architectures.
    return (threadsPerWarp == 16 || threadsPerWarp == 32)
               ? DPASAnalysisResult::True
               : DPASAnalysisResult::False;
  }

  return (threadsPerWarp == minSGSize) ? DPASAnalysisResult::True
                                       : DPASAnalysisResult::False;
}

template <typename DPASEngineType, typename Enable>
DPASEngineType
DPASAnalysis<DPASEngineType, Enable>::getDPASType(Operation *op) {
  if (auto dotOp = dyn_cast<DotOp>(op))
    return DPASAnalysis::getDPASType(dotOp);
  if (auto dotScaledOp = dyn_cast<DotScaledOp>(op))
    return DPASAnalysis::getDPASType(dotScaledOp);
  return DPASEngineType::NOT_APPLICABLE;
}

// This function determines the DPAS engine type for the given operation.
// It checks the element types of the tensors involved in the operation
// and returns the appropriate DPAS engine type based on the type combinations.
template <typename DPASEngineType, typename Enable>
template <typename OpTy>
typename std::enable_if<llvm::is_one_of<OpTy, DotOp, DotScaledOp>::value,
                        DPASEngineType>::type
DPASAnalysis<DPASEngineType, Enable>::getDPASType(OpTy op) {
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

    auto m = op->template getParentOfType<ModuleOp>();
    bool isFp8Supported =
        m->hasAttr(TritonIntelGPUDialect::getSupportDPASWithBF8AttrName());

    if (isa<FloatType>(dElemTy)) {
      if (dElemTy.isF32()) {
        if (aElemTy.isF16())
          return DPASEngineType::FP32_FP32_FP16_FP16;
        if (aElemTy.isBF16())
          return DPASEngineType::FP32_FP32_BF16_BF16;
        if (aElemTy.isF32() && op.getInputPrecision() == InputPrecision::TF32)
          return DPASEngineType::FP32_FP32_TF32_TF32;

        // For FP8XFP8->FP32, upcast to FP16 when fp8 DPAS is not supported
        if (isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy)) {
          if (!isFp8Supported)
            return DPASEngineType::FP32_FP32_FP16_FP16;
          return DPASEngineType::FP32_FP32_FP8_FP8;
        }
      } else if (dElemTy.isF16()) {
        if (aElemTy.isF16())
          return DPASEngineType::FP16_FP16_FP16_FP16;
      } else if (dElemTy.isBF16()) {
        if (aElemTy.isBF16())
          return DPASEngineType::BF16_BF16_BF16_BF16;
        if constexpr (std::is_same<DPASEngineType, DPASEngineTypeXe3P>::value)
          if (isFp8Supported && isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy))
            return DPASEngineType::BF16_BF16_FP8_FP8;
      }
    }
  }

  if constexpr (std::is_same_v<OpTy, DotScaledOp>) {
    aTy = cast<RankedTensorType>(op.getA().getType());
    bTy = cast<RankedTensorType>(op.getB().getType());
    aElemTy = aTy.getElementType();
    bElemTy = bTy.getElementType();

    if (isa<FloatType>(dElemTy)) {
      if (dElemTy.isF32()) {
        if (aElemTy.isBF16() && bElemTy.isBF16())
          return DPASEngineType::FP32_FP32_BF16_BF16;
        if (aElemTy.isBF16() && isa<Float8E4M3FNType, Float8E5M2Type>(bElemTy))
          return DPASEngineType::FP32_FP32_BF16_FP8;
        if (aElemTy.isBF16() && bElemTy.isInteger(8))
          return DPASEngineType::FP32_FP32_BF16_FP4;
        if (isa<Float8E4M3FNType, Float8E5M2Type>(aElemTy) && bElemTy.isBF16())
          return DPASEngineType::FP32_FP32_FP8_BF16;
        if (aElemTy.isF16() && bElemTy.isF16())
          return DPASEngineType::FP32_FP32_FP16_FP16;
        if (aElemTy.isF16() && isa<Float8E4M3FNType, Float8E5M2Type>(bElemTy))
          return DPASEngineType::FP32_FP32_FP16_FP8;
        if (aElemTy.isF16() && bElemTy.isInteger(8))
          return DPASEngineType::FP32_FP32_FP16_FP4;
        if (isa<Float8E4M3FNType, Float8E5M2Type>(aElemTy) && bElemTy.isF16())
          return DPASEngineType::FP32_FP32_FP8_FP16;
        if (isa<Float8E4M3FNType, Float8E5M2Type>(aElemTy) &&
            isa<Float8E4M3FNType, Float8E5M2Type>(bElemTy))
          return DPASEngineType::FP32_FP32_FP8_FP8;
        if ((isa<Float8E4M3FNType>(aElemTy) || isa<Float8E5M2Type>(aElemTy)) &&
            bElemTy.isInteger(8))
          return DPASEngineType::FP32_FP32_FP8_FP4;
        if (aElemTy.isInteger(8) && bElemTy.isBF16())
          return DPASEngineType::FP32_FP32_FP4_BF16;
        if (aElemTy.isInteger(8) && bElemTy.isF16())
          return DPASEngineType::FP32_FP32_FP4_FP16;
        if (aElemTy.isInteger(8) &&
            isa<Float8E4M3FNType, Float8E5M2Type>(bElemTy))
          return DPASEngineType::FP32_FP32_FP4_FP8;
        if (aElemTy.isInteger(8) && bElemTy.isInteger(8))
          return DPASEngineType::FP32_FP32_FP4_FP4;
      }
    }
  }
  return DPASEngineType::NOT_APPLICABLE;
}

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_DPAS_TPP
