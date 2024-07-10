#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {

DPASAnalysis::DPASAnalysis(Operation *root) {
  if (auto m = dyn_cast<ModuleOp>(root))
    mod = m;
  else
    mod = root->getParentOfType<ModuleOp>();

  DeviceArch arch = getDeviceArch(mod);
  bool isLTS = mod->hasAttr(TritonIntelGPUDialect::getLTSAttrName());

  // Populate the maps.
  mod.walk([&](FunctionOpInterface funcOp) {
    auto it = funcToDotMap.find(funcOp);

    funcOp.walk([&](DotOp dotOp) {
      if (it != funcToDotMap.end())
        it->second.push_back(dotOp);
      else
        funcToDotMap[funcOp] = {dotOp};

      DPASEngineType dpasEngineType = (isLTS || arch == DeviceArch::UNKNOWN)
                                          ? DPASEngineType::NOT_APPLICABLE
                                          : DPASAnalysis::getDPASType(dotOp);
      dotToDPASEngineMap[dotOp] = dpasEngineType;

      // Only PVC supports TF32.
      if (dpasEngineType == DPASEngineType::FP32_FP32_TF32_TF32) {
        if (arch != DeviceArch::PVC ||
            dotOp.getInputPrecision() != InputPrecision::TF32)
          dotToDPASEngineMap[dotOp] = DPASEngineType::NOT_APPLICABLE;
      }
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
  for (const DotOp &dotOp : it->second) {
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
  DeviceArch arch = getDeviceArch(mod);
  if (threadsPerWarp == supportedThreadsPerWarp(arch))
    return Result::True;

  return Result::False;
}

unsigned DPASAnalysis::supportedThreadsPerWarp(DeviceArch arch) {
  switch (arch) {
  case DeviceArch::PVC:
    return 16;
  case DeviceArch::ATS:
    return 8;
  default:
    llvm_unreachable("Unexpected target architecture");
  }
}

DPASAnalysis::DPASEngineType DPASAnalysis::getDPASType(DotOp op) {
  // d = a * b + c
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  auto bTy = cast<RankedTensorType>(op.getB().getType());
  auto cTy = cast<RankedTensorType>(op.getC().getType());
  auto dTy = cast<RankedTensorType>(op.getD().getType());
  Type aElemTy = aTy.getElementType();
  Type bElemTy = bTy.getElementType();
  Type cElemTy = cTy.getElementType();
  Type dElemTy = dTy.getElementType();

  assert(cElemTy == dElemTy && "Unexpected element type mismatch");

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
      if (aElemTy.isFloat8E4M3FNUZ())
        return DPASEngineType::FP32_FP32_FP16_FP16;
    } else if (dElemTy.isF16()) {
      if (aElemTy.isF16())
        return DPASEngineType::FP16_FP16_FP16_FP16;
    } else if (dElemTy.isBF16()) {
      if (aElemTy.isBF16())
        return DPASEngineType::BF16_BF16_BF16_BF16;
    }
  }

  return DPASEngineType::NOT_APPLICABLE;
}

} // namespace mlir::triton::gpu::intel
