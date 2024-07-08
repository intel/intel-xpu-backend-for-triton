#include "intel/include/Analysis/DPAS.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton::gpu::intel {

DPASAnalysis::DPASAnalysis(FunctionOpInterface func)
    : mod(func->getParentOfType<mlir::ModuleOp>()) {
  DeviceArch arch = getDeviceArch(mod);

  // Populate the DPAS map.
  func.walk([&](DotOp dotOp) {
    dpasMap[dotOp] =
        (mod->hasAttr("triton_gpu.is_lts") || arch == DeviceArch::UNKNOWN)
            ? DPASEngineType::NOT_APPLICABLE
            : DPASAnalysis::getDPASType(dotOp);

    // Only PVC supports TF32.
    if (dpasMap[dotOp] == DPASEngineType::FP32_FP32_TF32_TF32) {
      if (arch != DeviceArch::PVC ||
          dotOp.getInputPrecision() != InputPrecision::TF32)
        dpasMap[dotOp] = DPASEngineType::NOT_APPLICABLE;
    }
  });
}

DPASAnalysis::Result DPASAnalysis::canUseDPAS() const {
  if (dpasMap.empty())
    return Result::False;

  // Ensure all dot operations can be lowered to DPAS instructions.
  if (llvm::any_of(dpasMap, [&](auto &entry) {
        return entry.second == DPASEngineType::NOT_APPLICABLE;
      }))
    return Result::False;

  // Verify whether the module has the correct number of threads per warp.
  // Note: if the module doesn't have the warp size attribute, return
  // Result::Maybe to allow the caller to set warp size.
  gpu::intel::DeviceArch arch = getDeviceArch(mod);
  Attribute threadsPerWarpAttr =
      mod->getDiscardableAttr("triton_gpu.threads-per-warp");

  if (!threadsPerWarpAttr)
    return Result::Maybe;

  unsigned threadsPerWarp = cast<IntegerAttr>(threadsPerWarpAttr).getInt();
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
    return 0;
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

  if (aElemTy != bElemTy || cElemTy != dElemTy)
    return DPASEngineType::NOT_APPLICABLE;

  if (dElemTy.isIntOrIndex()) {
    if (dElemTy.getIntOrFloatBitWidth() == 32) {
      if (aElemTy.getIntOrFloatBitWidth() == 8 &&
          bElemTy.getIntOrFloatBitWidth() == 8)
        return dElemTy.isSignedInteger() ? DPASEngineType::S32_S32_S8_S8
                                         : DPASEngineType::U32_U32_U8_U8;
    }
    return DPASEngineType::NOT_APPLICABLE;
  }

  if (isa<FloatType>(dElemTy)) {
    if (dElemTy.isF32()) {
      if (aElemTy.isF16() && bElemTy.isF16())
        return DPASEngineType::FP32_FP32_FP16_FP16;
      if (aElemTy.isBF16() && bElemTy.isBF16())
        return DPASEngineType::FP32_FP32_BF16_BF16;
      if (aElemTy.isF32() && bElemTy.isF32() &&
          op.getInputPrecision() == InputPrecision::TF32)
        return DPASEngineType::FP32_FP32_TF32_TF32;
      // For FP8XFP8->FP32, upcast to FP16
      if (aElemTy.isFloat8E5M2() && bElemTy.isFloat8E5M2())
        return DPASEngineType::FP32_FP32_FP16_FP16;
      if (aElemTy.isFloat8E4M3FNUZ() && bElemTy.isFloat8E4M3FNUZ())
        return DPASEngineType::FP32_FP32_FP16_FP16;
    } else if (dElemTy.isF16()) {
      if (aElemTy.isF16() && bElemTy.isF16())
        return DPASEngineType::FP16_FP16_FP16_FP16;
    } else if (dElemTy.isBF16()) {
      if (aElemTy.isBF16() && bElemTy.isBF16())
        return DPASEngineType::BF16_BF16_BF16_BF16;
    }
  }

  return DPASEngineType::NOT_APPLICABLE;
}

} // namespace mlir::triton::gpu::intel
