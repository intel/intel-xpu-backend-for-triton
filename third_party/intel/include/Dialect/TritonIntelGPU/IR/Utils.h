//===- Utils.h - TritonIntelGPU Utils -----------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H
#define TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H

#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <triton/Tools/Sys/GetEnv.h>

namespace mlir::triton::gpu::intel {

// FIXME: temporary A/B scaffolding for issue #6719 (delete the Intel-specific
// ReduceOpToLLVM.cpp). Both knobs default to false, so an unset environment
// reproduces today's behaviour exactly. Strip together with the Intel pattern.

/// True if `tt.reduce` should be lowered by the common upstream pattern instead
/// of the Intel one. Must be read by the pattern registration site *and* the
/// scratch-allocation callback: the common lowering sizes shared memory with
/// `ReduceOpHelper::getScratchSizeInBytes()`, and mixing the two allocators
/// with the wrong lowering silently under-allocates.
inline bool useCommonReduceLowering() {
  return tools::getBoolEnv("TRITON_INTEL_REDUCE_USE_COMMON_LOWERING");
}

/// True if the within-thread combine should use upstream's tree reduction for
/// *all* types. Inverted sense: unset keeps the left fold that #6667/#6914
/// added for non-float and sub-32-bit-float reductions.
inline bool disableReduceLeftFold() {
  return tools::getBoolEnv("TRITON_INTEL_REDUCE_DISABLE_LEFT_FOLD");
}

/// Calculate the optimal number of elements per thread for a given operation
/// along an axis with greatest continuity.
inline unsigned getNumElementsPerThread(
    Operation *op, SmallVector<unsigned> order,
    mlir::triton::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  Value val = getMemAccessPtr(op);
  Type valTy = val.getType();
  auto ty = cast<RankedTensorType>(valTy);
  auto shapePerCTA = getShapePerCTA(ty);
  mlir::triton::AxisInfo &valInfo = *axisInfoAnalysis.getAxisInfo(val);

  unsigned elemNumBits = getElementBitWidth(ty);
  unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
  unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
  unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
  unsigned maxContig =
      std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
  unsigned alignment = std::min(maxMultiple, maxContig);
  return std::min(alignment, 128 / elemNumBits);
}

// Check if module's target arch is SPIRV. If there is no target arch
// attribute, then we assume SPIRV target by default.
inline bool hasSpirvTargetArch(Operation *op) {
  if (!isa<ModuleOp>(op))
    op = op->getParentOfType<ModuleOp>();
  auto arch = op->getAttrOfType<StringAttr>(
      triton::gpu::intel::TritonIntelGPUDialect::getTargetArchAttrName());
  return !arch || arch.str().substr(0, 4) == "spir";
}

inline LLVM::cconv::CConv getDefaultCConv(Operation *op) {
  if (hasSpirvTargetArch(op))
    return LLVM::cconv::CConv::SPIR_FUNC;
  llvm_unreachable("Unexpected target architecture");
}

inline LLVM::cconv::CConv getRequiredCConv(CallOpInterface callOp) {
  // If we call a function, return its calling convention.
  auto callable = callOp.resolveCallable();
  if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(callable))
    return funcOp.getCConv();
  return getDefaultCConv(callOp);
}
} // namespace mlir::triton::gpu::intel

#endif // TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H
