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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <triton/Tools/Sys/GetEnv.h>

namespace mlir::triton::gpu::intel {

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

/// Whether to lower `tt.scan` to the hardware sub-group scan builtin.
inline bool isSubgroupScanEnabled() {
  return tools::isEnvValueBool(tools::getStrEnv("TRITON_INTEL_SUBGROUP_SCAN"))
      .value_or(true);
}

/// Match a combine region consisting of a single binary operation applied to
/// the region's two block arguments, and return that operation.
///
/// Purely structural: carries no reduce- or scan-specific policy, so both
/// `warpReduce` and `warpScan` can share it without inheriting each other's
/// restrictions.
inline FailureOr<Operation *> matchSingleBinaryCombine(Region &combine) {
  if (!combine.hasOneBlock())
    return failure();
  Block &block = *combine.begin();
  Operation *yield = block.getTerminator();
  if (yield->getNumOperands() != 1)
    return failure();
  Operation *binOp = yield->getOperand(0).getDefiningOp();
  if (!binOp || binOp->getNumOperands() != 2 || binOp->getNumResults() != 1)
    return failure();
  if (binOp->getOperand(0) != block.getArgument(0) ||
      binOp->getOperand(1) != block.getArgument(1))
    return failure();
  return binOp;
}

/// Whether `op` can be lowered to a sub-group scan builtin, ignoring layout.
/// Returns the combine operation to scan with.
///
/// This is the layout-independent half of the legality gate; callers must still
/// check that the scan axis covers every lane of the sub-group, since
/// `InclusiveScan` scans the whole sub-group all-or-nothing.
inline FailureOr<Operation *> matchEligibleSubgroupScan(triton::ScanOp op) {
  // Tuple scans would need one builtin per component. Deliberately deferred.
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return failure();

  FailureOr<Operation *> combineOp =
      matchSingleBinaryCombine(op.getCombineOp());
  if (failed(combineOp))
    return failure();

  if (!isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
           arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
           arith::AndIOp, arith::OrIOp, arith::XOrIOp>(*combineOp))
    return failure();

  // `arith.maxnumf`/`minnumf` return the numeric operand when exactly one
  // operand is NaN, but SPIR-V leaves the group FMax/FMin choice undefined
  // there (signed zeros likewise). Excluded rather than inherited from the
  // reduce path: cumsum/cumprod do not need them.
  //
  // For i1, add/maxsi/minsi are not enabled for the scan builtin yet.
  Type resultType = (*combineOp)->getResult(0).getType();
  if (resultType.isInteger(1) &&
      isa<arith::AddIOp, arith::MaxSIOp, arith::MinSIOp>(*combineOp))
    return failure();

  return combineOp;
}
} // namespace mlir::triton::gpu::intel

#endif // TRITON_DIALECT_TRITON_INTEL_GPU_IR_UTILS_H
