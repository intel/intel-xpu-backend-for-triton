//===- AccelerateMatmul.cpp - Lower tt.dot to Intel DPAS --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;
namespace {
using tt::DotOp;
using ttg::BlockedEncodingAttr;
using ttg::ConvertLayoutOp;
using ttg::DotOperandEncodingAttr;
using ttg::SliceEncodingAttr;
using ttgi::DeviceArch;
using ttgi::DpasEncodingAttr;

struct IntelDPASCapability {
  uint32_t systolicDepth;
  uint32_t repeatCount;
  uint32_t executionSize;
  uint32_t opsChanBitWidths;
};

static IntelDPASCapability caps[] = {
    [(uint32_t)DeviceArch::UNKNOWN] = {},

    [(uint32_t)DeviceArch::ATS] =
        {
            .systolicDepth = 8,
            .repeatCount = 8,
            .executionSize = 8,
            .opsChanBitWidths = 32,
        },

    [(uint32_t)DeviceArch::PVC] =
        {
            .systolicDepth = 8,
            .repeatCount = 8,
            .executionSize = 16,
            .opsChanBitWidths = 32,
        },
};

IntelDPASCapability getDPASCapability(DeviceArch arch) {
  return caps[(uint32_t)arch];
}

SmallVector<unsigned> getWarpsPerTile(tt::DotOp dotOp,
                                      struct IntelDPASCapability dpasCap,
                                      const ArrayRef<int64_t> shape,
                                      unsigned numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  // TODO: revisit this in flash attention.
  for (Operation *op : slices)
    if (isa<DotOp>(op) && (op != dotOp))
      return {numWarps, 1};

  SmallVector<unsigned> ret{1, 1};
  SmallVector<int64_t> shapePerWarp{dpasCap.repeatCount, dpasCap.executionSize};

  // Try to find a proper tiling shape for the dot operation.
  // It doubles the warp number in col or row in each time based on column to
  // width ratio.
  // By this, we can minimize the duplication of the dot operands A and B.
  uint32_t rowColRatio =
      ceil<uint32_t>(dpasCap.repeatCount, dpasCap.executionSize);
  uint32_t colRowRatio =
      ceil<uint32_t>(dpasCap.executionSize, dpasCap.repeatCount);
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / (shapePerWarp[0] * colRowRatio) / ret[0] >=
        shape[1] / (shapePerWarp[1] * rowColRatio) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

class BlockedToDPAS : public mlir::RewritePattern {
  DeviceArch arch;

public:
  BlockedToDPAS(mlir::MLIRContext *context, DeviceArch arch)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context),
        arch(arch) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    DotOp dotOp = cast<DotOp>(op);
    RankedTensorType oldRetType =
        dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        oldRetType.getEncoding().isa<DpasEncodingAttr>())
      return failure();

    if (!supportDPAS(dotOp, arch))
      return failure();

    // Create DPAS encoding for the given number of warps
    ArrayRef<int64_t> retShape = oldRetType.getShape();
    ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
    unsigned numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    RankedTensorType oldAType = a.getType().cast<RankedTensorType>();
    RankedTensorType oldBType = b.getType().cast<RankedTensorType>();

    IntelDPASCapability dpasCap = getDPASCapability(arch);
    unsigned dpasElemBitWidths =
        oldAType.getElementType().getIntOrFloatBitWidth();
    unsigned opsPerChan = dpasCap.opsChanBitWidths / dpasElemBitWidths;

    SmallVector<unsigned> warpsPerTile =
        getWarpsPerTile(dotOp, dpasCap, retShape, numWarps);

    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    DpasEncodingAttr dpasEnc = DpasEncodingAttr::get(
        oldRetType.getContext(), dpasCap.repeatCount, dpasCap.systolicDepth,
        dpasCap.executionSize, opsPerChan, warpsPerTile, threadsPerWarp);

    RankedTensorType newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), dpasEnc);

    // convert accumulator
    Value oldAcc = dotOp.getOperand(2);
    ConvertLayoutOp newAcc = rewriter.create<ttg::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);

    DotOperandEncodingAttr newAEncoding = ttg::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(), opsPerChan);
    DotOperandEncodingAttr newBEncoding = ttg::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(), opsPerChan);

    RankedTensorType newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    RankedTensorType newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    DotOp newDot = rewriter.create<DotOp>(dotOp.getLoc(), newRetType, a, b,
                                          newAcc, dotOp.getInputPrecision(),
                                          dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType,
                                                      newDot.getResult());
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  auto tensorPromotedType =
      operand.getType().cast<RankedTensorType>().cloneWith(std::nullopt,
                                                           promotedType);
  Type elemType = tensorPromotedType.getElementType();
  return llvm::TypeSwitch<Type, Value>(elemType)
      .Case<FloatType>([&](auto) {
        return builder.create<tt::FpToFpOp>(loc, tensorPromotedType, operand);
      })
      .Case<IntegerType>([&](auto) {
        unsigned tgtBitWidth = elemType.getIntOrFloatBitWidth(),
                 valBitWidth = operand.getType()
                                   .cast<RankedTensorType>()
                                   .getElementTypeBitWidth();
        Operation *castOp = (valBitWidth <= tgtBitWidth)
                                ? builder.create<arith::ExtSIOp>(
                                      loc, tensorPromotedType, operand)
                                : builder.create<arith::TruncIOp>(
                                      loc, tensorPromotedType, operand);
        return castOp->getResult(0);
      });
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod) {
  mod.walk([](tt::DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    DpasEncodingAttr dpasLayout =
        D.getType().getEncoding().dyn_cast<DpasEncodingAttr>();
    if (dpasLayout) {
      // No operands promotion because of DPAS using different layout
      // to pack the dot operands for different scalar type.
      return;
    } else {
      // FMA case.
      Type DElType = D.getType().getElementType();
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

namespace mlir {
#define GEN_PASS_DEF_TRITONINTELGPUACCELERATEMATMUL
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir

class TritonIntelGPUAccelerateMatmulPass
    : public impl::TritonIntelGPUAccelerateMatmulBase<
          TritonIntelGPUAccelerateMatmulPass> {
public:
  using impl::TritonIntelGPUAccelerateMatmulBase<
      TritonIntelGPUAccelerateMatmulPass>::TritonIntelGPUAccelerateMatmulBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<::BlockedToDPAS>(context, deviceArch);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    // now that we pick the scalar type decompose dot that are not natively
    // supported.
    decomposeMixedModeDotOp(m);
  }
};
