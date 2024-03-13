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
namespace ttig = mlir::triton::gpu::intel;
namespace {
using tt::DotOp;
using ttg::BlockedEncodingAttr;
using ttg::ConvertLayoutOp;
using ttg::DotOperandEncodingAttr;
using ttg::SliceEncodingAttr;
using ttig::DeviceArch;
using ttig::DpasEncodingAttr;

struct IntelDPASCapability {
  uint32_t systolicDepth;
  uint32_t repeatCount;
  uint32_t executionSize;
  uint32_t opsChanBitWidths;
};

static IntelDPASCapability caps[] = {
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
  assert(arch <= DeviceArch::UNKNOWN && "Unknown Intel GPU archs");
  return caps[(uint32_t)arch];
}

SmallVector<unsigned, 2> getWarpsPerTile(tt::DotOp dotOp,
                                         struct IntelDPASCapability dpasCap,
                                         const ArrayRef<int64_t> shape,
                                         int numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  // TODO: revisit this in flash attention.
  for (Operation *op : slices)
    if (isa<DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {dpasCap.repeatCount,
                                          dpasCap.executionSize};
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
    auto dotOp = cast<DotOp>(op);
    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        oldRetType.getEncoding().isa<DpasEncodingAttr>())
      return failure();

    if (!supportDPAS(dotOp, arch))
      return failure();

    // Create DPAS encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();

    auto dpasCap = getDPASCapability(arch);
    unsigned dpasElemBitWidths =
        oldAType.getElementType().getIntOrFloatBitWidth();
    unsigned opsPerChan = dpasCap.opsChanBitWidths / dpasElemBitWidths;

    auto warpsPerTile = getWarpsPerTile(dotOp, dpasCap, retShape, numWarps);

    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    DpasEncodingAttr dpasEnc = DpasEncodingAttr::get(
        oldRetType.getContext(), dpasCap.repeatCount, dpasCap.systolicDepth,
        dpasCap.executionSize, opsPerChan, warpsPerTile, threadsPerWarp);

    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), dpasEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(oldAcc.getLoc(),
                                                        newRetType, oldAcc);

    auto newAEncoding = ttg::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(), opsPerChan);
    auto newBEncoding = ttg::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(), opsPerChan);

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<DotOp>(dotOp.getLoc(), newRetType, a, b,
                                         newAcc, dotOp.getAllowTF32(),
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

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

class TritonIntelGPUAccelerateMatmulPass
    : public TritonIntelGPUAccelerateMatmulBase<
          TritonIntelGPUAccelerateMatmulPass> {
public:
  TritonIntelGPUAccelerateMatmulPass() = default;
  TritonIntelGPUAccelerateMatmulPass(ttig::DeviceArch arch) {
    this->deviceArch = arch;
  }
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

std::unique_ptr<Pass>
mlir::createTritonIntelGPUAccelerateMatmulPass(ttig::DeviceArch arch) {
  return std::make_unique<TritonIntelGPUAccelerateMatmulPass>(arch);
}
