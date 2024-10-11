#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

#define PVC_2D_LOAD_MAXIMUM_NUMBER_OF_ROWS 32
#define PVC_2D_LOAD_MAXIMUM_BYTES_OF_COLS 64

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using DPASAnalysis = intel::DPASAnalysis;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUACCELERATEMATMUL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

struct IntelDPASCapability {
  uint32_t systolicDepth;
  uint32_t repeatCount;
  uint32_t executionSize;
  uint32_t opsChanBitWidths;
};

IntelDPASCapability getDPASCapability(unsigned minSGSize) {
  switch (minSGSize) {
  case 8: {
    IntelDPASCapability cap;
    cap.systolicDepth = 8;
    cap.repeatCount = 8;
    cap.executionSize = 8;
    cap.opsChanBitWidths = 32;
    return cap;
  }
  case 16: {
    IntelDPASCapability cap;
    cap.systolicDepth = 8;
    cap.repeatCount = 8;
    cap.executionSize = 16;
    cap.opsChanBitWidths = 32;
    return cap;
  }
  default:
    return IntelDPASCapability();
  }
}

SmallVector<unsigned> getWarpsPerTile(DotOp dotOp,
                                      struct IntelDPASCapability dpasCap,
                                      const ArrayRef<int64_t> shape,
                                      unsigned numWarps) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };

  SetVector<Operation *> slices = getSlice(dotOp, {filter});
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
      if (ret[0] < shape[0] / shapePerWarp[0])
        ret[0] *= 2;
      else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);

  return ret;
}

class BlockedToDPAS : public RewritePattern {
  const DPASAnalysis &dpasAnalysis;

public:
  BlockedToDPAS(MLIRContext *context, const DPASAnalysis &dpasAnalysis)
      : RewritePattern(DotOp::getOperationName(), 2, context),
        dpasAnalysis(dpasAnalysis) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    DotOp dotOp = cast<DotOp>(op);
    RankedTensorType oldRetType =
        cast<RankedTensorType>(dotOp.getResult().getType());
    if (!oldRetType.getEncoding() ||
        isa<intel::DpasEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    if (dpasAnalysis.canUseDPAS(funcOp) != DPASAnalysis::Result::True)
      return failure();

    // Create DPAS encoding for the given number of warps
    ArrayRef<int64_t> retShape = oldRetType.getShape();
    size_t rank = retShape.size();
    ModuleOp mod = funcOp->getParentOfType<ModuleOp>();
    unsigned numWarps = TritonGPUDialect::getNumWarps(mod);

    Value a = dotOp.getA();
    Value b = dotOp.getB();
    RankedTensorType oldAType = cast<RankedTensorType>(a.getType());
    RankedTensorType oldBType = cast<RankedTensorType>(b.getType());

    unsigned minSGSize =
        mod->getAttrOfType<IntegerAttr>(
               intel::TritonIntelGPUDialect::getMinSGSizeAttrName())
            .getInt();
    IntelDPASCapability dpasCap = getDPASCapability(minSGSize);
    unsigned dpasElemBitWidths =
        oldAType.getElementType().getIntOrFloatBitWidth();

    // We are upcasting FP8 to FP16
    if (oldAType.getElementType().isFloat8E5M2() ||
        oldAType.getElementType().isFloat8E4M3FN())
      dpasElemBitWidths = 2 * dpasElemBitWidths;

    unsigned opsPerChan = dpasCap.opsChanBitWidths / dpasElemBitWidths;
    SmallVector<unsigned> warpsPerTile =
        getWarpsPerTile(dotOp, dpasCap, retShape, numWarps);
    SmallVector<unsigned> repCluster(rank, 1);

    unsigned threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    auto dpasEnc = intel::DpasEncodingAttr::get(
        oldRetType.getContext(), dpasCap.repeatCount, dpasCap.systolicDepth,
        dpasCap.executionSize, opsPerChan, warpsPerTile, repCluster,
        threadsPerWarp);

    if (dpasCap.executionSize == 16 /* PVC */) {
      // Enlarge the repCluster size to use the large 2D load for A and B
      // operands.
      unsigned maxRepClusterM =
          PVC_2D_LOAD_MAXIMUM_NUMBER_OF_ROWS / dpasCap.repeatCount;
      SmallVector<int64_t> repA =
          dpasEnc.getDPASRepetitions(oldAType.getShape(), 0);
      unsigned repClusterDimM =
          std::min(maxRepClusterM, static_cast<unsigned>(repA[0]));

      unsigned maxRepClusterN =
          PVC_2D_LOAD_MAXIMUM_BYTES_OF_COLS /
          ((dpasElemBitWidths / 8) * dpasCap.executionSize);
      SmallVector<int64_t> repB =
          dpasEnc.getDPASRepetitions(oldBType.getShape(), 1);
      unsigned repClusterDimN =
          std::min(maxRepClusterN, static_cast<unsigned>(repB[1]));
      if (rank == 3)
        repCluster[0] = 1;
      repCluster[rank - 2] = repClusterDimM;
      repCluster[rank - 1] = repClusterDimN;

      dpasEnc = intel::DpasEncodingAttr::get(
          oldRetType.getContext(), dpasCap.repeatCount, dpasCap.systolicDepth,
          dpasCap.executionSize, opsPerChan, warpsPerTile, repCluster,
          threadsPerWarp);
    }

    RankedTensorType newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), dpasEnc);

    // convert accumulator
    Value oldAcc = dotOp.getC();
    ConvertLayoutOp newAcc =
        rewriter.create<ConvertLayoutOp>(oldAcc.getLoc(), newRetType, oldAcc);

    DotOperandEncodingAttr newAEncoding = DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(), opsPerChan);
    DotOperandEncodingAttr newBEncoding = DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(), opsPerChan);

    RankedTensorType newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    RankedTensorType newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ConvertLayoutOp>(b.getLoc(), newBType, b);
    DotOp newDot = rewriter.create<DotOp>(dotOp.getLoc(), newRetType, a, b,
                                          newAcc, dotOp.getInputPrecision(),
                                          dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, oldRetType,
                                                 newDot.getResult());
    return success();
  }
};

} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  auto tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  Type elemType = tensorPromotedType.getElementType();

  return llvm::TypeSwitch<Type, Value>(elemType)
      .Case<FloatType>([&](auto) {
        return builder.create<FpToFpOp>(loc, tensorPromotedType, operand);
      })
      .Case<IntegerType>([&](auto) {
        unsigned tgtBitWidth = elemType.getIntOrFloatBitWidth(),
                 valBitWidth = cast<RankedTensorType>(operand.getType())
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
  mod.walk([](DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    auto dpasLayout =
        dyn_cast<intel::DpasEncodingAttr>(D.getType().getEncoding());

    Type promoteType;
    if (dpasLayout) {
      bool isNativeFP8 = AElType.isFloat8E5M2() || AElType.isFloat8E4M3FN();
      // fp8 is not natively supported by the the DPAS instruction, promote it
      // to fp16.
      if (!isNativeFP8)
        return;
      promoteType = builder.getF16Type();
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

class TritonIntelGPUAccelerateMatmulPass
    : public triton::gpu::intel::impl::TritonIntelGPUAccelerateMatmulBase<
          TritonIntelGPUAccelerateMatmulPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUAccelerateMatmulBase<
      TritonIntelGPUAccelerateMatmulPass>::TritonIntelGPUAccelerateMatmulBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    DPASAnalysis &dpasAnalysis = getAnalysis<DPASAnalysis>();

    RewritePatternSet patterns(context);
    patterns.add<BlockedToDPAS>(context, dpasAnalysis);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    // now that we pick the scalar type decompose dot that are not natively
    // supported.
    decomposeMixedModeDotOp(m);
  }
};
