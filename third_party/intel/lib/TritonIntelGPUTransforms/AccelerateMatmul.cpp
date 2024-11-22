#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

#define PVC_2D_LOAD_MAXIMUM_NUMBER_OF_ROWS 32
#define PVC_2D_LOAD_MAXIMUM_BYTES_OF_COLS 64

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUACCELERATEMATMUL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

SmallVector<unsigned>
getWarpsPerTile(tt::DotOp dotOp,
                ttg::intel::DpasEncodingAttr::DPASCapability dpasCap,
                const ArrayRef<int64_t> shape, unsigned numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };

  SetVector<Operation *> slices = getSlice(dotOp, {filter});
  // TODO: revisit this in flash attention.
  for (Operation *op : slices)
    if (isa<tt::DotOp>(op) && (op != dotOp))
      return {numWarps, 1};

  size_t rank = shape.size();
  SmallVector<unsigned> ret(rank, 1);

  if (rank == 3) {
    int batchWarp = numWarps;
    while (batchWarp > shape[0])
      batchWarp /= 2;
    ret[0] = batchWarp;
    numWarps /= batchWarp;
  }

  // Try to find a proper tiling shape for the dot operation.
  // It doubles the warp number in col or row in each time based on column to
  // width ratio.
  // By this, we can minimize the duplication of the dot operands A and B.
  SmallVector<int64_t> shapePerWarp{dpasCap.repeatCount, dpasCap.executionSize};
  uint32_t rowColRatio =
      ceil<uint32_t>(dpasCap.repeatCount, dpasCap.executionSize);
  uint32_t colRowRatio =
      ceil<uint32_t>(dpasCap.executionSize, dpasCap.repeatCount);

  int rowDim = rank - 2, colDim = rank - 1;
  do {
    if (ret[rowDim] * ret[colDim] >= numWarps)
      break;
    if (shape[rowDim] / (shapePerWarp[0] * colRowRatio) / ret[rowDim] >=
        shape[colDim] / (shapePerWarp[1] * rowColRatio) / ret[colDim]) {
      if (ret[rowDim] < shape[rowDim] / shapePerWarp[0])
        ret[rowDim] *= 2;
      else
        ret[colDim] *= 2;
    } else {
      ret[colDim] *= 2;
    }
  } while (true);

  return ret;
}

class BlockedToDPAS : public OpRewritePattern<tt::DotOp> {
  const ttg::intel::DPASAnalysis &dpasAnalysis;
  using TensorValue = TypedValue<RankedTensorType>;

public:
  BlockedToDPAS(MLIRContext *context,
                const ttg::intel::DPASAnalysis &dpasAnalysis)
      : OpRewritePattern<tt::DotOp>(context), dpasAnalysis(dpasAnalysis) {}

  LogicalResult matchAndRewrite(tt::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        isa<ttg::intel::DpasEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    auto funcOp = dotOp->getParentOfType<FunctionOpInterface>();
    if (dpasAnalysis.canUseDPAS(funcOp) !=
        ttg::intel::DPASAnalysis::Result::True)
      return failure();

    // Create DPAS encoding for the given number of warps
    ArrayRef<int64_t> retShape = oldRetType.getShape();
    ModuleOp mod = funcOp->getParentOfType<ModuleOp>();
    unsigned numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    TensorValue a = dotOp.getA();
    TensorValue b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    auto dpasCap = ttg::intel::DpasEncodingAttr::getDPASCapability(mod);
    Type elemType = oldAType.getElementType();
    unsigned opsPerChan =
        ttg::intel::DpasEncodingAttr::getOpsPerChannel(dpasCap, elemType);
    SmallVector<unsigned> warpsPerTile =
        getWarpsPerTile(dotOp, dpasCap, retShape, numWarps);
    size_t rank = retShape.size();
    SmallVector<unsigned> repCluster(rank, 1);

    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    auto dpasEnc = ttg::intel::DpasEncodingAttr::get(
        oldRetType.getContext(), dpasCap.repeatCount, dpasCap.systolicDepth,
        dpasCap.executionSize, opsPerChan, warpsPerTile, repCluster,
        threadsPerWarp);

    if (dpasCap.executionSize == 16 /* PVC */) {
      unsigned dpasElemBitWidths =
          oldAType.getElementType().getIntOrFloatBitWidth();

      // We are upcasting FP8 to FP16
      if (oldAType.getElementType().isFloat8E5M2() ||
          oldAType.getElementType().isFloat8E4M3FN())
        dpasElemBitWidths = 2 * dpasElemBitWidths;

      // Enlarge the repCluster size to use the large 2D load for A and B
      // operands.
      unsigned maxRepClusterM =
          PVC_2D_LOAD_MAXIMUM_NUMBER_OF_ROWS / dpasCap.repeatCount;
      SmallVector<int64_t> repA =
          dpasEnc.getDPASRepetitions(oldAType.getShape(), 0);
      unsigned repClusterDimM =
          std::min(maxRepClusterM, static_cast<unsigned>(repA[1]));

      unsigned maxRepClusterN =
          PVC_2D_LOAD_MAXIMUM_BYTES_OF_COLS /
          ((dpasElemBitWidths / 8) * dpasCap.executionSize);
      SmallVector<int64_t> repB =
          dpasEnc.getDPASRepetitions(oldBType.getShape(), 1);
      unsigned repClusterDimN =
          std::min(maxRepClusterN, static_cast<unsigned>(repB[2]));
      repCluster[rank - 2] = repClusterDimM;
      repCluster[rank - 1] = repClusterDimN;

      dpasEnc = ttg::intel::DpasEncodingAttr::get(
          oldRetType.getContext(), dpasCap.repeatCount, dpasCap.systolicDepth,
          dpasCap.executionSize, opsPerChan, warpsPerTile, repCluster,
          threadsPerWarp);
    }

    RankedTensorType newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), dpasEnc);

    // convert accumulator
    TensorValue oldAcc = dotOp.getC();
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
    auto newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newRetType, a, b,
                                             newAcc, dotOp.getInputPrecision(),
                                             dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(dotOp, oldRetType,
                                                      newDot.getResult());
    return success();
  }
};

class DecomposeScaledBlocked : public OpRewritePattern<tt::DotScaledOp> {
  const ttg::intel::DPASAnalysis &dpasAnalysis;
  using TensorValue = TypedValue<RankedTensorType>;

public:
  DecomposeScaledBlocked(MLIRContext *context,
                         const ttg::intel::DPASAnalysis &dpasAnalysis)
      : OpRewritePattern<tt::DotScaledOp>(context), dpasAnalysis(dpasAnalysis) {
  }

  mlir::LogicalResult
  matchAndRewrite(tt::DotScaledOp scaledDotOp,
                  PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = scaledDotOp.getType();
    if (!oldRetType.getEncoding() ||
        isa<ttg::intel::DpasEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    MLIRContext *ctx = scaledDotOp.getContext();
    TensorValue a = scaledDotOp.getLhs();
    TensorValue b = scaledDotOp.getRhs();
    TensorValue scale = scaledDotOp.getLhsScale();
    tt::ScaleDotElemType aType = scaledDotOp.getLhsType();
    tt::ScaleDotElemType bType = scaledDotOp.getRhsType();

    assert(scaledDotOp.getRhsScale() == nullptr && "rhs scale NYI");
    assert((aType == tt::ScaleDotElemType::E4M3 ||
            aType == tt::ScaleDotElemType::E5M2 ||
            aType == tt::ScaleDotElemType::E2M1) &&
           "NYI: lhs supports fp4 or fp8");
    assert(bType == tt::ScaleDotElemType::E4M3 ||
           bType == tt::ScaleDotElemType::E5M2 ||
           bType == tt::ScaleDotElemType::BF16 &&
               "NYI: rhs supports fp8 and bf16");

    // Convert accumulator.
    ttg::intel::DpasEncodingAttr dpasEnc =
        getDPASEncoding(rewriter, scaledDotOp);
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), dpasEnc);
    TensorValue oldAcc = scaledDotOp.getC();
    TensorValue newAcc = rewriter.create<ttg::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);

    // Upcast A operand.
    auto dpasEncForA = ttg::intel::DpasEncodingAttr::get(
        ctx, dpasEnc.getRepeatCount(), dpasEnc.getSystolicDepth(),
        dpasEnc.getExecutionSize(), 2 * dpasEnc.getOpsPerChannel(),
        dpasEnc.getWarpsPerCTA(), dpasEnc.getRepCluster(),
        dpasEnc.getSubGroupSize());
    auto newAEncoding = ttg::DotOperandEncodingAttr::get(
        ctx, 0, dpasEncForA, dpasEncForA.getOpsPerChannel());
    a = createArg(rewriter, a, aType, newAEncoding);

    auto mod = scaledDotOp->getParentOfType<ModuleOp>();
    unsigned warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned instrShapeM = dpasEnc.getDPASInstShapeA()[1];
    SmallVector<unsigned> threadsPerWarp{instrShapeM, warpSize / instrShapeM};
    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());
    auto newScaleEncoding = ttg::BlockedEncodingAttr::get(
        ctx, {1, 1}, threadsPerWarp, newAEncoding.getWarpsPerCTA(),
        newAEncoding.getCTAOrder(), CTALayout);
    scale = createScale(rewriter, scale, newScaleEncoding);

    auto retTypeEncoding = ttg::DotOperandEncodingAttr::get(
        ctx, 0, dpasEnc, dpasEnc.getOpsPerChannel());
    a = createUpcastMxfpOp(rewriter, a, scale, aType, retTypeEncoding);

    // Create B operand.
    assert(bType != tt::ScaleDotElemType::E2M1 && "NYI: rhs scale for fp4");
    auto newBEncoding = ttg::DotOperandEncodingAttr::get(
        ctx, 1, dpasEnc, dpasEnc.getOpsPerChannel());
    b = createArg(rewriter, b, bType, newBEncoding);

    auto newDot = rewriter.create<tt::DotOp>(scaledDotOp.getLoc(), newRetType,
                                             a, b, newAcc);
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(scaledDotOp, oldRetType,
                                                      newDot);
    return success();
  }

private:
  ttg::intel::DpasEncodingAttr
  getDPASEncoding(PatternRewriter &rewriter,
                  tt::DotScaledOp scaledDotOp) const {
    auto mod = scaledDotOp->getParentOfType<ModuleOp>();
    auto dpasCap = ttg::intel::DpasEncodingAttr::getDPASCapability(mod);
    Type elemType = scaledDotOp.getRhs().getType().getElementType();
    unsigned opsPerChan =
        ttg::intel::DpasEncodingAttr::getOpsPerChannel(dpasCap, elemType);

    unsigned numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
    SmallVector<unsigned> warpsPerTile = {numWarps, 1};

    ArrayRef<int64_t> retShape = scaledDotOp.getType().getShape();
    size_t rank = retShape.size();
    SmallVector<unsigned> repCluster(rank, 1);

    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    return ttg::intel::DpasEncodingAttr::get(
        rewriter.getContext(), dpasCap.repeatCount, dpasCap.systolicDepth,
        dpasCap.executionSize, opsPerChan, warpsPerTile, repCluster,
        threadsPerWarp);
  }

  TensorValue createArg(PatternRewriter &rewriter, TensorValue v,
                        tt::ScaleDotElemType type, Attribute vEncoding) const {
    RankedTensorType vType = v.getType();
    auto newVType = RankedTensorType::get(vType.getShape(),
                                          vType.getElementType(), vEncoding);
    TensorValue ret =
        rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);
    if (type != tt::ScaleDotElemType::E2M1 &&
        type != tt::ScaleDotElemType::BF16) {
      // convert to bf16
      assert(type == tt::ScaleDotElemType::E5M2 ||
             type == tt::ScaleDotElemType::E4M3);
      auto vTypeBf16 = RankedTensorType::get(
          newVType.getShape(), rewriter.getBF16Type(), newVType.getEncoding());
      ret = rewriter.create<tt::FpToFpOp>(v.getLoc(), vTypeBf16, ret);
    }
    return ret;
  }

  TensorValue createScale(PatternRewriter &rewriter, TensorValue scale,
                          Attribute scaleEncoding) const {
    RankedTensorType scaleType = scale.getType();
    auto newScaleDotElemType = RankedTensorType::get(
        scaleType.getShape(), scaleType.getElementType(), scaleEncoding);
    return rewriter.create<ttg::ConvertLayoutOp>(scale.getLoc(),
                                                 newScaleDotElemType, scale);
  }

  TensorValue createUpcastMxfpOp(PatternRewriter &rewriter, TensorValue a,
                                 TensorValue scale, tt::ScaleDotElemType type,
                                 Attribute retTypeEncoding) const {
    auto aType = cast<RankedTensorType>(a.getType());
    auto retType = RankedTensorType::get(
        aType.getShape(), aType.getElementType(), retTypeEncoding);
    if (type == tt::ScaleDotElemType::E2M1) {
      RankedTensorType retTy;
      SmallVector<int64_t> newShape(aType.getShape());
      newShape.back() *= 2;
      retType = RankedTensorType::get(
          newShape, FloatType::getBF16(rewriter.getContext()), retTypeEncoding);
    }
    // TODO: Check whether constructing without explicit retType works.
    return rewriter.create<ttg::UpcastMXFPOp>(a.getLoc(), retType, a, scale,
                                              type);
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
        return builder.create<tt::FpToFpOp>(loc, tensorPromotedType, operand);
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
  mod.walk([](tt::DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    auto dpasLayout =
        dyn_cast<ttg::intel::DpasEncodingAttr>(D.getType().getEncoding());

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
    auto &dpasAnalysis = getAnalysis<ttg::intel::DPASAnalysis>();

    RewritePatternSet patterns(context);
    patterns.add<BlockedToDPAS, DecomposeScaledBlocked>(context, dpasAnalysis);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    decomposeMixedModeDotOp(m);
  }
};
