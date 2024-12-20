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
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"

#define PVC_2D_LOAD_MAXIMUM_NUMBER_OF_ROWS 32
#define PVC_2D_LOAD_MAXIMUM_BYTES_OF_COLS 64

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUACCELERATEMATMUL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

SmallVector<unsigned>
getWarpsPerTile(tt::DotOp dotOp, ttgi::DpasEncodingAttr::DPASCapability dpasCap,
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
        isa<ttgi::DpasEncodingAttr>(oldRetType.getEncoding()))
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

    auto dpasCap = ttgi::DpasEncodingAttr::getDPASCapability(mod);
    Type elemType = oldAType.getElementType();
    unsigned opsPerChan = ttgi::DpasEncodingAttr::getOpsPerChannel(elemType);
    SmallVector<unsigned> warpsPerTile =
        getWarpsPerTile(dotOp, dpasCap, retShape, numWarps);
    size_t rank = retShape.size();
    SmallVector<unsigned> repCluster(rank, 1);

    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    auto dpasEnc = ttgi::DpasEncodingAttr::get(
        oldRetType.getContext(), dpasCap.repeatCount, dpasCap.systolicDepth,
        dpasCap.executionSize, opsPerChan, warpsPerTile, repCluster,
        threadsPerWarp);

    if (dpasCap.isPVC() || dpasCap.isFalconShore()) {
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

      dpasEnc = ttgi::DpasEncodingAttr::get(
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
    if (!isa_and_nonnull<ttg::BlockedEncodingAttr>(oldRetType.getEncoding()))
      return rewriter.notifyMatchFailure(
          scaledDotOp, "expected blocked encoding result tensor");

    unsigned rank = oldRetType.getRank();
    if (rank == 3)
      return rewriter.notifyMatchFailure(scaledDotOp, "NYI: 3d case");

    TensorValue a = scaledDotOp.getLhs();
    TensorValue b = scaledDotOp.getRhs();
    TensorValue aScale = scaledDotOp.getLhsScale();
    TensorValue bScale = scaledDotOp.getRhsScale();
    if (aScale && bScale)
      return rewriter.notifyMatchFailure(scaledDotOp,
                                         "NYI: both LHS and RHS scale");

    tt::ScaleDotElemType aElemType = scaledDotOp.getLhsType();
    tt::ScaleDotElemType bElemType = scaledDotOp.getRhsType();
    auto supportsTypes = [](tt::ScaleDotElemType elemType) {
      return elemType == tt::ScaleDotElemType::E2M1 ||
             elemType == tt::ScaleDotElemType::E4M3 ||
             elemType == tt::ScaleDotElemType::E5M2 ||
             elemType == tt::ScaleDotElemType::BF16;
    };
    if (!supportsTypes(aElemType) || !supportsTypes(bElemType))
      return rewriter.notifyMatchFailure(scaledDotOp, "NYI: mxfp6 operand");

    ttgi::DpasEncodingAttr dpasEnc = getDPASEncoding(scaledDotOp, rewriter);

    TensorValue newAcc = convertAccumulator(scaledDotOp, dpasEnc, rewriter);
    RankedTensorType newRetType = newAcc.getType();

    std::tie(a, b) = convertOperands(
        {a, aElemType, aScale}, {b, bElemType, bScale}, dpasEnc, newRetType,
        scaledDotOp->getParentOfType<ModuleOp>(), rewriter);

    auto newDot = rewriter.create<tt::DotOp>(scaledDotOp.getLoc(), newRetType,
                                             a, b, newAcc);
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(scaledDotOp, oldRetType,
                                                      newDot);
    return success();
  }

private:
  const bool upcastMXFPUseDotOpEnc =
      mlir::triton::tools::getBoolEnv("TRITON_INTEL_UPCASTMXFP_DOTOP_ENCODING");

  struct OpDescriptor {
    TensorValue op;
    triton::ScaleDotElemType elemType;
    TensorValue scale;
  };

  std::pair<TensorValue, TensorValue>
  convertOperands(OpDescriptor aDesc, OpDescriptor bDesc,
                  ttgi::DpasEncodingAttr dpasEnc, RankedTensorType newRetType,
                  ModuleOp mod, PatternRewriter &rewriter) const {
    assert((aDesc.scale || bDesc.scale) && "No scale provided");
    assert(!(aDesc.scale && bDesc.scale) && "NYI: Both LHS and RHS scale");

    if (aDesc.scale) {
      TensorValue newA =
          convertScaledOperand<ttgi::DpasEncodingAttr::OpIdx::OperandA>(
              aDesc, dpasEnc, newRetType, mod, rewriter);
      TensorValue newB =
          convertUnscaledOperand<ttgi::DpasEncodingAttr::OpIdx::OperandB>(
              bDesc, dpasEnc, newRetType, rewriter);
      return {newA, newB};
    }

    TensorValue newB =
        convertScaledOperand<ttgi::DpasEncodingAttr::OpIdx::OperandB>(
            bDesc, dpasEnc, newRetType, mod, rewriter);
    TensorValue newA =
        convertUnscaledOperand<ttgi::DpasEncodingAttr::OpIdx::OperandA>(
            aDesc, dpasEnc, newRetType, rewriter);
    return {newA, newB};
  }

  template <ttgi::DpasEncodingAttr::OpIdx opIdx>
  TensorValue convertScaledOperand(OpDescriptor opDesc,
                                   ttg::intel::DpasEncodingAttr dpasEnc,
                                   RankedTensorType retType, ModuleOp mod,
                                   PatternRewriter &rewriter) const {
    assert(opDesc.scale && "Expecting valid operand & scale");

    MLIRContext *ctx = opDesc.op.getContext();
    unsigned numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
    unsigned warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned opsPerChannel = dpasEnc.getOpsPerChannel();
    unsigned rank = retType.getRank();

    if (upcastMXFPUseDotOpEnc) {
      // if (opDesc.elemType == tt::ScaleDotElemType::E2M1)
      //   opsPerChannel *= 2;

      auto opEncoding = ttg::intel::DpasEncodingAttr::get(
          ctx, dpasEnc.getRepeatCount(), dpasEnc.getSystolicDepth(),
          dpasEnc.getExecutionSize(), opsPerChannel, dpasEnc.getWarpsPerCTA(),
          dpasEnc.getRepCluster(), dpasEnc.getSubGroupSize());

      auto newOpEncoding = ttg::DotOperandEncodingAttr::get(
          ctx, unsigned(opIdx), opEncoding, opEncoding.getOpsPerChannel());
      TensorValue op =
          createArg(opDesc.op, opDesc.elemType, newOpEncoding, rewriter);

      unsigned warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
      unsigned repeatCount = dpasEnc.getRepeatCount();
      unsigned instrShapeOuter;
      if (opIdx == 0)
        instrShapeOuter = dpasEnc.getDPASInstShapeA()[opIdx];
      else
        instrShapeOuter = dpasEnc.getDPASInstShapeB()[opIdx];
      SmallVector<unsigned, 2> threadsPerWarp{instrShapeOuter,
                                              warpSize / instrShapeOuter};
      // auto scaleTy = cast<RankedTensorType>(opDesc.scale.getType());
      // unsigned scalingBlocks = scaleTy.getShape()[1];
      // SmallVector<unsigned, 2> threadsPerWarp = {repeatCount, warpSize /
      // repeatCount};
      SmallVector<unsigned, 2> warpsPerCTA(rank, 1);
      warpsPerCTA[0] = numWarps;
      auto CTALayout = ttg::getCTALayout(retType.getEncoding());

      auto newScaleEncoding = ttg::BlockedEncodingAttr::get(
          ctx, {1, 1}, threadsPerWarp, warpsPerCTA, newOpEncoding.getCTAOrder(),
          CTALayout);
      TensorValue scale = createScale(opDesc.scale, newScaleEncoding, rewriter);

      return createUpcastMxfpOp(op, scale, opDesc.elemType, rewriter);
    }

    auto scaleEncoding = dyn_cast<ttg::BlockedEncodingAttr>(
        opDesc.scale.getType().getEncoding());
    assert(scaleEncoding && "Expecting blocked encoding for scale");

    // Referring to
    // https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    // the scalingBlockSize should be 32 for E5M2, E4M3 and E2M1
    unsigned scalingBlockSize = 32;
    // 2 FP4E2M1 are packed in one i8
    if (opDesc.elemType == tt::ScaleDotElemType::E2M1)
      scalingBlockSize = 16;

    SmallVector<unsigned> sizePerThread = {1, 1};
    SmallVector<unsigned> threadsPerWarp = {1, 1};

    sizePerThread[!unsigned(opIdx)] = scalingBlockSize;
    threadsPerWarp[unsigned(opIdx)] = warpSize;
    SmallVector<unsigned> warpsPerCTA = {numWarps, 1};

    auto newOpEncoding = ttg::BlockedEncodingAttr::get(
        ctx, sizePerThread, threadsPerWarp, warpsPerCTA,
        scaleEncoding.getCTAOrder(), scaleEncoding.getCTALayout());
    TensorValue op =
        createArg(opDesc.op, opDesc.elemType, newOpEncoding, rewriter);

    warpsPerCTA = bool(opIdx) ? SmallVector<unsigned>{1, numWarps}
                              : SmallVector<unsigned>{numWarps, 1};
    auto newScaleEncoding = ttg::BlockedEncodingAttr::get(
        ctx, {1, 1}, {warpSize, 1}, warpsPerCTA, scaleEncoding.getCTAOrder(),
        scaleEncoding.getCTALayout());
    TensorValue scale = createScale(opDesc.scale, newScaleEncoding, rewriter);

    auto retDpasEncoding = ttg::intel::DpasEncodingAttr::get(
        ctx, dpasEnc.getRepeatCount(), dpasEnc.getSystolicDepth(),
        dpasEnc.getExecutionSize(), opsPerChannel, dpasEnc.getWarpsPerCTA(),
        dpasEnc.getRepCluster(), dpasEnc.getSubGroupSize());
    auto retDotOpEncoding =
        ttg::DotOperandEncodingAttr::get(ctx, unsigned(opIdx), retDpasEncoding,
                                         retDpasEncoding.getOpsPerChannel());

    auto upcastOp = createUpcastMxfpOp(op, scale, opDesc.elemType, rewriter);

    auto resultType = cast<RankedTensorType>(upcastOp.getType());
    resultType = RankedTensorType::get(
        resultType.getShape(), resultType.getElementType(), retDotOpEncoding);
    return rewriter.create<ttg::ConvertLayoutOp>(opDesc.op.getLoc(), resultType,
                                                 upcastOp);
  }

  template <ttgi::DpasEncodingAttr::OpIdx opIdx>
  TensorValue convertUnscaledOperand(OpDescriptor opDesc,
                                     ttg::intel::DpasEncodingAttr dpasEnc,
                                     RankedTensorType retType,
                                     PatternRewriter &rewriter) const {
    assert(!opDesc.scale && "Scale should be NULL");

    auto newOpEncoding = ttg::DotOperandEncodingAttr::get(
        opDesc.op.getContext(), unsigned(opIdx), dpasEnc,
        dpasEnc.getOpsPerChannel());
    return createArg(opDesc.op, opDesc.elemType, newOpEncoding, rewriter);
  }

  ttg::intel::DpasEncodingAttr
  getDPASEncoding(tt::DotScaledOp scaledDotOp,
                  PatternRewriter &rewriter) const {
    auto mod = scaledDotOp->getParentOfType<ModuleOp>();
    TensorValue a = scaledDotOp.getLhs();
    TensorValue b = scaledDotOp.getRhs();
    TensorValue aScale = scaledDotOp.getLhsScale();
    TensorValue bScale = scaledDotOp.getRhsScale();
    assert((!aScale || !bScale) && "NYI: both LHS and RHS scale");

    Type elemType =
        aScale ? b.getType().getElementType() : a.getType().getElementType();
    unsigned opsPerChan =
        ttg::intel::DpasEncodingAttr::getOpsPerChannel(elemType);
    unsigned numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
    SmallVector<unsigned> warpsPerTile = {numWarps, 1};

    ArrayRef<int64_t> retShape = scaledDotOp.getType().getShape();
    size_t rank = retShape.size();
    SmallVector<unsigned> repCluster(rank, 1);

    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    auto dpasCap = ttg::intel::DpasEncodingAttr::getDPASCapability(mod);

    return ttg::intel::DpasEncodingAttr::get(
        rewriter.getContext(), dpasCap.repeatCount, dpasCap.systolicDepth,
        dpasCap.executionSize, opsPerChan, warpsPerTile, repCluster,
        threadsPerWarp);
  }

  TensorValue convertAccumulator(tt::DotScaledOp scaledDotOp,
                                 ttg::intel::DpasEncodingAttr &dpasEnc,
                                 PatternRewriter &rewriter) const {
    RankedTensorType retType = scaledDotOp.getType();
    auto newRetType = RankedTensorType::get(retType.getShape(),
                                            retType.getElementType(), dpasEnc);
    TensorValue oldAcc = scaledDotOp.getC();
    return rewriter.create<ttg::ConvertLayoutOp>(oldAcc.getLoc(), newRetType,
                                                 oldAcc);
  }

  TensorValue createArg(TensorValue v, tt::ScaleDotElemType type,
                        Attribute vEncoding, PatternRewriter &rewriter) const {
    RankedTensorType vType = v.getType();
    auto newVType = RankedTensorType::get(vType.getShape(),
                                          vType.getElementType(), vEncoding);
    TensorValue ret =
        rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);

    // convert to bf16
    if (type != tt::ScaleDotElemType::E2M1 &&
        type != tt::ScaleDotElemType::BF16) {
      assert(type == tt::ScaleDotElemType::E5M2 ||
             type == tt::ScaleDotElemType::E4M3);
      auto vTypeBf16 = RankedTensorType::get(
          newVType.getShape(), rewriter.getBF16Type(), newVType.getEncoding());
      ret = cast<TypedValue<RankedTensorType>>(
          rewriter.create<tt::FpToFpOp>(v.getLoc(), vTypeBf16, ret)
              .getResult());
    }
    return ret;
  }

  TensorValue createScale(TensorValue scale, Attribute scaleEncoding,
                          PatternRewriter &rewriter) const {
    assert(scale && scaleEncoding && "Expecting valid scale and encoding");
    RankedTensorType scaleType = scale.getType();
    auto newScaleType = RankedTensorType::get(
        scaleType.getShape(), scaleType.getElementType(), scaleEncoding);
    return rewriter.create<ttg::ConvertLayoutOp>(scale.getLoc(), newScaleType,
                                                 scale);
  }

  TensorValue createUpcastMxfpOp(TensorValue v, TensorValue scale,
                                 tt::ScaleDotElemType elemType,
                                 PatternRewriter &rewriter) const {
    if (!scale)
      return v;

    return rewriter.create<ttg::UpcastMXFPOp>(v.getLoc(), v, scale, elemType);
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
