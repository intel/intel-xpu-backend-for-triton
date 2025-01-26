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
#include <optional>

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
      if (isa<Float8E5M2Type>(oldAType.getElementType()) ||
          isa<Float8E4M3FNType>(oldAType.getElementType()))
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
    // opA are packed to i16 for scalar type < 16 bits. opB are packed to i32.
    auto newAEncoding = ttg::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(),
        opsPerChan == 1 ? opsPerChan : opsPerChan / 2);
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
             elemType == tt::ScaleDotElemType::BF16 ||
             elemType == tt::ScaleDotElemType::FP16;
    };
    if (!supportsTypes(aElemType) || !supportsTypes(bElemType))
      return rewriter.notifyMatchFailure(scaledDotOp, "NYI: mxfp6 operand");

    ttgi::DpasEncodingAttr dpasEnc = getDPASEncoding(scaledDotOp, rewriter);

    TensorValue newAcc = convertAccumulator(scaledDotOp, dpasEnc, rewriter);
    RankedTensorType newRetType = newAcc.getType();

    std::tie(a, b) =
        convertOperands({a, aElemType, aScale}, {b, bElemType, bScale},
                        scaledDotOp.getFastMath(), dpasEnc, newRetType,
                        scaledDotOp->getParentOfType<ModuleOp>(), rewriter);

    auto newDot = rewriter.create<tt::DotOp>(scaledDotOp.getLoc(), newRetType,
                                             a, b, newAcc);
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(scaledDotOp, oldRetType,
                                                      newDot);
    return success();
  }

private:
  struct OpDescriptor {
    TensorValue op;
    triton::ScaleDotElemType elemType;
    TensorValue scale;
  };

  std::pair<TensorValue, TensorValue>
  convertOperands(OpDescriptor aDesc, OpDescriptor bDesc, bool fastMath,
                  ttgi::DpasEncodingAttr dpasEnc, RankedTensorType newRetType,
                  ModuleOp mod, PatternRewriter &rewriter) const {
    assert((aDesc.scale || bDesc.scale) && "No scale provided");
    assert(!(aDesc.scale && bDesc.scale) && "NYI: Both LHS and RHS scale");

    bool useFp16 = aDesc.elemType == tt::ScaleDotElemType::FP16 ||
                   bDesc.elemType == tt::ScaleDotElemType::FP16;

    if (aDesc.scale) {
      TensorValue newA =
          convertScaledOperand<ttgi::DpasEncodingAttr::OpIdx::OperandA>(
              aDesc, useFp16, fastMath, dpasEnc, newRetType, mod, rewriter);
      TensorValue newB =
          convertUnscaledOperand<ttgi::DpasEncodingAttr::OpIdx::OperandB>(
              bDesc, useFp16, dpasEnc, newRetType, rewriter);
      return {newA, newB};
    }

    TensorValue newB =
        convertScaledOperand<ttgi::DpasEncodingAttr::OpIdx::OperandB>(
            bDesc, useFp16, fastMath, dpasEnc, newRetType, mod, rewriter);
    TensorValue newA =
        convertUnscaledOperand<ttgi::DpasEncodingAttr::OpIdx::OperandA>(
            aDesc, useFp16, dpasEnc, newRetType, rewriter);
    return {newA, newB};
  }

  template <ttgi::DpasEncodingAttr::OpIdx opIdx>
  TensorValue convertScaledOperand(OpDescriptor opDesc, bool useFp16,
                                   bool fastMath,
                                   ttg::intel::DpasEncodingAttr dpasEnc,
                                   RankedTensorType retType, ModuleOp mod,
                                   PatternRewriter &rewriter) const {
    assert(opDesc.scale && "Expecting valid operand & scale");

    MLIRContext *ctx = opDesc.op.getContext();
    unsigned numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
    unsigned warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned opsPerChannel = dpasEnc.getOpsPerChannel();
    unsigned rank = retType.getRank();

    auto opEncoding = ttg::intel::DpasEncodingAttr::get(
        ctx, dpasEnc.getRepeatCount(), dpasEnc.getSystolicDepth(),
        dpasEnc.getExecutionSize(), opsPerChannel, dpasEnc.getWarpsPerCTA(),
        dpasEnc.getRepCluster(),
        product<unsigned>(dpasEnc.getThreadsPerWarp()));

    int kWidth = dpasEnc.getOpsPerChannel();
    if constexpr (opIdx == ttgi::DpasEncodingAttr::OpIdx::OperandA) {
      // Operand A is packed to i16 for scalar type < 16 bits.
      kWidth = kWidth == 1 ? 1 : kWidth / 2;
    }
    auto newOpEncoding = ttg::DotOperandEncodingAttr::get(ctx, unsigned(opIdx),
                                                          opEncoding, kWidth);
    TensorValue op =
        createArg(opDesc.op, opDesc.elemType, useFp16, newOpEncoding, rewriter);

    unsigned instrShapeM = dpasEnc.getDPASInstShapeA()[0];
    SmallVector<unsigned, 2> threadsPerWarp{instrShapeM,
                                            warpSize / instrShapeM};

    SmallVector<unsigned, 2> warpsPerCTA(rank, 1);
    warpsPerCTA[0] = numWarps;
    auto CTALayout = ttg::getCTALayout(retType.getEncoding());

    auto newScaleEncoding =
        ttg::BlockedEncodingAttr::get(ctx, {1, 1}, threadsPerWarp, warpsPerCTA,
                                      newOpEncoding.getCTAOrder(), CTALayout);
    TensorValue scale = createScale(opDesc.scale, newScaleEncoding, rewriter);

    auto upcastOp = createUpcastMxfpOp(op, scale, opDesc.elemType, useFp16,
                                       fastMath, rewriter);
    if (opDesc.elemType == tt::ScaleDotElemType::E2M1) {
      auto resultType = cast<RankedTensorType>(upcastOp.getType());
      auto newRetType = RankedTensorType::get(
          resultType.getShape(), resultType.getElementType(), newOpEncoding);
      upcastOp = rewriter.create<ttg::ConvertLayoutOp>(opDesc.op.getLoc(),
                                                       newRetType, upcastOp);
    }
    return upcastOp;
  }

  template <ttgi::DpasEncodingAttr::OpIdx opIdx>
  TensorValue convertUnscaledOperand(OpDescriptor opDesc, bool useFp16,
                                     ttg::intel::DpasEncodingAttr dpasEnc,
                                     RankedTensorType retType,
                                     PatternRewriter &rewriter) const {
    assert(!opDesc.scale && "Scale should be NULL");
    int kWidth = dpasEnc.getOpsPerChannel();
    if constexpr (opIdx == ttgi::DpasEncodingAttr::OpIdx::OperandA) {
      // Operand A is packed to i16 for scalar type < 16 bits.
      kWidth = kWidth == 1 ? 1 : kWidth / 2;
    }
    auto newOpEncoding = ttg::DotOperandEncodingAttr::get(
        opDesc.op.getContext(), unsigned(opIdx), dpasEnc, kWidth);
    return createArg(opDesc.op, opDesc.elemType, useFp16, newOpEncoding,
                     rewriter);
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

  TensorValue createArg(TensorValue v, tt::ScaleDotElemType type, bool useFp16,
                        Attribute vEncoding, PatternRewriter &rewriter) const {
    RankedTensorType vType = v.getType();
    auto newVType = RankedTensorType::get(vType.getShape(),
                                          vType.getElementType(), vEncoding);
    TensorValue ret =
        rewriter.create<ttg::ConvertLayoutOp>(v.getLoc(), newVType, v);

    // convert to bf16
    if (type != tt::ScaleDotElemType::E2M1 &&
        type != tt::ScaleDotElemType::BF16 &&
        type != tt::ScaleDotElemType::FP16) {
      assert(type == tt::ScaleDotElemType::E5M2 ||
             type == tt::ScaleDotElemType::E4M3);
      auto upcastedType = RankedTensorType::get(
          newVType.getShape(),
          useFp16 ? rewriter.getF16Type() : rewriter.getBF16Type(),
          newVType.getEncoding());
      ret = cast<TypedValue<RankedTensorType>>(
          rewriter.create<tt::FpToFpOp>(v.getLoc(), upcastedType, ret)
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
                                 tt::ScaleDotElemType elemType, bool useFp16,
                                 bool fastMath,
                                 PatternRewriter &rewriter) const {
    if (!scale)
      return v;

    Builder b(v.getContext());
    Type outputElemType = useFp16 ? b.getF16Type() : b.getBF16Type();
    auto retTy = triton::gpu::intel::UpcastMXFPOp::deduceOutputType(
        v, elemType, outputElemType);
    return rewriter.create<ttgi::UpcastMXFPOp>(v.getLoc(), retTy, v, scale,
                                               elemType, fastMath);
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
      bool isNativeFP8 =
          isa<Float8E5M2Type>(AElType) || isa<Float8E4M3FNType>(AElType);
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

static void updateValueType(Value v, Attribute encoding,
                            ArrayRef<int64_t> shape) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  auto newType =
      RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  v.setType(newType);
}

static std::optional<tt::TransOp>
updateUsers(Value result, const SetVector<Operation *> &slice) {
  if (llvm::any_of(result.getUsers(),
                   [&](Operation *user) { return slice.count(user) == 0; })) {
    OpBuilder builder(result.getContext());
    builder.setInsertionPointAfterValue(result);
    auto transOp =
        builder.create<tt::TransOp>(result.getLoc(), result, ArrayRef({1, 0}));
    result.replaceUsesWithIf(transOp.getResult(), [&](OpOperand &operand) {
      return operand.getOwner() != transOp.getOperation() &&
             slice.count(operand.getOwner()) == 0;
    });
    return transOp;
  }
  return std::nullopt;
}

// TODO: Sync the transpose in the IR, this is done to avoid generating convert
// layout when we have a transpose right after a dot as mma layout cannot be
// propagated through transpose op. Once we have layouts that can represent
// transposed MMA we can remove this transformation.
static void sinkTransposeOp(tt::TransOp input) {
  SmallVector<tt::TransOp> queue = {input};
  while (!queue.empty()) {
    tt::TransOp transOp = queue.back();
    Value currentValue = transOp.getResult();
    queue.pop_back();
    mlir::ForwardSliceOptions options;
    options.filter = [](Operation *op) {
      if (op->hasTrait<OpTrait::Elementwise>() && op->getNumOperands() == 1)
        return true;
      if (isa<scf::YieldOp>(op))
        return isa<scf::ForOp>(op->getParentOp());
      return isa<ttg::ConvertLayoutOp>(op);
    };
    SetVector<Operation *> slice;
    mlir::getForwardSlice(currentValue, &slice, options);
    for (Operation *op : slice) {
      if (op->hasTrait<OpTrait::Elementwise>()) {
        // Update users of transpose op.
        if (op->getOperand(0) == transOp.getResult())
          op->setOperand(0, transOp.getOperand());
        // Update the type of the result.
        for (Value result : op->getResults()) {
          auto srcType = cast<RankedTensorType>(op->getOperand(0).getType());
          updateValueType(result, srcType.getEncoding(), srcType.getShape());
          updateUsers(result, slice);
        }
        continue;
      }
      if (auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(op)) {
        // Update users of transpose op.
        if (op->getOperand(0) == transOp.getResult())
          op->setOperand(0, transOp.getOperand());
        auto resultEncoding = cvtOp.getType().getEncoding();
        auto newDstEncoding = ttgi::inferSrcEncoding(transOp, resultEncoding);
        assert(newDstEncoding && "Expecting valid result encoding");
        auto srcType = cast<RankedTensorType>(cvtOp.getOperand().getType());
        updateValueType(cvtOp.getResult(), newDstEncoding, srcType.getShape());
        updateUsers(cvtOp.getResult(), slice);
        continue;
      }
      assert(isa<scf::YieldOp>(op) &&
             "Transpose forward slice should contain "
             "only elementwise, convert layout and yield ops.");
      auto forOp = cast<scf::ForOp>(op->getParentOp());
      for (OpOperand &operand : op->getOpOperands()) {
        Operation *def = operand.get().getDefiningOp();
        if (def && (slice.count(def)) || def == transOp.getOperation()) {
          if (def == transOp.getOperation())
            operand.set(transOp.getOperand());
          Type newType = operand.get().getType();
          forOp.getResult(operand.getOperandNumber()).setType(newType);
          std::optional<tt::TransOp> retTrans =
              updateUsers(forOp.getResult(operand.getOperandNumber()), slice);
          // Recursively try to propagate the new transpose inserted.
          if (retTrans.has_value())
            queue.push_back(retTrans.value());
          forOp.getRegionIterArg(operand.getOperandNumber()).setType(newType);
          std::optional<tt::TransOp> argTrans = updateUsers(
              forOp.getRegionIterArg(operand.getOperandNumber()), slice);
          if (argTrans.has_value())
            queue.push_back(argTrans.value());
          OpBuilder builder(forOp);
          OpOperand &init = forOp.getInitsMutable()[operand.getOperandNumber()];
          auto initTranspose = builder.create<tt::TransOp>(
              forOp.getLoc(), init.get(), ArrayRef({1, 0}));
          init.set(initTranspose);
        }
      }
    }
  }
}

static tt::TransOp transposeDotOp(tt::DotScaledOp dotOp) {
  assert(dotOp.getLhsScale() == nullptr && dotOp.getRhsScale() != nullptr &&
         "Transpose DotOp expects scale on RHS");
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getLhs();
  std::array<int, 2> transOrder = {1, 0};
  auto lhsTransposed =
      builder.create<tt::TransOp>(lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getRhs();
  auto rhsTransposed =
      builder.create<tt::TransOp>(rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  auto cTransposed = builder.create<tt::TransOp>(c.getLoc(), c, transOrder);
  auto result = builder.create<tt::DotScaledOp>(
      dotOp.getLoc(), cTransposed.getType(), rhsTransposed, lhsTransposed,
      cTransposed, dotOp.getRhsScale(), dotOp.getLhsScale(), dotOp.getRhsType(),
      dotOp.getLhsType(), dotOp.getFastMath());
  auto transOp =
      builder.create<tt::TransOp>(result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transOp.getOperation());
  dotOp.erase();
  return transOp;
}

static void transposeDots(ModuleOp m) {
  SmallVector<tt::DotScaledOp> toTranspose;
  m.walk([&](tt::DotScaledOp dotOp) -> void {
    if (dotOp.getLhsScale() == nullptr && dotOp.getRhsScale() != nullptr)
      toTranspose.push_back(dotOp);
  });
  SmallVector<tt::TransOp> transposes;
  for (tt::DotScaledOp &dotOp : toTranspose) {
    tt::TransOp transpose = transposeDotOp(dotOp);
    transposes.push_back(transpose);
  }

  for (tt::TransOp transpose : transposes) {
    sinkTransposeOp(transpose);
  }
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

    // Transpose dotOp operations that have a scale on the RHS.
    transposeDots(m);

    RewritePatternSet patterns(context);
    patterns.add<BlockedToDPAS, DecomposeScaledBlocked>(context, dpasAnalysis);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    decomposeMixedModeDotOp(m);
  }
};
