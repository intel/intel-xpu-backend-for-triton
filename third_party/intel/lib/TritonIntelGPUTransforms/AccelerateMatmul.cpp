#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "Dialect/TritonIntelGPU/Transforms/DecomposeScaledBlocked.h"
#include "Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
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

// FIXME: Remove once IGC can split large 2D block loads.
static void setAttrOnBOperand(tt::DotOp dotOp, StringRef attrName,
                              Attribute attr) {
  Operation *defOp = dotOp.getB().getDefiningOp();
  while (auto convOp = dyn_cast_or_null<ttg::ConvertLayoutOp>(defOp))
    defOp = convOp.getSrc().getDefiningOp();
  if (auto transOp = dyn_cast_or_null<tt::TransOp>(defOp))
    defOp = transOp.getOperand().getDefiningOp();
  if (auto loadOp = dyn_cast_or_null<tt::LoadOp>(defOp))
    loadOp->setAttr(attrName, attr);
}

unsigned getOpsPerChannel(Type elemType, ModuleOp m) {
  assert(elemType.isIntOrFloat() && "unsupported type for DpasEncodingAttr");

  unsigned dpasElemBitWidths = elemType.getIntOrFloatBitWidth();
  bool supportsFP8 = m->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                                    getSupportBlockScaleDPASAttrName());
  if (!supportsFP8 && llvm::isa<Float8E5M2Type, Float8E4M3FNType>(elemType))
    dpasElemBitWidths *= 2; // We are upcasting FP8 to FP16.

  return ttg::intel::DpasEncodingAttr::DPASCapability::opsChanBitWidths /
         dpasElemBitWidths;
}

SmallVector<unsigned>
getWarpsPerTile(tt::DotOp dotOp, ttgi::DpasEncodingAttr::DPASCapability dpasCap,
                const ArrayRef<int64_t> shape, unsigned numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };

  SetVector<Operation *> slices = getSlice(dotOp, {filter});
  for (Operation *op : slices) {
    if (isa<tt::DotOp>(op) && (op != dotOp)) {
      if (auto forOp = op->getParentOfType<scf::ForOp>()) {
        // FIXME: Remove once IGC can split large 2D block loads.
        MLIRContext *ctx = forOp->getContext();
        StringRef attrName =
            ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName();
        setAttrOnBOperand(dotOp, attrName, UnitAttr::get(ctx));
        setAttrOnBOperand(cast<tt::DotOp>(op), attrName, UnitAttr::get(ctx));
      }
      SmallVector<unsigned> ret(shape.size(), 1);
      ret[0] = numWarps;
      return ret;
    }
  }

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
    unsigned numWarps = ttg::lookupNumWarps(funcOp);

    TensorValue a = dotOp.getA();
    TensorValue b = dotOp.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    ModuleOp mod = funcOp->getParentOfType<ModuleOp>();
    auto dpasCap = ttgi::DpasEncodingAttr::getDPASCapability(mod);
    Type elemType = oldAType.getElementType();
    unsigned opsPerChan = getOpsPerChannel(elemType, mod);
    SmallVector<unsigned> warpsPerTile =
        getWarpsPerTile(dotOp, dpasCap, retShape, numWarps);
    size_t rank = retShape.size();
    SmallVector<unsigned> repCluster(rank, 1);

    unsigned repeatCount =
        std::min(dpasCap.repeatCount, (unsigned)retShape[rank - 2] /*M*/);
    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned numElemsPerRowForA =
        opsPerChan == 1
            ? dpasCap.systolicDepth
            : dpasCap.systolicDepth * 2; // A is packed to i16 or i32.
    unsigned minM = mlir::ceil<unsigned>(threadsPerWarp, numElemsPerRowForA);
    repeatCount = std::max(repeatCount, minM);
    auto dpasEnc = ttgi::DpasEncodingAttr::get(
        oldRetType.getContext(), repeatCount, dpasCap.systolicDepth,
        dpasCap.executionSize, opsPerChan, warpsPerTile, repCluster,
        threadsPerWarp);

    if (dpasCap.isPVC() || dpasCap.isFalconShore()) {
      unsigned dpasElemBitWidths =
          oldAType.getElementType().getIntOrFloatBitWidth();

      // We are upcasting FP8 to FP16
      if (isa<Float8E5M2Type, Float8E4M3FNType>(oldAType.getElementType()))
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
          oldRetType.getContext(), repeatCount, dpasCap.systolicDepth,
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
      bool isNativeFP8 = isa<Float8E5M2Type, Float8E4M3FNType>(AElType);
      // fp8 is not always natively supported by the the DPAS instruction,
      // promote it to fp16 when necessary

      auto m = dotOp->getParentOfType<ModuleOp>();
      bool supportsFP8 = m->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                                        getSupportBlockScaleDPASAttrName());
      if (supportsFP8 || !isNativeFP8)
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
  assert(dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr &&
         "Transpose DotOp expects scale on RHS");
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getA();
  std::array<int, 2> transOrder = {1, 0};
  auto lhsTransposed =
      builder.create<tt::TransOp>(lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getB();
  auto rhsTransposed =
      builder.create<tt::TransOp>(rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  auto cTransposed = builder.create<tt::TransOp>(c.getLoc(), c, transOrder);
  auto result = builder.create<tt::DotScaledOp>(
      dotOp.getLoc(), cTransposed.getType(), rhsTransposed, lhsTransposed,
      cTransposed, dotOp.getBScale(), dotOp.getAScale(), dotOp.getBElemType(),
      dotOp.getAElemType(), dotOp.getFastMath());
  auto transOp =
      builder.create<tt::TransOp>(result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transOp.getOperation());
  dotOp.erase();
  return transOp;
}

static void transposeDots(ModuleOp m) {
  SmallVector<tt::DotScaledOp> toTranspose;
  m.walk([&](tt::DotScaledOp dotOp) -> void {
    if (dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr)
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
    constexpr int benefitDefault = 1;
    patterns.add<BlockedToDPAS>(context, dpasAnalysis);
    ttgi::populateDecomposeScaledBlockedPatterns(patterns, benefitDefault);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    decomposeMixedModeDotOp(m);
  }
};
