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

#include "Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
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
static void setAttrOnBOperand(Operation *op, StringRef attrName,
                              Attribute attr) {
  Operation *defOp;
  llvm::TypeSwitch<Operation *>(op).Case<tt::DotOp, tt::DotScaledOp>(
      [&](auto op) { defOp = op.getB().getDefiningOp(); });
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
  bool supportsFP8 = m->hasAttr(
      ttgi::TritonIntelGPUDialect::getSupportBlockScaleDPASAttrName());
  if (!supportsFP8 && llvm::isa<Float8E5M2Type, Float8E4M3FNType>(elemType))
    dpasElemBitWidths *= 2; // We are upcasting FP8 to FP16.

  return ttgi::DpasEncodingAttr::DPASCapability::opsChanBitWidths /
         dpasElemBitWidths;
}

SmallVector<unsigned>
getWarpsPerTile(Operation *dotOp,
                ttgi::DpasEncodingAttr::DPASCapability dpasCap,
                const ArrayRef<int64_t> shape, unsigned numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };

  SetVector<Operation *> slices = getSlice(dotOp, {filter});
  for (Operation *op : slices) {
    if (isa<tt::DotOp, tt::DotScaledOp>(op) && (op != dotOp)) {
      if (auto forOp = op->getParentOfType<scf::ForOp>()) {
        // FIXME: Remove once IGC can split large 2D block loads.
        MLIRContext *ctx = forOp->getContext();
        StringRef attrName =
            ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName();
        setAttrOnBOperand(dotOp, attrName, UnitAttr::get(ctx));
        setAttrOnBOperand(op, attrName, UnitAttr::get(ctx));
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

template <class OpTy, typename = std::enable_if_t<llvm::is_one_of<
                          OpTy, tt::DotOp, tt::DotScaledOp>::value>>
class BlockedToDPAS : public OpRewritePattern<OpTy> {
  const ttgi::DPASAnalysis &dpasAnalysis;
  using TensorValue = TypedValue<RankedTensorType>;

public:
  BlockedToDPAS(MLIRContext *context, const ttgi::DPASAnalysis &dpasAnalysis,
                int benefit)
      : OpRewritePattern<OpTy>(context, benefit), dpasAnalysis(dpasAnalysis) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = op.getType();
    if (!oldRetType.getEncoding() ||
        isa<ttgi::DpasEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    auto funcOp = op->template getParentOfType<FunctionOpInterface>();
    if (dpasAnalysis.canUseDPAS(funcOp) != ttgi::DPASAnalysis::Result::True)
      return failure();

    auto dpasType = dpasAnalysis.getDPASType(op);

    if constexpr (std::is_same<OpTy, tt::DotScaledOp>::value) {
      switch (dpasType) {
      case ttgi::DPASAnalysis::DPASEngineType::FP32_FP32_FP16_FP16:
      case ttgi::DPASAnalysis::DPASEngineType::FP32_FP32_BF16_BF16:
      case ttgi::DPASAnalysis::DPASEngineType::BF16_BF16_BF16_BF16:
      case ttgi::DPASAnalysis::DPASEngineType::FP16_FP16_FP16_FP16:
      case ttgi::DPASAnalysis::DPASEngineType::FP32_FP32_FP8_FP8:
      case ttgi::DPASAnalysis::DPASEngineType::BF16_BF16_FP8_FP8:
        break;
      case ttgi::DPASAnalysis::DPASEngineType::FP32_FP32_FP4_FP4:
        // BDPAS only support to pack along K for A and B matrix.
        if (!(op.getRhsKPack() && op.getLhsKPack())) {
          return failure();
        }
        break;
      default:
        return failure();
      }

      auto aElemType = op.getAElemType();
      auto bElemType = op.getBElemType();
      bool isBothFP8 = (aElemType == tt::ScaleDotElemType::E4M3 ||
                        aElemType == tt::ScaleDotElemType::E5M2) &&
                       (bElemType == tt::ScaleDotElemType::E4M3 ||
                        bElemType == tt::ScaleDotElemType::E5M2);
      if (!isBothFP8) {
        // Doesn't support these mixed precision in bdpas natively.
        // Need to decompose to simpler tt.dot with software scale for now.
        // TODO: improve this by decompose to simpler tt.dot_scale with hardware
        // scaling. (intel-tools/intel-xpu-backend-for-triton#755)
        if (aElemType != bElemType)
          return failure();
      }
    }

    // Create DPAS encoding for the given number of warps
    ArrayRef<int64_t> retShape = oldRetType.getShape();
    unsigned numWarps = ttg::lookupNumWarps(funcOp);

    TensorValue a = op.getA();
    TensorValue b = op.getB();
    auto oldAType = cast<RankedTensorType>(a.getType());
    auto oldBType = cast<RankedTensorType>(b.getType());

    ModuleOp mod = funcOp->template getParentOfType<ModuleOp>();
    ttgi::DpasEncodingAttr::DPASCapability dpasCap =
        ttgi::DpasEncodingAttr::getDPASCapability(mod);
    Type elemType = oldAType.getElementType();
    unsigned opsPerChan = getOpsPerChannel(elemType, mod);
    SmallVector<unsigned> warpsPerTile =
        getWarpsPerTile(op, dpasCap, retShape, numWarps);
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
        threadsPerWarp,
        dpasType == ttgi::DPASAnalysis::DPASEngineType::FP32_FP32_FP4_FP4
            ? std::make_optional(2)
            : std::nullopt);

    if (!dpasCap.isATSM()) {
      // Adjust repCluster size for arch after ATSM which has 2d block IO
      // capability.
      unsigned dpasElemBitWidths =
          oldAType.getElementType().getIntOrFloatBitWidth();

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
          threadsPerWarp,
          dpasType == ttgi::DPASAnalysis::DPASEngineType::FP32_FP32_FP4_FP4
              ? std::make_optional(2)
              : std::nullopt);
    }

    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), dpasEnc);

    // convert accumulator
    TensorValue oldAcc = op.getC();
    auto newAcc = ttg::ConvertLayoutOp::create(rewriter, oldAcc.getLoc(),
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

    a = ttg::ConvertLayoutOp::create(rewriter, a.getLoc(), newAType, a);
    b = ttg::ConvertLayoutOp::create(rewriter, b.getLoc(), newBType, b);

    Value res = nullptr;
    if constexpr (std::is_same<OpTy, tt::DotScaledOp>::value) {
      MLIRContext *ctx = rewriter.getContext();
      TensorValue scaleA = op.getAScale();
      if (scaleA) {
        tt::LinearLayout scaleALayout = BlockScaledDPAStoLinearLayout(
            scaleA.getType().getShape(), dpasEnc, 3);
        auto newScaleAType = RankedTensorType::get(
            scaleA.getType().getShape(), scaleA.getType().getElementType(),
            ttg::LinearEncodingAttr::get(ctx, scaleALayout));
        scaleA = ttg::ConvertLayoutOp::create(rewriter, scaleA.getLoc(),
                                              newScaleAType, scaleA);
      }
      TensorValue scaleB = op.getBScale();
      if (scaleB) {
        tt::LinearLayout scaleBLayout = BlockScaledDPAStoLinearLayout(
            scaleB.getType().getShape(), dpasEnc, 4);
        auto newScaleBType = RankedTensorType::get(
            scaleB.getType().getShape(), scaleB.getType().getElementType(),
            ttg::LinearEncodingAttr::get(ctx, scaleBLayout));
        scaleB = ttg::ConvertLayoutOp::create(rewriter, scaleB.getLoc(),
                                              newScaleBType, scaleB);
      }
      auto newOp = tt::DotScaledOp::create(
          rewriter, op.getLoc(), newRetType, a, b, newAcc, scaleA, scaleB,
          op.getAElemType(), op.getBElemType(), op.getFastMath());
      res = newOp.getResult();
    } else if constexpr (std::is_same<OpTy, tt::DotOp>::value) {
      auto newOp =
          tt::DotOp::create(rewriter, op.getLoc(), newRetType, a, b, newAcc,
                            op.getInputPrecision(), op.getMaxNumImpreciseAcc());
      res = newOp.getResult();
    }
    assert(res && "Expecting a valid value");

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType, res);
    return success();
  }
};

class UpcastScaledBlocked : public OpRewritePattern<tt::DotScaledOp> {

public:
  using OpRewritePattern<tt::DotScaledOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::DotScaledOp scaledDotOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = scaledDotOp.getType();
    if (!oldRetType.getEncoding() ||
        isa<ttgi::DpasEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    tt::ScaleDotElemType precA;
    tt::ScaleDotElemType precB;
    if (!(scaledDotOp.getRhsKPack() && scaledDotOp.getLhsKPack())) {
      // BDPAS only supports the fp4 which is packed along K for the A and B
      // matrices. Upcast A and B to unpack the FP4 which is packed on non-k
      // dim. FP16 is chosen here because:
      // 1. It can represent all FP4 values exactly (without rounding)
      // 2. Sufficient range for the FP4.
      // 3. The Fp4ToFp only supports upcast to fp16/bf16 for now.
      precA = precB = tt::ScaleDotElemType::FP16;
    } else {
      // Upcast A and B for mixed types.
      std::optional<std::tuple<tt::ScaleDotElemType, tt::ScaleDotElemType>>
          computeType = getComputeType(scaledDotOp.getAElemType(),
                                       scaledDotOp.getBElemType(), rewriter);
      if (!computeType)
        return failure();
      precA = std::get<0>(*computeType);
      precB = std::get<1>(*computeType);
    }

    TypedValue<RankedTensorType> A =
        upcastMatrix(rewriter, scaledDotOp, 0, precA);
    TypedValue<RankedTensorType> B =
        upcastMatrix(rewriter, scaledDotOp, 1, precB);
    auto newDot = tt::DotScaledOp::create(
        rewriter, scaledDotOp.getLoc(), scaledDotOp->getResultTypes(), A, B,
        scaledDotOp.getC(), scaledDotOp.getAScale(), scaledDotOp.getBScale(),
        precA, precB, scaledDotOp.getFastMath(), scaledDotOp.getLhsKPack(),
        scaledDotOp.getRhsKPack());

    rewriter.replaceOp(scaledDotOp, newDot);
    return success();
  }

private:
  static std::optional<unsigned>
  getScaleDotElemTypeBitWidth(tt::ScaleDotElemType type) {
    switch (type) {
    case tt::ScaleDotElemType::E2M1:
      return 4;
    case tt::ScaleDotElemType::E4M3:
    case tt::ScaleDotElemType::E5M2:
      return 8;
    case tt::ScaleDotElemType::BF16:
    case tt::ScaleDotElemType::FP16:
      return 16;
    default:
      // For other unsupported float types.
      return std::nullopt;
    }
  };

  // Retrieve the precision type of matrix A and B supported by bdpas for mixed
  // tt.dot_scaled operations.
  std::optional<std::tuple<tt::ScaleDotElemType, tt::ScaleDotElemType>>
  getComputeType(tt::ScaleDotElemType aType, tt::ScaleDotElemType bType,
                 PatternRewriter &rewriter) const {
    if (aType == bType) // Skip the dot_scaled which is not mixed precision.
      return std::nullopt;
    std::optional<unsigned> aBitWidth = getScaleDotElemTypeBitWidth(aType);
    std::optional<unsigned> bBitWidth = getScaleDotElemTypeBitWidth(bType);
    if (!aBitWidth || !bBitWidth) // unsupported type.
      return std::nullopt;
    unsigned minBitWidth = std::min(*aBitWidth, *bBitWidth);
    unsigned maxBitWidth = std::max(*aBitWidth, *bBitWidth);
    if (minBitWidth < maxBitWidth) {
      // align to the larger bit width type.
      if (minBitWidth == 4) {
        // There is limitation in Fp4ToFpOp that it only supports to upcast to
        // fp16/bf16.
        if (aType == tt::ScaleDotElemType::FP16 ||
            bType == tt::ScaleDotElemType::FP16)
          return std::make_tuple(tt::ScaleDotElemType::FP16,
                                 tt::ScaleDotElemType::FP16);
        else
          return std::make_tuple(tt::ScaleDotElemType::BF16,
                                 tt::ScaleDotElemType::BF16);
      }

      if (aBitWidth > bBitWidth) {
        return std::make_tuple(aType, aType);
      } else {
        return std::make_tuple(bType, bType);
      }
    } else {
      // align to the type with larger range.
      assert(minBitWidth != 4 &&
             "invalid packed dot_scaled with different fp4");

      if (minBitWidth == 8) {
        // BDPAS support mixed fp8 natively.
        return std::nullopt;
      }

      if (minBitWidth == 16) {
        return std::make_tuple(tt::ScaleDotElemType::BF16,
                               tt::ScaleDotElemType::BF16);
      }
    }
    return std::nullopt;
  }

  // Upcast the matrix A or B of the tt.dot_scaled.
  TypedValue<RankedTensorType>
  upcastMatrix(PatternRewriter &rewriter, tt::DotScaledOp scaledDotOp,
               int opIdx, tt::ScaleDotElemType computeType) const {
    TypedValue<RankedTensorType> v =
        opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
    TypedValue<RankedTensorType> res = scaledDotOp.getD();
    bool isFp4 =
        tt::ScaleDotElemType::E2M1 ==
        (opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType());

    Location loc = v.getLoc();
    int64_t rank = v.getType().getRank();
    int64_t kDim = opIdx == 0 ? rank - 1 : rank - 2;

    // Upcast value to computeType (fp16/bf16)
    if (isFp4) {
      ArrayRef<int64_t> resShape = res.getType().getShape();
      ArrayRef<int64_t> vShape = v.getType().getShape();
      int64_t packDim = kDim;
      if ((opIdx == 0 && resShape[rank - 2] != vShape[rank - 2]) ||
          (opIdx == 1 && resShape[rank - 1] != vShape[rank - 1])) {
        packDim = (packDim + 1) % 2;
      }
      v = ttg::Fp4ToFpOp::create(rewriter, loc, v,
                                 getScalarType(rewriter, computeType), packDim);
    } else {
      RankedTensorType vType =
          v.getType().clone(getScalarType(rewriter, computeType));
      tt::FpToFpOp op = tt::FpToFpOp::create(rewriter, loc, vType, v);
      v = cast<TypedValue<RankedTensorType>>(op.getResult());
    }
    return v;
  }

  FloatType getScalarType(PatternRewriter &rewriter,
                          tt::ScaleDotElemType computeType) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    switch (computeType) {
    case tt::ScaleDotElemType::BF16:
      return rewriter.getBF16Type();
    case tt::ScaleDotElemType::FP16:
      return rewriter.getF16Type();
    case tt::ScaleDotElemType::E5M2:
      return mlir::Float8E5M2Type::get(ctx);
    case tt::ScaleDotElemType::E4M3:
      return mlir::Float8E4M3FNType::get(ctx);
    case tt::ScaleDotElemType::E2M1:
      return mlir::Float4E2M1FNType::get(ctx);
    default:
      assert(false && "unsupported precision type");
    }
    return {};
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
        return tt::FpToFpOp::create(builder, loc, tensorPromotedType, operand);
      })
      .Case<IntegerType>([&](auto) {
        unsigned tgtBitWidth = elemType.getIntOrFloatBitWidth(),
                 valBitWidth = cast<RankedTensorType>(operand.getType())
                                   .getElementTypeBitWidth();
        Operation *castOp =
            (valBitWidth <= tgtBitWidth)
                ? arith::ExtSIOp::create(builder, loc, tensorPromotedType,
                                         operand)
                : arith::TruncIOp::create(builder, loc, tensorPromotedType,
                                          operand);
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
        dyn_cast<ttgi::DpasEncodingAttr>(D.getType().getEncoding());

    Type promoteType;
    if (dpasLayout) {
      bool isNativeFP8 = isa<Float8E5M2Type, Float8E4M3FNType>(AElType);
      // fp8 is not always natively supported by the the DPAS instruction,
      // promote it to fp16 when necessary.

      auto m = dotOp->getParentOfType<ModuleOp>();
      bool supportsFP8 = m->hasAttr(
          ttgi::TritonIntelGPUDialect::getSupportBlockScaleDPASAttrName());
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
        tt::TransOp::create(builder, result.getLoc(), result, ArrayRef({1, 0}));
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
          auto initTranspose = tt::TransOp::create(
              builder, forOp.getLoc(), init.get(), ArrayRef({1, 0}));
          init.set(initTranspose);
        }
      }
    }
  }
}

static tt::TransOp transposeDotScaleOp(tt::DotScaledOp dotOp) {
  assert(dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr &&
         "Transpose DotOp expects scale on RHS");
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getA();
  std::array<int, 2> transOrder = {1, 0};
  auto lhsTransposed =
      tt::TransOp::create(builder, lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getB();
  auto rhsTransposed =
      tt::TransOp::create(builder, rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  auto cTransposed = tt::TransOp::create(builder, c.getLoc(), c, transOrder);
  auto result = tt::DotScaledOp::create(
      builder, dotOp.getLoc(), cTransposed.getType(), rhsTransposed,
      lhsTransposed, cTransposed, dotOp.getBScale(), dotOp.getAScale(),
      dotOp.getBElemType(), dotOp.getAElemType(), dotOp.getFastMath());
  auto transOp =
      tt::TransOp::create(builder, result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transOp.getOperation());
  dotOp.erase();
  return transOp;
}

static void transposeDotScale(ModuleOp m) {
  SmallVector<tt::DotScaledOp> toTranspose;
  m.walk([&](tt::DotScaledOp dotOp) -> void {
    if (dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr)
      toTranspose.push_back(dotOp);
  });
  SmallVector<tt::TransOp> transposes;
  for (tt::DotScaledOp &dotOp : toTranspose) {
    tt::TransOp transpose = transposeDotScaleOp(dotOp);
    transposes.push_back(transpose);
  }

  for (tt::TransOp transpose : transposes) {
    sinkTransposeOp(transpose);
  }
}

class TritonIntelGPUAccelerateMatmulPass
    : public ttgi::impl::TritonIntelGPUAccelerateMatmulBase<
          TritonIntelGPUAccelerateMatmulPass> {
public:
  using ttgi::impl::TritonIntelGPUAccelerateMatmulBase<
      TritonIntelGPUAccelerateMatmulPass>::TritonIntelGPUAccelerateMatmulBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    auto &dpasAnalysis = getAnalysis<ttgi::DPASAnalysis>();

    // Transpose dot scale operations that have a scale on the RHS.
    bool supportBlockScaleDPAS = m->hasAttr(
        ttgi::TritonIntelGPUDialect::getSupportBlockScaleDPASAttrName());
    if (!supportBlockScaleDPAS)
      transposeDotScale(m);

    RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    patterns.add<BlockedToDPAS<tt::DotOp>>(context, dpasAnalysis,
                                           benefitDefault + 1);
    if (supportBlockScaleDPAS) {
      patterns.add<BlockedToDPAS<tt::DotScaledOp>>(context, dpasAnalysis,
                                                   benefitDefault + 1);
      patterns.add<UpcastScaledBlocked>(context, benefitDefault + 1);
    }
    ttgi::populateDecomposeScaledBlockedPatterns(patterns, benefitDefault);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    decomposeMixedModeDotOp(m);
  }
};
