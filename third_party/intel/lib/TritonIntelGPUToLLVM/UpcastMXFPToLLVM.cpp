#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu::intel;

namespace {
SmallVector<Value> convertMxfp4x2ToBf16x2(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> values) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> results;
  for (auto v : values) {
    auto em0 = b.and_(v, b.i8_val(0x7));
    auto em1 = b.and_(v, b.i8_val(0x70));
    Value v0 =
        b.or_(b.shl(b.zext(i16_ty, em0), b.i16_val(6)),
              b.shl(b.zext(i16_ty, b.and_(v, b.i8_val(0x8))), b.i16_val(12)));
    Value v1 =
        b.or_(b.shl(b.zext(i16_ty, em1), b.i16_val(2)),
              b.shl(b.zext(i16_ty, b.and_(v, b.i8_val(0x80))), b.i16_val(8)));
    // Three cases:
    // 1) x is normal and non-zero: Correct bias
    v0 = b.select(b.icmp_ne(b.and_(em0, b.i8_val(0x6)), b.i8_val(0)),
                  b.add(v0, b.i16_val((127 - 1) << 7)), v0);
    v1 = b.select(b.icmp_ne(b.and_(em1, b.i8_val(0x60)), b.i8_val(0)),
                  b.add(v1, b.i16_val((127 - 1) << 7)), v1);
    // 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in
    // bf16
    v0 = b.bitcast(
        b.select(b.icmp_eq(em0, b.i8_val(0x1)),
                 b.or_(b.i16_val(16128), b.and_(v0, b.i16_val(0x8000))), v0),
        bf16_ty);
    v1 = b.bitcast(
        b.select(b.icmp_eq(em1, b.i8_val(0x10)),
                 b.or_(b.i16_val(16128), b.and_(v1, b.i16_val(0x8000))), v1),
        bf16_ty);
    // 3) x is zero, nothing to do
    results.push_back(v0);
    results.push_back(v1);
  }
  return results;
}

SmallVector<Value> convertMxfp4x2ToFp16x2(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> values) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> results;
  for (auto v : values) {
    auto em0 = b.and_(v, b.i8_val(0x7));
    auto em1 = b.and_(v, b.i8_val(0x70));
    // FP16 bits: sign = 1, exponent = 5, mantissa = 10
    Value v0 =
        b.or_(b.shl(b.zext(i16_ty, em0), b.i16_val(10 - 1)),
              b.shl(b.zext(i16_ty, b.and_(v, b.i8_val(0x8))), b.i16_val(12)));
    Value v1 =
        b.or_(b.shl(b.zext(i16_ty, em1), b.i16_val(10 - 1 - 4)),
              b.shl(b.zext(i16_ty, b.and_(v, b.i8_val(0x80))), b.i16_val(8)));

    // Three cases:
    // 1) x is normal and non-zero: Correct bias
    v0 = b.select(b.icmp_ne(b.and_(em0, b.i8_val(0x6)), b.i8_val(0)),
                  b.add(v0, b.i16_val((15 - 1) << 10)), v0);
    v1 = b.select(b.icmp_ne(b.and_(em1, b.i8_val(0x60)), b.i8_val(0)),
                  b.add(v1, b.i16_val((15 - 1) << 10)), v1);

    // 2) x is subnormal (x == 0bs001 where s is the sign): Map to fp16 +-0.5
    v0 = b.bitcast(
        b.select(b.icmp_eq(em0, b.i8_val(0x1)),
                 b.or_(b.i16_val(0x3800), b.and_(v0, b.i16_val(0x8000))), v0),
        f16_ty);
    v1 = b.bitcast(
        b.select(b.icmp_eq(em1, b.i8_val(0x10)),
                 b.or_(b.i16_val(0x3800), b.and_(v1, b.i16_val(0x8000))), v1),
        f16_ty);
    // 3) x is zero, nothing to do
    results.push_back(v0);
    results.push_back(v1);
  }
  return results;
}

Value mxfpScaleFp16(ConversionPatternRewriter &rewriter, Location loc, Value v,
                    Value scale, bool fastMath) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value scaleF32 =
      b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
  Value scaleF16 = LLVM::intel::convertFp32ToFp16(loc, rewriter, scaleF32,
                                                  RoundingMode::RTNE);
  Value mulF16 = b.fmul(v, scaleF16);
  if (fastMath)
    return mulF16;
  // Account for NaN in the scale as per the mxfp specification.
  Value scaleIsNan = b.icmp_eq(scale, b.i8_val(0xff));
  Value nanF16 = b.bitcast(b.i16_val(0x7c01), f16_ty);
  return b.select(scaleIsNan, nanF16, b.bitcast(mulF16, f16_ty));
};

static Value mxfpScaleBf16(ConversionPatternRewriter &rewriter, Location loc,
                           Value v, Value scale, bool fastMath) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value vBf16 = b.bitcast(v, bf16_ty);
  Value scaleBf16 =
      b.bitcast(b.shl(b.zext(i16_ty, scale), b.i16_val(7)), bf16_ty);

  Value v0 = mlir::triton::intel::convertBf16ToFp32(loc, rewriter, vBf16);
  Value v1 = mlir::triton::intel::convertBf16ToFp32(loc, rewriter, scaleBf16);
  auto result = rewriter.create<LLVM::FMulOp>(loc, f32_ty, v0, v1);
  auto undefRounding = static_cast<mlir::triton::RoundingMode>(-1);
  Value scaledBf16 = mlir::triton::intel::convertFp32ToBf16(
      loc, rewriter, result, undefRounding);
  if (fastMath)
    return scaledBf16;
  // Account for NaN in the scale as per the mxfp specification.
  Value scaleIsNan = b.icmp_eq(scale, b.i8_val(0xff));
  Value nanBf16 = b.bitcast(b.i16_val(0x7fff), bf16_ty);
  return b.select(scaleIsNan, nanBf16, scaledBf16);
};

class UpcastMXFPOpPattern : public ConvertOpToLLVMPattern<UpcastMXFPOp> {
private:
  const TargetInfoBase &targetInfo;

public:
  UpcastMXFPOpPattern(LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<UpcastMXFPOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(UpcastMXFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto operands = adaptor.getOperands();
    SmallVector<Value> xVals = unpackLLElements(loc, operands[0], rewriter);
    SmallVector<Value> scaleVals = unpackLLElements(loc, operands[1], rewriter);
    ScaleDotElemType fpType = op.getFpType();

    Value tid = b.tid_val();
    auto mod = op->getParentOfType<ModuleOp>();
    Value warpSize =
        b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpId = b.udiv(tid, warpSize);
    Value laneId = b.urem(tid, warpSize);

    bool useFp16 = op.getType().getElementType().isF16();
    if (fpType == ScaleDotElemType::E2M1) {
      xVals = useFp16 ? convertMxfp4x2ToFp16x2(rewriter, loc, xVals)
                      : convertMxfp4x2ToBf16x2(rewriter, loc, xVals);
    }

    auto xType = cast<RankedTensorType>(op->getOperandTypes()[0]);
    auto dotEnc = cast<DotOperandEncodingAttr>(xType.getEncoding());
    // For RHS dot operand with dot layout encoding, each thread access tensor
    // elements in column which require scaling values from threads across
    // warps. We can only access the scaling values by shuffle in the same warp.
    assert(dotEnc.getOpIdx() == 0 && "NYI: rhs scale with dot encoding");

    auto dpasEnc = cast<DpasEncodingAttr>(dotEnc.getParent());
    unsigned instShapeM = dpasEnc.getDPASInstShapeA()[0];
    unsigned instShapeK = dpasEnc.getDPASInstShapeA()[1];
    constexpr unsigned scalingBlockSize = 32;
    unsigned repSize = scalingBlockSize / instShapeK;
    unsigned subTileSize = instShapeM;
    // kWidth here is the contiguous number of elements each thread access.
    unsigned kWidth = dpasEnc.getOpsPerChannel() / 2;
    unsigned numMxfp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod) / instShapeM;
    unsigned mxfpSize = repSize * subTileSize * kWidth;
    constexpr unsigned numScales = 16;

    if (fpType == ScaleDotElemType::E2M1) {
      repSize /= 2; // for E2M1, actual scaling block size is 16
      kWidth *= 2;  // 2 fp4 are packed in one i8
    }

    Value c = b.udiv(laneId, b.i32_val(numScales));
    SmallVector<Value, 16> ci;
    for (int row = 0; row < numMxfp; ++row)
      for (int col = 0; col < subTileSize; ++col)
        ci.emplace_back(b.add(c, b.i32_val(row + 2 * col)));

    for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
      for (int mxfp = 0; mxfp < numMxfp; ++mxfp) {
        SmallVector<Value, 8> si;
        for (int subTile = 0; subTile < 8; ++subTile)
          si.emplace_back(targetInfo.shuffleIdx(rewriter, loc, scaleVal,
                                                ci[8 * mxfp + subTile]));
        for (int rep = 0; rep < repSize; ++rep) {
          for (int subTile = 0; subTile < subTileSize; ++subTile) {
            for (int k = 0; k < kWidth; ++k) {
              unsigned idx = i * scalingBlockSize + mxfp * mxfpSize +
                             rep * subTileSize * kWidth + subTile * kWidth + k;
              xVals[idx] = useFp16
                               ? mxfpScaleFp16(rewriter, loc, xVals[idx],
                                               si[subTile], op.getFastMath())
                               : mxfpScaleBf16(rewriter, loc, xVals[idx],
                                               si[subTile], op.getFastMath());
            }
          }
        }
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::intel::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<UpcastMXFPOpPattern>(typeConverter, targetInfo, benefit);
}
