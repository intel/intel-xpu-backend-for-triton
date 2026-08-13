// Per-element LLVM IR lowering for ttig.upcast_scaled.
//
// This lowering operates on individual scalar elements (unpacked from the LLVM
// struct representation of the tensor), emitting a tight sequence of i16
// instructions for each element.  By working at the scalar level, the
// intermediate values (sign, magnitude, biasedMag) have micro-instruction-scale
// live ranges that LLVM's register allocator handles without the
// 4-GRF-tensor-chunk pressure that caused register spill in the MLIR-level
// exponent-add attempts.
//
// Algorithm (per element, given bf16 operand and E8M0 scale byte):
//   i16 = bitcast(bf16)
//   sign     = i16 & 0x8000
//   mag      = i16 & 0x7FFF
//   scaleI16 = zext(scaleByte, i16)
//   shifted  = scaleI16 << 7
//   sum      = mag + shifted
//   biased   = sum - 0x3F80
//   satInf   = ugt(biased, 0x7F80) ? 0x7F80 : biased   // clamp overflow → ±Inf
//   satZero  = uge(biased, 0xC080) ? 0 : satInf          // flush underflow →
//   ±0 result   = sign | satZero (if !fast_math): result = eq(scaleByte, 0xFF)
//   ? NaN : result return bitcast(result, bf16)
//
// Reference: AMD ScaledUpcastToLLVM.cpp for the unpack/pack idiom.
// Reference: ~/bdpas-sim-optimization/opportunity1-attempt1.md for the
// register-
//            pressure analysis that motivated this approach.

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::LLVM::intel;
using namespace mlir::triton;
using namespace mlir::triton::intel;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

namespace {

// Helper: create a scalar i16 constant.
static Value mkI16(ConversionPatternRewriter &rewriter, Location loc,
                   uint16_t val) {
  auto i16Ty = rewriter.getIntegerType(16);
  return LLVM::ConstantOp::create(rewriter, loc, i16Ty,
                                  rewriter.getIntegerAttr(i16Ty, val));
}

// Helper: create a scalar i8 constant.
static Value mkI8(ConversionPatternRewriter &rewriter, Location loc,
                  uint8_t val) {
  auto i8Ty = rewriter.getIntegerType(8);
  return LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                  rewriter.getIntegerAttr(i8Ty, val));
}

class UpcastScaledOpPattern : public ConvertOpToLLVMPattern<UpcastScaledOp> {
public:
  UpcastScaledOpPattern(LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<UpcastScaledOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(UpcastScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    int32_t scaleFactor = op.getScaleFactor();
    bool fastMath = op.getFastMath().has_value();

    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType scaleTy = op.getScale().getType();
    if (!srcTy.getElementType().isBF16())
      return rewriter.notifyMatchFailure(op, "only bf16 src is supported");
    if (!srcTy.getEncoding() || !scaleTy.getEncoding())
      return rewriter.notifyMatchFailure(op, "operands must have encodings");

    StringAttr kReg = StringAttr::get(ctx, "register");
    StringAttr kLane = StringAttr::get(ctx, "lane");
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    StringAttr kBlock = StringAttr::get(ctx, "block");

    // The LLVM struct holds one element per *unique* register, i.e. one per
    // register basis of the register-broadcast-stripped layout (see
    // TypeConverter.cpp:47-55 -> getUniqueElemsPerThread). So the layouts used
    // to build the index map must be stripped the same way, or index i into
    // the unpacked vector does not correspond to register i of the layout.
    SmallVector<Value> srcElems =
        unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> scaleElems =
        unpackUniqueTensorElements(loc, adaptor.getScale(), rewriter);

    LinearLayout srcLL = toLinearLayout(srcTy).removeZeroBasesAlongDim(kReg);
    LinearLayout scaleLL =
        toLinearLayout(scaleTy).removeZeroBasesAlongDim(kReg);
    assert(srcLL.getInDimSize(kReg) == (int32_t)srcElems.size());
    assert(scaleLL.getInDimSize(kReg) == (int32_t)scaleElems.size());

    // src (register, lane, warp, block) -> scale tensor coordinates ...
    auto srcToScale =
        UpcastScaledOp::computeScaleLayout(srcLL, op.getAxis(), scaleFactor);
    if (!srcToScale)
      return rewriter.notifyMatchFailure(op, "cannot derive scale layout");
    // ... then scale coordinates -> scale (register, lane, warp, block).
    LinearLayout srcRegToScaleReg = srcToScale->invertAndCompose(scaleLL);

    // The map is only usable as a compile-time per-register lookup if the
    // scale register index does not depend on lane/warp/block. That holds iff
    // the scale operand really carries the derived compact layout. The verifier
    // guarantees it; check here so that a stale layout is a legalization
    // failure instead of silently wrong code (invertAndCompose's own asserts
    // do NOT catch this -- see below).
    for (StringAttr inDim : {kLane, kWarp, kBlock})
      for (int b = 0, e = srcRegToScaleReg.getInDimSizeLog2(inDim); b < e; ++b)
        if (srcRegToScaleReg.getBasis(inDim, b, kReg) != 0)
          return rewriter.notifyMatchFailure(
              op, "scale layout is not the compact layout derived from src");

    // Check if mapping is simple (consecutive groups of scaleFactor elements).
    // Simple = scale register index is just floor(src_register / scaleFactor).
    bool isSimpleMapping = true;
    for (int32_t i = 0, n = std::min(static_cast<int32_t>(srcElems.size()),
                                     scaleFactor * 4);
         i < n && isSimpleMapping; ++i) {
      auto coord = srcRegToScaleReg.apply(
          {{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      auto it = llvm::find_if(coord,
                              [&](const auto &kv) { return kv.first == kReg; });
      assert(it != coord.end());
      if (it->second != i / scaleFactor)
        isSimpleMapping = false;
    }

    // For simple mappings, use runtime division (zero constants).
    // For complex mappings, precompute minimal table (group leaders only).
    SmallVector<int32_t> scaleRegForSrcReg;
    if (!isSimpleMapping) {
      scaleRegForSrcReg.reserve(srcElems.size());
      for (int32_t i = 0, n = srcElems.size(); i < n; ++i) {
        auto coord = srcRegToScaleReg.apply(
            {{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        auto it = llvm::find_if(
            coord, [&](const auto &kv) { return kv.first == kReg; });
        scaleRegForSrcReg.push_back(it->second);
      }
    }

    auto i16Ty = rewriter.getIntegerType(16);
    auto i8Ty = rewriter.getIntegerType(8);
    auto bf16Ty = rewriter.getBF16Type();

    // Pre-built constants (created once, reused for each element).
    Value cst_7 = mkI16(rewriter, loc, 7);
    Value cst_7FFF = mkI16(rewriter, loc, 0x7FFF);
    Value cst_8000 = mkI16(rewriter, loc, 0x8000u);
    Value cst_3F80 = mkI16(rewriter, loc, 0x3F80);
    Value cst_7F80 = mkI16(rewriter, loc, 0x7F80);
    Value cst_C080 = mkI16(rewriter, loc, 0xC080u);
    Value cst_0 = mkI16(rewriter, loc, 0);
    // bf16 canonical quiet NaN (0x7FC0) as i16 for the 0xFF-scale NaN mask.
    Value cst_NaN = mkI16(rewriter, loc, 0x7FC0);
    Value cst_ff = mkI8(rewriter, loc, 0xFFu);

    SmallVector<Value> results;
    results.reserve(srcElems.size());

    for (int i = 0, n = srcElems.size(); i < n; ++i) {
      Value elemBf16 = srcElems[i];
      int32_t scaleIdx =
          isSimpleMapping ? (i / scaleFactor) : scaleRegForSrcReg[i];
      Value scaleByte = scaleElems[scaleIdx];

      // Bitcast bf16 → i16.
      Value elemI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, elemBf16);

      // Extract sign and magnitude.
      Value sign = LLVM::AndOp::create(rewriter, loc, elemI16, cst_8000);
      Value mag = LLVM::AndOp::create(rewriter, loc, elemI16, cst_7FFF);

      // Decode E8M0 scale: zext i8 → i16, shift left 7.
      Value scaleI16 = LLVM::ZExtOp::create(rewriter, loc, i16Ty, scaleByte);
      Value shifted = LLVM::ShlOp::create(rewriter, loc, scaleI16, cst_7);

      // biasedMag = magnitude + scaleShifted - 0x3F80.
      Value sum = LLVM::AddOp::create(rewriter, loc, mag, shifted);
      Value biased = LLVM::SubOp::create(rewriter, loc, sum, cst_3F80);

      // Overflow saturation: if biased > 0x7F80 (unsigned), clamp to 0x7F80.
      Value gt_inf = LLVM::ICmpOp::create(
          rewriter, loc, LLVM::ICmpPredicate::ugt, biased, cst_7F80);
      Value satInf =
          LLVM::SelectOp::create(rewriter, loc, gt_inf, cst_7F80, biased);

      // Underflow saturation: if biased >= 0xC080 (unsigned), flush to 0.
      Value uge_under = LLVM::ICmpOp::create(
          rewriter, loc, LLVM::ICmpPredicate::uge, biased, cst_C080);
      Value satZero =
          LLVM::SelectOp::create(rewriter, loc, uge_under, cst_0, satInf);

      // Restore sign bit.
      Value resultI16 = LLVM::OrOp::create(rewriter, loc, sign, satZero);

      // NaN propagation: scale byte 0xFF → output NaN (MXFP spec requirement).
      // Skipped when fast_math is set (caller guarantees 0xFF never occurs).
      if (!fastMath) {
        Value isNaN = LLVM::ICmpOp::create(
            rewriter, loc, LLVM::ICmpPredicate::eq, scaleByte, cst_ff);
        resultI16 =
            LLVM::SelectOp::create(rewriter, loc, isNaN, cst_NaN, resultI16);
      }

      results.push_back(
          LLVM::BitcastOp::create(rewriter, loc, bf16Ty, resultI16));
    }

    rewriter.replaceOp(op, packUniqueTensorElements(loc, getTypeConverter(),
                                                    results, rewriter,
                                                    op.getType()));
    return success();
  }
};

} // anonymous namespace

void mlir::triton::intel::populateUpcastScaledToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<UpcastScaledOpPattern>(typeConverter, benefit);
}
