// Per-element LLVM IR lowering for ttig.upcast_scaled.
//
// Uses the AMD software fallback pattern (ScaledUpcastToLLVM.cpp, Path C):
// process each bf16 element and its E8M0 scale byte individually in f32,
// allowing IGC's register allocator to reuse registers across elements.
//
// Algorithm (per element):
//   vF32    = bitcast(shl(zext(bitcast(src_bf16, i16), i32), 16), f32)
//   scaleF32 = bitcast(shl(zext(scale_i8, i32), 23), f32)
//   mulF32  = fmul(vF32, scaleF32)   // f32 handles ±Inf/±0 naturally
//   result  = bitcast(trunc(i16, lshr(bitcast(mulF32, i32), 16)), bf16)
//
// NaN (scale=0xFF) is handled by the maskNan call in DecomposeScaledBlocked.
//
// Reference: AMD ScaledUpcastToLLVM.cpp Path C for the algorithm.

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

    // Build compile-time scale register index for each src register.
    // Always use LinearLayout — no isSimpleMapping heuristic that may miss
    // elements or misclassify DPAS layouts.
    SmallVector<int32_t> scaleRegForSrcReg;
    scaleRegForSrcReg.reserve(srcElems.size());
    for (int32_t i = 0, n = srcElems.size(); i < n; ++i) {
      auto coord = srcRegToScaleReg.apply(
          {{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      auto it = llvm::find_if(coord,
                              [&](const auto &kv) { return kv.first == kReg; });
      assert(it != coord.end());
      scaleRegForSrcReg.push_back(it->second);
    }

    auto i16Ty = rewriter.getIntegerType(16);
    auto i32Ty = rewriter.getIntegerType(32);
    auto bf16Ty = rewriter.getBF16Type();
    auto f32Ty = rewriter.getF32Type();

    // Per-element AMD-style f32 multiply (Path C from AMD ScaledUpcastToLLVM).
    // vF32    = bitcast(shl(zext(bitcast(src_bf16, i16), i32), 16), f32)
    // scaleF32 = bitcast(shl(zext(scale_i8, i32), 23), f32)
    // mulF32  = fmul(vF32, scaleF32)
    // result  = bitcast(trunc(i16, lshr(bitcast(mulF32, i32), 16)), bf16)
    //
    // All three temporaries (vF32, scaleF32, mulF32) die before the next
    // iteration — IGC can reuse the same physical registers across elements,
    // avoiding the large simultaneous f32 tensor liveness that causes spill.
    // NaN (scale=0xFF) is handled by maskNan at the MLIR level.
    auto cst16 = [&](uint32_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v));
    };
    Value shift16 = cst16(16);
    Value shift23 = cst16(23);

    SmallVector<Value> results;
    results.reserve(srcElems.size());

    for (int i = 0, n = srcElems.size(); i < n; ++i) {
      Value scaleByte = scaleElems[scaleRegForSrcReg[i]];

      // bf16 → f32 via upper-bits: shl(zext(bitcast(bf16, i16), i32), 16)
      Value vI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, srcElems[i]);
      Value vI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, vI16);
      Value vF32 = LLVM::BitcastOp::create(
          rewriter, loc, f32Ty,
          LLVM::ShlOp::create(rewriter, loc, vI32, shift16));

      // E8M0 i8 → f32: shl(zext(scale_i8, i32), 23) puts scale into exponent
      Value scI32 = LLVM::ZExtOp::create(rewriter, loc, i32Ty, scaleByte);
      Value scF32 = LLVM::BitcastOp::create(
          rewriter, loc, f32Ty,
          LLVM::ShlOp::create(rewriter, loc, scI32, shift23));

      // f32 multiply — ±Inf, ±0, subnormal all handled naturally by IEEE f32
      Value mulF32 = LLVM::FMulOp::create(rewriter, loc, vF32, scF32);

      // f32 → bf16 via upper bits: trunc(i16, lshr(bitcast(mulF32, i32), 16))
      Value mulI32 = LLVM::BitcastOp::create(rewriter, loc, i32Ty, mulF32);
      Value mulI16 = LLVM::TruncOp::create(
          rewriter, loc, i16Ty,
          LLVM::LShrOp::create(rewriter, loc, mulI32, shift16));
      results.push_back(LLVM::BitcastOp::create(rewriter, loc, bf16Ty, mulI16));
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
