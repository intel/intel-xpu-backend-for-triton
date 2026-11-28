#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::isPermutationMatrixLayout;
using ::mlir::triton::gpu::toLinearLayout;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

namespace fma_details {
template <typename OpTy>
LogicalResult convertDPAS(OpTy op, typename OpTy::Adaptor adaptor,
                          TritonIntelGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);
LogicalResult convertIntegerFMADot(triton::DotOp op,
                                   triton::DotOp::Adaptor adaptor,
                                   const LLVMTypeConverter *typeConverter,
                                   ConversionPatternRewriter &rewriter);
} // namespace fma_details

namespace {
struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    auto DTensorTy = cast<RankedTensorType>(D.getType());
    if (!isPermutationMatrixLayout(
            toLinearLayout(DTensorTy.getShape(), DTensorTy.getEncoding())))
      return rewriter.notifyMatchFailure(
          op,
          "DotOp result encoding must have a permutation-matrix linear layout");

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    if (!isOuter && isa<DpasEncodingAttr>(DTensorTy.getEncoding())) {
      return fma_details::convertDPAS(op, adaptor, getTypeConverter(),
                                      rewriter);
    }

    if (isa<BlockedEncodingAttr>(DTensorTy.getEncoding())) {
      // FIXME: Drop once every supported driver has IGC 2.42+ (#7854, #9038).
      // Where IGC predates 2.42 (the LTS driver, and rolling drivers before
      // compute-runtime 26.40.40196), lower integer dots to dp4a rather than
      // to a mul/add chain: that IGC folds such a chain back into a dp4a and
      // mis-pairs its two operands (intel/intel-xpu-backend-for-triton#7854).
      // IGC 2.42 fixes the fold, so the generic lowering is used there.
      auto mod = op->getParentOfType<ModuleOp>();
      bool useDp4a =
          mod->hasAttr(
              triton::gpu::intel::TritonIntelGPUDialect::getIsLTSAttrName()) ||
          mod->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                           getUseDp4aIntegerFMAAttrName());
      if (useDp4a && isa<IntegerType>(DTensorTy.getElementType()))
        return fma_details::convertIntegerFMADot(op, adaptor,
                                                 getTypeConverter(), rewriter);
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);
    }

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct DotScaledOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotScaledOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value D = op.getResult();
    if (isa<DpasEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding())) {
      return fma_details::convertDPAS(op, adaptor, getTypeConverter(),
                                      rewriter);
    }

    llvm::report_fatal_error(
        "Unsupported DotScaledOp found when converting TritonGPU to LLVM.");
  }
};
} // namespace

void mlir::triton::intel::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DotOpConversion, DotScaledOpConversion>(typeConverter, benefit);
}
