#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

namespace fma_details {
LogicalResult convertDPAS(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonIntelGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);
template <typename OpTy>
LogicalResult convertDPAS(OpTy op, typename OpTy::Adaptor adaptor,
                          TritonIntelGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter,
                          const TargetInfoBase &targetInfo);
} // namespace fma_details

namespace {
template <typename OpTy>
struct DotLikeOpConversion : public ConvertTritonGPUOpToLLVMPattern<OpTy> {

  DotLikeOpConversion(LLVMTypeConverter &converter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<OpTy>(converter, benefit),
        targetInfo(targetInfo) {}

  const TargetInfoBase &targetInfo;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    if (!isOuter && isa<DpasEncodingAttr>(
                        cast<RankedTensorType>(D.getType()).getEncoding())) {
      return fma_details::convertDPAS(op, adaptor, this->getTypeConverter(),
                                      rewriter, targetInfo);
    }

    if constexpr (std::is_same<OpTy, DotOp>::value) {
      if (isa<BlockedEncodingAttr>(
              cast<RankedTensorType>(D.getType()).getEncoding()))
        return convertFMADot(op, adaptor, this->getTypeConverter(), rewriter);
    }

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};
} // namespace

void mlir::triton::intel::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<DotLikeOpConversion<DotOp>, DotLikeOpConversion<DotScaledOp>>(
      typeConverter, targetInfo, benefit);
}
