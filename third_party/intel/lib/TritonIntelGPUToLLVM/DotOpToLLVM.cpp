#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

namespace fma_details {
template <typename OpTy>
LogicalResult convertDPAS(OpTy op, typename OpTy::Adaptor adaptor,
                          TritonIntelGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter,
                          const TargetInfoBase &targetInfo);
} // namespace fma_details

namespace {
struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  DotOpConversion(LLVMTypeConverter &converter,
                  const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::DotOp>(converter, benefit),
        targetInfo(targetInfo) {}

  const TargetInfoBase &targetInfo;
  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    if (!isOuter && isa<DpasEncodingAttr>(
                        cast<RankedTensorType>(D.getType()).getEncoding())) {
      return fma_details::convertDPAS(op, adaptor, this->getTypeConverter(),
                                      rewriter, targetInfo);
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct DotScaledOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::DotScaledOp> {
  DotScaledOpConversion(LLVMTypeConverter &converter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::DotScaledOp>(converter,
                                                             benefit),
        targetInfo(targetInfo) {}

  const TargetInfoBase &targetInfo;

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value D = op.getResult();
    if (isa<DpasEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding())) {
      return fma_details::convertDPAS(op, adaptor, getTypeConverter(), rewriter,
                                      targetInfo);
    }

    llvm::report_fatal_error(
        "Unsupported DotScaledOp found when converting TritonGPU to LLVM.");
  }
};
} // namespace

void mlir::triton::intel::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<DotOpConversion, DotScaledOpConversion>(typeConverter,
                                                       targetInfo, benefit);
}
