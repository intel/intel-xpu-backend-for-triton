#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

namespace fma_details {
LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonIntelGPUToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

LogicalResult convertDPAS(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonIntelGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter,
                          const TargetInfoBase &targetInfo);
} // namespace fma_details

namespace {
struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  //  using ConvertTritonGPUOpToLLVMPattern<
  //      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

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
      return fma_details::convertDPAS(op, adaptor, getTypeConverter(), rewriter,
                                      targetInfo);
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return fma_details::convertFMADot(op, adaptor, getTypeConverter(),
                                        rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

} // namespace

void mlir::triton::intel::populateDotOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    const TargetInfoBase &targetInfo, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, targetInfo, benefit);
}
