#include "Conversion/ProtonGPUToLLVM/ProtonIntelGPUToLLVM/IntelPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonIntelGPUToLLVM/TargetInfo.h"
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENMemorySpace.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct CircularStoreOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::CircularStoreOp> {
  explicit CircularStoreOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::CircularStoreOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::CircularStoreOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dataPack =
        lowerCircularStore(op, adaptor.getSegment(), adaptor.getCounter(),
                           adaptor.getDynamicScopeId(), rewriter);
    uint32_t addrSpace = dataPack.addrSpace;
    if (addrSpace == TritonGEN::TritonGENMemorySpace::kCrossWorkgroup) {
      // `dataPack.record` is a packed <2 x i32> (tag + clock word); bitcast
      // to i64 so `storeRelaxed`'s `getIntOrFloatBitWidth()`-based alignment
      // computation accepts it (vector types aren't "int or float").
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      Value record64 = b.bitcast(dataPack.record, i64_ty);
      targetInfo.getTritonTargetInfo().storeRelaxed(rewriter, loc, dataPack.ptr,
                                                    record64, dataPack.isWriter,
                                                    MemSyncScope::GPU);
    } else if (addrSpace == TritonGEN::TritonGENMemorySpace::kWorkgroup) {
      targetInfo.getTritonTargetInfo().storeDShared(rewriter, loc, dataPack.ptr,
                                                    Value(), dataPack.record,
                                                    dataPack.isWriter);
    } else {
      llvm::report_fatal_error("unsupported address space in circular store");
    }
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::proton::gpu::Intel {
void populateProtonGPUOpIntelPatterns(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      const TargetInfo &targetInfo,
                                      PatternBenefit benefit) {
  patterns.add<CircularStoreOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::proton::gpu::Intel
