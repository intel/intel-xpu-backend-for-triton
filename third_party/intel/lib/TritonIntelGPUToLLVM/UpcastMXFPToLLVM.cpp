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
#include <array>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

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
    return failure();

    // TODO: Implement this
#if 0
    ScaleDotElemType fpType = op.getFpType();
    Location loc = op.getLoc();
    SmallVector<Value> xVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> scaleVals =
        unpackLLElements(loc, adaptor.getScale(), rewriter);

    auto mod = op->getParentOfType<ModuleOp>();
    llvm::errs() << "mod: " << mod << "\n";
    llvm::errs() << "adaptor.getScale(): " << adaptor.getScale() << "\n";

    LDBG("x: " << xVals.size() << " x " << xVals.front().getType());
    LDBG("scale: " << scaleVals.size() << " x " << scaleVals.front().getType());

    bool isPacked = fpType == ScaleDotElemType::E2M1;
    if (xVals.size() != scaleVals.size() * (isPacked ? 16 : 32))
      return rewriter.notifyMatchFailure(op, "unsupported problem size");

    unsigned numThreads = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value warpSize = i32_val(numThreads);
    Value tid = tid_val();
    Value warpId = udiv(tid, warpSize);
    Value laneId = urem(tid, warpSize);

    if (fpType == ScaleDotElemType::E2M1) {
      llvm::errs() << "at line: " << __LINE__ << "\n";
      xVals = LLVM::convertMxfp4x2ToBf16x2(rewriter, loc, xVals);
    }

    llvm::errs() << "xVals size: " << xVals.size() << "\n";

    // Each thread owns elements of 4 mxfp vectors so we need 4 scales
    // Letting c = tid / 4 * 2, we need the elements from threads c, c + 1, c +
    // 16, c + 17
    LLVM::MulOp c = mul(udiv(laneId, i32_val(4)), i32_val(2));
    std::array<Value, 4> ci = {c, add(c, i32_val(1)), add(c, i32_val(16)),
                               add(c, i32_val(17))};

    for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
      // column major as per the DotOperandEncoding(opidx=0) layout
      auto si = std::array<Value, 4>{
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[0]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[2]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[1]),
          targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[3]),
      };

      for (int j = 0; j < 32; ++j) {
        int index = 32 * i + j;
        llvm::errs() << "index: " << index << "\n";
        xVals[index] =
            LLVM::mxfpScaleBf16(rewriter, loc, xVals[index], si[j / 8]);
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
#endif
  }
};
} // anonymous namespace

void mlir::triton::intel::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<UpcastMXFPOpPattern>(typeConverter, targetInfo, benefit);
}
