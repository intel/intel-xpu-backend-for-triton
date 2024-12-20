#include "PatternTritonGPUOpToLLVM.h"
#include <iostream>

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
using namespace mlir::triton::gpu;

namespace {

static Value mxfpScaleBf16(ConversionPatternRewriter &rewriter, Location loc,
                           Value v, Value scale) {
  Value vBf16 = bitcast(v, bf16_ty);
  Value nanBf16 = bitcast(i16_val(0x7fff), bf16_ty);
  Value scaleIsNan = icmp_eq(scale, i8_val(0xff));
  Value scaleBf16 = bitcast(shl(zext(i16_ty, scale), i16_val(7)), bf16_ty);

  Value v0 = mlir::triton::intel::convertBf16ToFp32(loc, rewriter, vBf16);
  Value v1 = mlir::triton::intel::convertBf16ToFp32(loc, rewriter, scaleBf16);
  auto result = rewriter.create<LLVM::FMulOp>(loc, f32_ty, v0, v1);
  auto undefRounding = static_cast<mlir::triton::RoundingMode>(-1);
  Value scaledBf16 = mlir::triton::intel::convertFp32ToBf16(
      loc, rewriter, result, undefRounding);
  // Value scaledBf16 = fmul(vBf16, scaleBf16);
  // Account for NaN in the scale as per the mxfp specification.
  return select(scaleIsNan, nanBf16, scaledBf16);
};

class UpcastMXFPOpPattern : public ConvertOpToLLVMPattern<UpcastMXFPOp> {
private:
  const TargetInfoBase &targetInfo;
  const bool upcastMXFPUseDotOpEnc =
      mlir::triton::tools::getBoolEnv("TRITON_INTEL_UPCASTMXFP_DOTOP_ENCODING");

public:
  UpcastMXFPOpPattern(LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<UpcastMXFPOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(UpcastMXFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto operands = adaptor.getOperands();
    SmallVector<Value> xVals = unpackLLElements(loc, operands[0], rewriter);
    SmallVector<Value> scaleVals = unpackLLElements(loc, operands[1], rewriter);
    ScaleDotElemType fpType = op.getFpType();

    Value tid = tid_val();
    auto mod = op->getParentOfType<ModuleOp>();
    Value warpSize =
        i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpId = udiv(tid, warpSize);
    Value laneId = urem(tid, warpSize);

    // TODO: check if using the correct Dot/DPAS term mapping
    auto xType = cast<RankedTensorType>(op->getOperandTypes()[0]);
    unsigned kWidth =
        cast<DotOperandEncodingAttr>(xType.getEncoding()).getKWidth();

    std::cout << "xVals size before fp4 -> fp8: " << xVals.size() << std::endl;
    if (fpType == ScaleDotElemType::E2M1)
      xVals = LLVM::convertMxfp4x2ToBf16x2(rewriter, loc, xVals);
    std::cout << "xVals size after fp4 -> fp8: " << xVals.size() << std::endl;
    std::cout << "scaledVals size: " << scaleVals.size() << std::endl;

    // TODO: need to refactor the logic, the concepts of DPAS do not match loops
    // correctly
    if (upcastMXFPUseDotOpEnc) {
      // FIXME: Doc is not completedly correct with Intel DPAS layout
      // For Intel GPU PVC, each thread owns elements of 16 mxfp vectors so we
      // need 16 scales Since we go from a threadShape of 2x16 to 32x1, we let
      // c = tid / 16.
      Value c = udiv(laneId, i32_val(16));
      unsigned repSize = 2;
      unsigned subTileSize = 8;
      unsigned kRepSize = kWidth / 2;
      unsigned scalingBlockSize = 32;
      unsigned mxfpSize = 16;

      std::cout << "ci: ";
      SmallVector<Value, 16> ci;
      for (int row = 0; row < repSize; ++row)
        for (int col = 0; col < subTileSize; ++col) {
          ci.emplace_back(i32_val(row + 2 * col));
          std::cout << " " << row + 2 * col;
        }
      std::cout << std::endl;

      if (fpType == ScaleDotElemType::E2M1) {
        repSize /= 2;
        kRepSize *= 2;
      }
      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        for (int mxfp = 0; mxfp < 2; ++mxfp) {
          SmallVector<Value, 8> si;
          std::cout << "si: ";
          for (int subTile = 0; subTile < 8; ++subTile) {
            si.emplace_back(targetInfo.shuffleIdx(rewriter, loc, scaleVal,
                                                  ci[8 * mxfp + subTile]));
            std::cout << " " << 8 * mxfp + subTile;
          }
          std::cout << "\nIdx: ";
          for (int rep = 0; rep < repSize; ++rep)
            for (int subTile = 0; subTile < subTileSize; ++subTile) {
              std::cout << " Subtile(" << subTile << ") ";
              for (int k = 0; k < kRepSize; ++k) {
                auto idx = i * scalingBlockSize + mxfp * mxfpSize +
                           subTileSize * rep + subTile * kRepSize + k;
                std::cout << " " << idx;
                xVals[idx] =
                    mxfpScaleBf16(rewriter, loc, xVals[idx], si[subTile]);
              }
            }
          std::cout << std::endl;
        }
      }
    } else {
      for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
        for (int j = 0; j < 32; ++j) {
          xVals[32 * i + j] =
              mxfpScaleBf16(rewriter, loc, xVals[32 * i + j], scaleVal);
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
