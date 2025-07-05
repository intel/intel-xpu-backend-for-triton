#include "TritonIntelGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    auto K = a.size();
    assert(b.size() == K);
    Value accum = c;
    Type tgtTy = accum.getType();
    for (auto it = llvm::zip(a, b).begin(); it != llvm::zip(a, b).end(); ++it) {
      const auto &aElem = std::get<0>(*it);
      const auto &bElem = std::get<1>(*it);

      assert(aElem.getType() == tgtTy);
      assert(bElem.getType() == tgtTy);

      llvm::TypeSwitch<Type>(tgtTy)
          .Case<FloatType>([&](auto) {
            accum = builder.create<LLVM::FMulAddOp>(loc, aElem, bElem, accum);
          })
          .Case<IntegerType>([&](auto) {
            accum = builder.create<LLVM::AddOp>(
                loc, builder.create<LLVM::MulOp>(loc, aElem, bElem), accum);
          });
    }
    return accum;
  }
};

} // namespace

namespace fma_details {

LogicalResult
convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
              const TritonIntelGPUToLLVMTypeConverter *typeConverter,
              ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();
  GenericFMAVectorMultiplier multiplier(rewriter, loc);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}

} // namespace fma_details
