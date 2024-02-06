#include "../Utility.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

using ValueTableFMA = std::map<std::pair<int, int>, Value>;

static ValueTableFMA
getValueTableFromStructFMA(Value val, int K, int n0, int shapePerCTATile,
                           int sizePerThread,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const LLVMTypeConverter *typeConverter, Type type) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTATile)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
  }
  return res;
}

static Value convertIfRequired(Value val, Type tgtTy, Location loc,
                               ConversionPatternRewriter &rewriter) {
  Type valTy = val.getType();
  if (valTy == tgtTy)
    return val;

  assert(tgtTy.isIntOrFloat() && valTy.isIntOrFloat() &&
         "Unexpected tgtTy or valTy types");

  auto convertToFloat = [&](Type valTy, FloatType tgtTy) -> Value {
    unsigned tgtBitWidth = tgtTy.getIntOrFloatBitWidth(),
             valBitWidth = valTy.getIntOrFloatBitWidth();

    return llvm::TypeSwitch<Type, Value>(valTy)
        .Case<FloatType>([&](FloatType ty) {
          Operation *castOp =
              (valBitWidth <= tgtBitWidth)
                  ? rewriter.create<LLVM::FPExtOp>(loc, tgtTy, val)
                  : rewriter.create<LLVM::FPTruncOp>(loc, tgtTy, val);
          return castOp->getResult(0);
        })
        .Case<IntegerType>([&](IntegerType ty) {
          Operation *castOp =
              (ty.isSigned() || ty.isSignless())
                  ? rewriter.create<LLVM::SIToFPOp>(loc, tgtTy, val)
                  : rewriter.create<LLVM::UIToFPOp>(loc, tgtTy, val);
          return castOp->getResult(0);
        });
  };

  auto convertToInteger = [&](Type valTy, IntegerType tgtTy) -> Value {
    unsigned tgtBitWidth = tgtTy.getIntOrFloatBitWidth(),
             valBitWidth = valTy.getIntOrFloatBitWidth();

    return llvm::TypeSwitch<Type, Value>(valTy)
        .Case<FloatType>([&](FloatType ty) {
          Operation *castOp =
              (tgtTy.isSigned() || tgtTy.isSignless())
                  ? rewriter.create<LLVM::FPToSIOp>(loc, tgtTy, val)
                  : rewriter.create<LLVM::FPToUIOp>(loc, tgtTy, val);
          return castOp->getResult(0);
        })
        .Case<IntegerType>([&](IntegerType ty) {
          Operation *castOp =
              (valBitWidth <= tgtBitWidth)
                  ? rewriter.create<LLVM::SExtOp>(loc, tgtTy, val)
                  : rewriter.create<LLVM::TruncOp>(loc, tgtTy, val);
          return castOp->getResult(0);
        });
  };

  return llvm::TypeSwitch<Type, Value>(tgtTy)
      .Case<FloatType>([&](auto ty) { return convertToFloat(valTy, ty); })
      .Case<IntegerType>([&](auto ty) { return convertToInteger(valTy, ty); });
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = A.getType().cast<RankedTensorType>();
  auto bTensorTy = B.getType().cast<RankedTensorType>();
  auto dTensorTy = D.getType().cast<RankedTensorType>();

  auto aShapePerCTA = getShapePerCTA(aTensorTy);
  auto bShapePerCTA = getShapePerCTA(bTensorTy);

  BlockedEncodingAttr dLayout =
      dTensorTy.getEncoding().cast<BlockedEncodingAttr>();
  auto order = dLayout.getOrder();
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTATile = getShapePerCTATile(dLayout);

  int K = aShapePerCTA[1];
  int M = aShapePerCTA[0];
  int N = bShapePerCTA[1];

  int mShapePerCTATile =
      order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int mSizePerThread =
      order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  int nShapePerCTATile =
      order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int nSizePerThread =
      order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];

  auto has =
      getValueTableFromStructFMA(llA, K, M, mShapePerCTATile, mSizePerThread,
                                 rewriter, loc, typeConverter, aTensorTy);
  auto hbs =
      getValueTableFromStructFMA(llB, K, N, nShapePerCTATile, nSizePerThread,
                                 rewriter, loc, typeConverter, bTensorTy);

  SmallVector<Value> ret = cc;
  bool isCRow = order[0] == 1;

  for (unsigned k = 0; k < K; k++) {
    for (unsigned m = 0; m < M; m += mShapePerCTATile)
      for (unsigned n = 0; n < N; n += nShapePerCTATile)
        for (unsigned mm = 0; mm < mSizePerThread; ++mm)
          for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
            int mIdx = m / mShapePerCTATile * mSizePerThread + mm;
            int nIdx = n / nShapePerCTATile * nSizePerThread + nn;

            int z = isCRow
                        ? mIdx * N / nShapePerCTATile * mSizePerThread + nIdx
                        : nIdx * M / mShapePerCTATile * nSizePerThread + mIdx;
            Type tgtTy = ret[z].getType();
            Value opA =
                convertIfRequired(has[{m + mm, k}], tgtTy, loc, rewriter);
            Value opB =
                convertIfRequired(hbs[{n + nn, k}], tgtTy, loc, rewriter);

            llvm::TypeSwitch<Type>(tgtTy)
                .Case<FloatType>([&](auto) {
                  ret[z] =
                      rewriter.create<LLVM::FMulAddOp>(loc, opA, opB, ret[z]);
                })
                .Case<IntegerType>([&](auto) {
                  ret[z] = rewriter.create<LLVM::AddOp>(
                      loc, rewriter.create<LLVM::MulOp>(loc, opA, opB), ret[z]);
                });
          }
  }

  auto res = packLLElements(loc, typeConverter, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}
