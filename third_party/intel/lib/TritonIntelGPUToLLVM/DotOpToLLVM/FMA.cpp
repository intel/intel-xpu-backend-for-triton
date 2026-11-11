//===- FMA.cpp - FMA dot lowering for Intel GPUs --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils/LLVMIntr.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

/// Number of 8-bit components consumed by one dp4a instruction.
constexpr unsigned NumDp4aComponents = 4;

/// Multiplier emitting `dp4a` for dot products over extended 8-bit integers.
///
/// Upstream's generic multiplier emits a `mul`/`add` chain, which IGC folds
/// back into a `dp4a` in `CustomSafeOptPass::matchDp4a()`. That pattern match
/// reorders the two operands of the folded instruction independently, so it can
/// pair `a[i]` with `b[j]`, `i != j`, and compute a different dot product from
/// the one that was written (issue #7854). Emitting the instruction directly is
/// correct by construction, and is better code than the chain regardless: one
/// instruction per four elements of K instead of four multiplies and four
/// additions.
///
/// The operands reach this point extended to the accumulator type (by
/// `decomposeMixedModeDotOp`, or by the kernel itself); \p aIsSigned and
/// \p bIsSigned record how, which is what makes truncating them back to i8 for
/// dp4a value-preserving.
class Dp4aFMAVectorMultiplier : public FMAVectorMultiplier {
  ConversionPatternRewriter &rewriter;
  Location loc;
  bool aIsSigned;
  bool bIsSigned;

  /// \returns the four \p vals, truncated to their original 8 bits and packed
  /// into a single i32, which is how dp4a takes its operands.
  Value packComponents(ArrayRef<Value> vals) {
    assert(vals.size() == NumDp4aComponents && "Unexpected operand count");
    TritonLLVMOpBuilder builder(loc, rewriter);
    Type i8Ty = rewriter.getI8Type();
    auto vecTy = VectorType::get(NumDp4aComponents, i8Ty);
    Value vec = builder.undef(vecTy);
    for (auto [idx, val] : llvm::enumerate(vals))
      vec = builder.insert_element(vecTy, vec, builder.trunc(i8Ty, val),
                                   builder.i32_val(idx));
    return builder.bitcast(vec, rewriter.getI32Type());
  }

  /// Emits `acc + a·b` as a single dp4a instruction.
  Value createDp4a(ArrayRef<Value> a, ArrayRef<Value> b, Value acc) {
    TritonLLVMOpBuilder builder(loc, rewriter);
    Type i32Ty = rewriter.getI32Type();
    Type i1Ty = rewriter.getI1Type();
    // The name encodes the signedness of the two operands, in that order.
    std::string funcName = "llvm.genx.GenISA.dp4a.";
    funcName += aIsSigned ? "s" : "u";
    funcName += bIsSigned ? "s" : "u";
    funcName += ".i32";

    SmallVector<Type> argTypes{i32Ty, i32Ty, i32Ty, i1Ty};
    SmallVector<Value> args{acc, packComponents(a), packComponents(b),
                            builder.i1_val(false) /*saturate*/};
    return intel::createDeviceFunctionCall(rewriter, funcName, i32Ty, argTypes,
                                           args, /*paramAttrs=*/{},
                                           intel::noUnwindWillReturnAttrs)
        .getResult();
  }

public:
  Dp4aFMAVectorMultiplier(ConversionPatternRewriter &rewriter, Location loc,
                          bool aIsSigned, bool bIsSigned)
      : rewriter(rewriter), loc(loc), aIsSigned(aIsSigned),
        bIsSigned(bIsSigned) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    unsigned K = a.size();
    assert(b.size() == K && "Operands must have the same size");
    TritonLLVMOpBuilder builder(loc, rewriter);

    Value accum = c;
    unsigned k = 0;
    for (; k + NumDp4aComponents <= K; k += NumDp4aComponents)
      accum = createDp4a(a.slice(k, NumDp4aComponents),
                         b.slice(k, NumDp4aComponents), accum);

    // Trailing elements when K is not a multiple of four. Fewer than four
    // products cannot be folded into a dp4a, so this tail is not exposed to the
    // mis-pairing described above.
    for (; k < K; ++k)
      accum = builder.add(builder.mul(a[k], b[k]), accum);
    return accum;
  }
};

/// \returns whether \p operand is an 8-bit integer tensor extended to a wider
/// type, and whether that extension is signed. Layout conversions are looked
/// through: they do not change the value.
std::optional<bool> getInt8ExtensionKind(Value operand) {
  while (auto cvt = operand.getDefiningOp<ConvertLayoutOp>())
    operand = cvt.getSrc();

  Operation *def = operand.getDefiningOp();
  bool isSigned;
  if (isa_and_nonnull<arith::ExtSIOp>(def))
    isSigned = true;
  else if (isa_and_nonnull<arith::ExtUIOp>(def))
    isSigned = false;
  else
    return std::nullopt;

  auto srcTy = dyn_cast<RankedTensorType>(def->getOperand(0).getType());
  if (!srcTy || !srcTy.getElementType().isInteger(8))
    return std::nullopt;
  return isSigned;
}

} // namespace

namespace fma_details {

LogicalResult convertIntegerFMADot(DotOp op, DotOp::Adaptor adaptor,
                                   const LLVMTypeConverter *typeConverter,
                                   ConversionPatternRewriter &rewriter) {
  // dp4a accumulates into i32 and multiplies extended 8-bit values; any other
  // integer dot keeps the generic mul/add chain, which IGC cannot fold into a
  // dp4a and therefore cannot mis-pair.
  if (!cast<RankedTensorType>(op.getType()).getElementType().isInteger(32))
    return convertFMADot(op, adaptor, typeConverter, rewriter);

  std::optional<bool> aIsSigned = getInt8ExtensionKind(op.getA());
  std::optional<bool> bIsSigned = getInt8ExtensionKind(op.getB());
  if (!aIsSigned || !bIsSigned)
    return convertFMADot(op, adaptor, typeConverter, rewriter);

  Dp4aFMAVectorMultiplier multiplier(rewriter, op.getLoc(), *aIsSigned,
                                     *bIsSigned);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}

} // namespace fma_details
