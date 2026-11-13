//===- Utility.h - Code generation utilities ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::LLVM::intel {

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i);

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value selector);

/// Create a predicated block, using \p cond as the condition and \p ops for the
/// values supplied by the conditional branch to the exit block. The \p
/// thenOpsFn function is used to inject operations in the 'then' branch:
///   cf.cond_br %cond, ^br1, ^br2(%ops)
///   ^br1:
///     %then_ops = `thenOpsFn()`
///     cf.br ^br2(%then_ops)
///   ^br2(%block_ops):
template <typename ThenOpsFn>
Block &createPredicatedBlock(RewriterBase &rewriter, Location loc, Value cond,
                             ArrayRef<Value> ops, ThenOpsFn &&thenOpsFn) {
  Block *insertionBlock = rewriter.getInsertionBlock();
  Block *thenBlock =
      rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
  Block *endBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());

  rewriter.setInsertionPointToEnd(insertionBlock);
  cf::CondBranchOp::create(rewriter, loc, cond, thenBlock, endBlock, ops);

  rewriter.setInsertionPointToStart(thenBlock);
  auto thenOps = thenOpsFn();
  assert(thenOps.size() == ops.size() && "Inconsistent size");
  assert(llvm::all_of(llvm::enumerate(ops, thenOps),
                      [](const auto &enumerator) {
                        auto [index, op, thenOp] = enumerator;
                        return op.getType() == thenOp.getType();
                      }) &&
         "type mismatch found");

  if (thenOps.empty())
    cf::BranchOp::create(rewriter, loc, endBlock);
  else
    cf::BranchOp::create(rewriter, loc, endBlock, thenOps);

  for (Value op : thenOps)
    endBlock->addArgument(op.getType(), op.getLoc());

  rewriter.setInsertionPointToStart(endBlock);
  return *endBlock;
}

/// Create a predicated block, using \p cond as the condition and \p thenOpsFn
/// to inject operations in the 'then' branch:
///   cf.cond_br %cond, ^br1, ^br2
///   ^br1:
///     `thenOpsFn()`
///     cf.br ^br2
///   ^br2:
template <typename ThenOpsFn>
Block &createPredicatedBlock(RewriterBase &rewriter, Location loc, Value cond,
                             ThenOpsFn &&thenOpsFn) {
  return createPredicatedBlock(rewriter, loc, cond, {}, thenOpsFn);
}

/// Create a two-armed predicated block, using \p cond to select between two
/// side-effecting, result-less bodies:
///   cf.cond_br %cond, ^fast, ^slow
///   ^fast:
///     `fastFn()`
///     cf.br ^end
///   ^slow:
///     `slowFn()`
///     cf.br ^end
///   ^end:
template <typename FastFn, typename SlowFn>
void createTwoArmedPredicatedBlock(RewriterBase &rewriter, Location loc,
                                   Value cond, FastFn &&fastFn,
                                   SlowFn &&slowFn) {
  Block *insertionBlock = rewriter.getInsertionBlock();
  Block *fastBlock =
      rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
  Block *slowBlock = rewriter.splitBlock(fastBlock, fastBlock->begin());
  Block *endBlock = rewriter.splitBlock(slowBlock, slowBlock->begin());

  rewriter.setInsertionPointToEnd(insertionBlock);
  cf::CondBranchOp::create(rewriter, loc, cond, fastBlock, slowBlock);

  rewriter.setInsertionPointToStart(fastBlock);
  (void)fastFn();
  cf::BranchOp::create(rewriter, loc, endBlock);

  rewriter.setInsertionPointToStart(slowBlock);
  (void)slowFn();
  cf::BranchOp::create(rewriter, loc, endBlock);

  rewriter.setInsertionPointToStart(endBlock);
}

LLVM::RoundingMode
convertTritonRoundingModeToLLVM(const triton::RoundingMode rounding);

Type getTypeWithSameShape(Type type, Type elementType);

bool hasModuleAttr(Operation *op, StringRef attrName);

// Pack the values in vectors, call func on each vector and unpack the returned
// vectors. The vector sizes are powers of 2 <= maxVecSize. If the input
// contains a single value, it's not packed. If func returns a null value, empty
// vector is returned.
SmallVector<Value>
vectorize(std::function<Value(TritonLLVMIRRewriter &, Value)> func,
          Location loc, ConversionPatternRewriter &rewriter,
          const SmallVector<Value> &values, size_t maxVecSize = 16);
} // namespace mlir::LLVM::intel

namespace mlir::triton::intel {

Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        Value v);
Value convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        Value v, RoundingMode rounding);

Value convertWithFunctionCall(TritonLLVMIRRewriter &rewriter, Value value,
                              StringRef baseName, Type inType, Type outType,
                              StringRef hasAttrName = {});

/// The fp8 -> fp16 lowering of a `tt.fp_to_fp`, for any subset of its
/// elements, with the instruction sequence of the `tt.fp_to_fp` lowering. Each
/// element converts independently, so the results are those of that lowering
/// for any `indices`; runs of consecutive elements aligned to the converter's
/// group size also keep its instruction grouping.
class Fp8ToFp16Lowering {
public:
  /// std::nullopt unless `op` is an fp8 -> fp16 conversion without a rounding
  /// mode. `llSrc` is the lowered source; it is unpacked once here.
  static std::optional<Fp8ToFp16Lowering>
  get(FpToFpOp op, Value llSrc, ConversionPatternRewriter &rewriter);

  /// Whether the bytes are read from the dwords of a 32-bit 2D block load.
  bool readsDwords() const { return !bytes.empty(); }
  size_t size() const { return readsDwords() ? bytes.size() : elems.size(); }
  SmallVector<Value> convert(ArrayRef<unsigned> indices,
                             ConversionPatternRewriter &rewriter) const;

private:
  FpToFpOp op;
  SmallVector<std::pair<Value, int64_t>> bytes;
  SmallVector<Value> elems;
};

/// Whether the DPAS lowering of the dot that uses `op` converts `op`'s fp8
/// operand to fp16 itself, one DPAS operand at a time; the `tt.fp_to_fp`
/// lowering then emits nothing for `op`.
bool isDPASOperandConvertedByDot(FpToFpOp op);

/// What the `tt.fp_to_fp` lowering leaves in place of a conversion that the
/// DPAS lowering does (see isDPASOperandConvertedByDot): a cast of the lowered
/// fp8 source `llSrc` to `type`, tagged so that getLazyFp8Cast finds it.
Value createLazyFp8Operand(OpBuilder &b, Location loc, Type type, Value llSrc);

/// The cast created by createLazyFp8Operand that defines `lowered`, if any.
UnrealizedConversionCastOp getLazyFp8Cast(Value lowered);

} // namespace mlir::triton::intel

#endif // TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H
