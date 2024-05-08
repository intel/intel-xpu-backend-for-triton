//===- Utility.h - Code generation utilities ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;

namespace mlir::LLVM::intel {

/// Create a predicated block, using \p cond as the condition and \p ops for the
/// values supplied by the conditional branch to the exit block. The \p
/// thenOpsFn function is used to inject operations in the 'then' branch:
///   cf.cond_br %cond, ^br1, ^br2(%ops)
///   ^br1:
///     %then_ops = `thenOpsFn()`
///     cf.br ^br2(%then_ops)
///   ^br2(%block_ops):
template <typename ThenOpsFn>
Block &createPredicatedBlock(ConversionPatternRewriter &rewriter, Location loc,
                             Value cond, ArrayRef<Value> ops,
                             ThenOpsFn &&thenOpsFn) {
  Block *insertionBlock = rewriter.getInsertionBlock();
  Block *thenBlock =
      rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
  Block *endBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());

  rewriter.setInsertionPointToEnd(insertionBlock);
  rewriter.create<cf::CondBranchOp>(loc, cond, thenBlock, endBlock, ops);

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
    rewriter.create<cf::BranchOp>(loc, endBlock);
  else
    rewriter.create<cf::BranchOp>(loc, endBlock, thenOps);

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
Block &createPredicatedBlock(ConversionPatternRewriter &rewriter, Location loc,
                             Value cond, ThenOpsFn &&thenOpsFn) {
  return createPredicatedBlock(rewriter, loc, cond, {}, thenOpsFn);
}

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred);

Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i);
Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter, Value val,
                int i);
Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i);
Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 Value i);

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content,
                        unsigned addressSpace);

LLVM::LLVMFuncOp getSpirvPrintfDeclaration(ConversionPatternRewriter &rewriter);

static Value getStackPointer(PatternRewriter &rewriter,
                             FunctionOpInterface funcOp) {
  auto mod = funcOp->getParentOfType<ModuleOp>();
  LLVM::LLVMPointerType ptrTy = ptr_ty(
      rewriter.getContext(), TritonGEN::TritonGENMemorySpace::kWorkgroup);
  if (mod->getAttrOfType<IntegerAttr>("triton_gpu.shared").getInt() == 0)
    return rewriter.create<LLVM::UndefOp>(funcOp.getLoc(), ptrTy);
  return funcOp.getArgument(funcOp.getNumArguments() - 1);
}

static Value getSharedMemoryBase(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Operation *op) {
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
  FunctionOpInterface func =
      op->template getParentOfType<FunctionOpInterface>();
  assert(op->hasAttr("allocation.offset"));
  size_t offset = op->getAttr("allocation.offset")
                      .cast<IntegerAttr>()
                      .getValue()
                      .getZExtValue();
  Value offVal = i32_val(offset);
  Value base =
      gep(ptrTy, i8_ty, LLVM::intel::getStackPointer(rewriter, func), offVal);
  return base;
}

// Returns a Value for the format string, which you can reuse.
Value llPrintf(ConversionPatternRewriter &rewriter, StringRef msg,
               ValueRange args);

void llPrintf(ConversionPatternRewriter &rewriter, Value msg, ValueRange args);

static Value getModuleWarpSize(RewriterBase &rewriter, Location loc) {
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  return i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
}

} // namespace mlir::LLVM::intel

using ::mlir::triton::gpu::intel::DpasEncodingAttr;

static void
emitOffsetForDpasLayoutPerCTA(const DpasEncodingAttr &dpasLayout,
                              SmallVector<SmallVector<unsigned>> &offsets,
                              unsigned ctaOffsetX, unsigned ctaOffsetY) {
  SmallVector<unsigned> sizePerThreads = getSizePerThread(dpasLayout);
  uint32_t elemsPerThreadPerGroup = product<unsigned>(sizePerThreads);
  uint32_t rowsPerWarp =
      dpasLayout.getSubGroupSize() / dpasLayout.getExecutionSize();
  SmallVector<unsigned> shapePerCTA =
      triton::gpu::getShapePerCTATile(dpasLayout);

  for (unsigned elem = 0; elem < elemsPerThreadPerGroup; elem++) {
    uint32_t elemRowIndex = (elem / sizePerThreads[1]) * rowsPerWarp;
    uint32_t elemColIndex = elem % sizePerThreads[1];
    offsets.push_back({ctaOffsetX + elemRowIndex, ctaOffsetY + elemColIndex});
  }
}

static SmallVector<SmallVector<unsigned>>
emitOffsetForDpasLayout(const DpasEncodingAttr &dpasLayout,
                        RankedTensorType type) {
  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<SmallVector<unsigned>> offsets;
  SmallVector<unsigned> shapePerCTA = getShapePerCTATile(dpasLayout);

  for (unsigned i = 0; i < shape[0]; i += shapePerCTA[0]) {
    for (unsigned j = 0; j < shape[1]; j += shapePerCTA[1]) {
      emitOffsetForDpasLayoutPerCTA(dpasLayout, offsets, i, j);
    }
  }

  return offsets;
}

// -----------------------------------------------------------------------
// Dpas layout indices
// -----------------------------------------------------------------------

static SmallVector<Value>
emitBaseIndexForDpasLayout(Location loc, RewriterBase &rewriter,
                           const DpasEncodingAttr &dpasLayout,
                           RankedTensorType type) {
  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(triton::gpu::getWarpSize(dpasLayout));
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);

  auto warpsPerCTA = dpasLayout.getWarpsPerCTA();
  ArrayRef<int64_t> shape = type.getShape();

  auto order = triton::gpu::getOrder(dpasLayout);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  // Compute the 2-dim coordinates of the warp containing the tensor element
  // operated on by this thread.
  SmallVector<unsigned> warpShape = dpasLayout.getShapeC();
  Value rowWarpId = urem(multiDimWarpId[0],
                         i32_val(mlir::ceil<unsigned>(shape[0], warpShape[0])));
  Value colWarpId = urem(multiDimWarpId[1],
                         i32_val(mlir::ceil<unsigned>(shape[1], warpShape[1])));
  Value rowWarpOffset = mul(rowWarpId, i32_val(warpShape[0]));
  Value colWarpOffset = mul(colWarpId, i32_val(warpShape[1]));

  // Compute the 2-dim coordinates of the first element in the warp operated
  // on by this thread.
  SmallVector<unsigned> threadsPerWarp = getThreadsPerWarp(dpasLayout);
  SmallVector<Value> multiDimBase = {
      add(udiv(laneId, i32_val(threadsPerWarp[1])), rowWarpOffset),
      add(urem(laneId, i32_val(threadsPerWarp[1])), colWarpOffset)};
  return multiDimBase;
}

#endif
