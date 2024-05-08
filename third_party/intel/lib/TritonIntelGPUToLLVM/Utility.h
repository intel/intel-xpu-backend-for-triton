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
    return rewriter.create<LLVM::PoisonOp>(funcOp.getLoc(), ptrTy);
  return funcOp.getArgument(funcOp.getNumArguments() - 1);
}

static Value getSharedMemoryBase(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Operation *op) {
  auto ptrTy = LLVM::LLVMPointerType::get(
      rewriter.getContext(), TritonGEN::TritonGENMemorySpace::kWorkgroup);
  FunctionOpInterface func = op->getParentOfType<FunctionOpInterface>();
  assert(op->hasAttr("allocation.offset"));
  size_t offset = cast<IntegerAttr>(op->getAttr("allocation.offset"))
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

#endif
