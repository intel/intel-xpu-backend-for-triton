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
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/Support/ErrorHandling.h"

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
Block &createPredicatedBlock(RewriterBase &rewriter, Location loc, Value cond,
                             ArrayRef<Value> ops, ThenOpsFn &&thenOpsFn) {
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
Block &createPredicatedBlock(RewriterBase &rewriter, Location loc, Value cond,
                             ThenOpsFn &&thenOpsFn) {
  return createPredicatedBlock(rewriter, loc, cond, {}, thenOpsFn);
}

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred);

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i);
Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i);

Value addStringToModule(Location loc, RewriterBase &rewriter, StringRef key,
                        StringRef content, unsigned addressSpace);

LLVM::LLVMFuncOp getSpirvPrintfDeclaration(RewriterBase &rewriter);

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
                                 const TargetInfoBase &target, Operation *op) {
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          target.getSharedAddressSpace());
  FunctionOpInterface func = op->getParentOfType<FunctionOpInterface>();
  // CI debugging usage here
  if (!op->hasAttr("allocation.offset")) {
    auto mod = op->getParentOfType<ModuleOp>();
    llvm::errs() << "op: " << *op << "\n";
    llvm::errs() << "mod:" << mod << "\n";
    llvm_unreachable("missing allocation.offset");
  }
  size_t offset = cast<IntegerAttr>(op->getAttr("allocation.offset"))
                      .getValue()
                      .getZExtValue();
  Value offVal = i32_val(offset);
  Value base =
      gep(ptrTy, i8_ty, LLVM::intel::getStackPointer(rewriter, func), offVal);
  return base;
}

static Value getModuleWarpSize(RewriterBase &rewriter, Location loc) {
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  return i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
}

void printTensor(StringRef msg, Value tensor, Type tensorTy,
                 ConversionPatternRewriter &rewriter,
                 const TargetInfoBase &targetInfo);

} // namespace mlir::LLVM::intel

// -----------------------------------------------------------------------
// Shared memory utilities
// -----------------------------------------------------------------------
using ::mlir::triton::getMultiDimIndex;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

static SmallVector<Value>
emitBaseIndexForDpasLayout(Location loc, RewriterBase &rewriter,
                           const DpasEncodingAttr &dpasLayout,
                           RankedTensorType type);

static void
emitOffsetForDpasLayoutPerCTA(const DpasEncodingAttr &dpasLayout,
                              SmallVector<SmallVector<unsigned>> &offsets,
                              unsigned ctaOffsetX, unsigned ctaOffsetY) {
  // clang-format off
  // For operand C the layout is:
  //               execution size = 16
  // <------------------------------------------------------------->
  // t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15       ^
  // .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .         | repeat count = 8
  // .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .         |
  // t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15       v
  // Then sizePerThreads = [8, 1], and coordinate offset for each element per lane should be:
  // [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]
  // clang-format on
  SmallVector<unsigned> instShapeC = dpasLayout.getDPASInstShapeC();
  SmallVector<unsigned> sizePerThreads = getSizePerThread(dpasLayout);
  ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
  SmallVector<unsigned> sizePerDPASInst = {sizePerThreads[0] / repCluster[0],
                                           sizePerThreads[1] / repCluster[1]};

  unsigned rowsPerElem = dpasLayout.getSubGroupSize() / instShapeC[1];
  unsigned colsPerElem = 1;

  unsigned repNumber = product<unsigned>(repCluster);
  unsigned elemNumberPerRep = product<unsigned>(sizePerDPASInst);
  for (unsigned repId = 0; repId < repNumber; ++repId) {
    for (unsigned elemId = 0; elemId < elemNumberPerRep; ++elemId) {
      // Follows the C++ order for the dpas layout.
      SmallVector<unsigned> repOffset = {
          (repId / repCluster[1]) * instShapeC[0],
          (repId % repCluster[1]) * instShapeC[1]};

      SmallVector<unsigned> elemOffset = {
          (elemId / sizePerDPASInst[1]) * rowsPerElem,
          (elemId % sizePerDPASInst[1]) * colsPerElem};

      offsets.push_back({repOffset[0] + elemOffset[0] + ctaOffsetX,
                         repOffset[1] + elemOffset[1] + ctaOffsetY});
    }
  }
}

static SmallVector<SmallVector<unsigned>>
emitOffsetForDotOpLayout(const DotOperandEncodingAttr &dotLayout,
                         RankedTensorType type) {
  auto dpasLayout = dyn_cast<DpasEncodingAttr>(dotLayout.getParent());
  if (!dpasLayout) {
    llvm::errs() << "dotLayout: " << dotLayout << "\n";
    llvm_unreachable("unsupported parent layout in emitOffsetForDotOpLayout");
  }

  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<SmallVector<unsigned>> offsets;
  SmallVector<int64_t> shapePerCTA = triton::gpu::getShapePerCTA(type);

  unsigned opIdx = dotLayout.getOpIdx();
  SmallVector<int64_t> numReps =
      dpasLayout.getDPASRepetitions(shapePerCTA, opIdx);
  SmallVector<unsigned> warpShape =
      (opIdx == 0) ? dpasLayout.getShapeA() : dpasLayout.getShapeB();
  SmallVector<unsigned> instShape = (opIdx == 0)
                                        ? dpasLayout.getDPASInstShapeA()
                                        : dpasLayout.getDPASInstShapeB();

  unsigned warpSize = triton::gpu::getWarpSize(dpasLayout);
  unsigned numElemPerInstPerThread = product<unsigned>(instShape) / warpSize;

  unsigned executionSize = dpasLayout.getExecutionSize();
  unsigned opsPerChannel = dpasLayout.getOpsPerChannel();

  unsigned numRowsPerPackedValue = 0u, numColsPerPackedValue = 0u;
  unsigned numColsPerLaneForPackedValue = 0u, numOpsPerPackedValue = 0u;
  switch (opIdx) {
  case 0: {
    assert((opsPerChannel == 4 || opsPerChannel == 2 || opsPerChannel == 1) &&
           "invalid opsPerChannel number.");
    SmallVector<unsigned> shapeA = dpasLayout.getShapeA();
    // Unlike the operand B, to pack the value to i16 for scalar bit width <=16.
    numOpsPerPackedValue = opsPerChannel == 4 ? 2 : 1;
    unsigned packedColNum = shapeA[1] / numOpsPerPackedValue;
    // Each value name represent multiple rows if warpSize > packedColNum
    numRowsPerPackedValue = mlir::ceil(warpSize, packedColNum);
    numColsPerPackedValue = std::min(warpSize, packedColNum);
    numColsPerLaneForPackedValue = mlir::ceil(packedColNum, warpSize);
  } break;
  case 1: {
    numOpsPerPackedValue = opsPerChannel;
    // Each value name represent multiple rows if warpSize > executionSize
    numRowsPerPackedValue = mlir::ceil(warpSize, executionSize) * opsPerChannel;
    numColsPerPackedValue = std::min(warpSize, executionSize);
    numColsPerLaneForPackedValue = mlir::ceil(executionSize, warpSize);
  } break;
  default:
    llvm_unreachable("unexpected operand index");
  }
  assert(numOpsPerPackedValue != 0 &&
         "numElemPerInstPerRowPerThread should not be zero");

  SmallVector<unsigned> shapePerCTATile = getShapePerCTATile(dotLayout);
  int64_t numRepOuter = numReps[opIdx];
  int64_t numRepK = numReps[(opIdx == 0) ? 1 : 0];

  ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
  unsigned repClusterSize = repCluster[opIdx];

  for (unsigned dimOuter = 0; dimOuter < numRepOuter; ++dimOuter)
    for (unsigned k = 0; k < numRepK; ++k)
      for (unsigned rep = 0; rep < repClusterSize; ++rep) {
        for (unsigned elemId = 0; elemId < numElemPerInstPerThread; ++elemId) {
          unsigned opsRowIndex =
              (opIdx == 0) ? 0 : elemId % numOpsPerPackedValue;
          unsigned opsColIndex =
              (opIdx == 0) ? elemId % numOpsPerPackedValue : 0;
          unsigned packedElemId = elemId / numOpsPerPackedValue;
          unsigned repRowIndex =
              shapePerCTATile[0] * (opIdx == 0 ? dimOuter : k);
          unsigned repColIndex =
              shapePerCTATile[1] * (opIdx == 0 ? k : dimOuter);
          unsigned repClusterRowIndex = opIdx == 0 ? rep * instShape[0] : 0;
          unsigned repClusterColIndex = opIdx == 0 ? 0 : rep * instShape[1];
          unsigned packedElemRowIndex =
              (packedElemId / numColsPerLaneForPackedValue) *
              numRowsPerPackedValue;
          unsigned packedElemColIndex =
              (packedElemId % numColsPerLaneForPackedValue) *
              numColsPerPackedValue;
          offsets.push_back({repRowIndex + repClusterRowIndex +
                                 packedElemRowIndex + opsRowIndex,
                             repColIndex + repClusterColIndex +
                                 packedElemColIndex + opsColIndex});
        }
      }

  return offsets;
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
emitBaseIndexForDotOpLayout(Location loc, RewriterBase &rewriter,
                            const DotOperandEncodingAttr &dotLayout,
                            RankedTensorType type) {
  auto dpasLayout = dyn_cast<DpasEncodingAttr>(dotLayout.getParent());
  if (!dpasLayout) {
    llvm::errs() << "dotLayout: " << dotLayout << "\n";
    llvm_unreachable(
        "unsupported parent layout in emitBaseIndexForDotOpLayout");
  }

  Value threadId = getThreadId(rewriter, loc);
  unsigned warpSize = triton::gpu::getWarpSize(dpasLayout);
  Value warpId = udiv(threadId, i32_val(warpSize));
  Value laneId = urem(threadId, i32_val(warpSize));

  const SmallVector<unsigned> warpsPerCTA = dpasLayout.getWarpsPerCTA();
  SmallVector<unsigned> order = triton::gpu::getOrder(dpasLayout);
  SmallVector<int64_t> shapePerCTA = triton::gpu::getShapePerCTA(type);

  unsigned opIdx = dotLayout.getOpIdx();
  SmallVector<unsigned> warpShape =
      (opIdx == 0) ? dpasLayout.getShapeA() : dpasLayout.getShapeB();
  SmallVector<int64_t> numReps =
      dpasLayout.getDPASRepetitions(shapePerCTA, opIdx);
  SmallVector<Value> multiDimWarpId =
      mlir::LLVM::delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  Value warpIndex =
      (opIdx == 0)
          ? urem(multiDimWarpId[0],
                 i32_val(mlir::ceil<unsigned>(shapePerCTA[0], warpShape[0])))
          : urem(multiDimWarpId[1],
                 i32_val(mlir::ceil<unsigned>(shapePerCTA[1], warpShape[1])));
  Value warpOffset = mul(warpIndex, i32_val(warpShape[opIdx]));

  // Compute the 2-dim coordinates of the first element in the warp operated
  // own by this thread.
  unsigned executionSize = dpasLayout.getExecutionSize();
  unsigned opsPerChannel = dpasLayout.getOpsPerChannel();

  Value laneRowIndex, laneColIndex;
  switch (opIdx) {
  case 0: {
    assert((opsPerChannel == 1 || opsPerChannel == 2 || opsPerChannel == 4) &&
           "invalid opsPerChannel number.");
    SmallVector<unsigned> shapeA = dpasLayout.getShapeA();
    // Unlike the operand B, to pack the value to i16 for scalar bit width
    // <=16.
    unsigned packedOpsPerLane = opsPerChannel == 4 ? 2 : 1;
    unsigned packedColNum = shapeA[1] / packedOpsPerLane;
    if (warpSize < packedColNum)
      llvm::report_fatal_error(
          "DpasEncodingAttr sub-group size could not "
          "be smaller than the threads required per row for A operand.");

    laneRowIndex = udiv(laneId, i32_val(packedColNum));
    laneColIndex = urem(laneId, i32_val(packedColNum));
    laneColIndex = mul(laneColIndex, i32_val(packedOpsPerLane));
  } break;
  case 1: {
    if (warpSize < executionSize)
      llvm::report_fatal_error(
          "DpasEncodingAttr sub-group size could not "
          "be smaller than the execution size for B operand.");

    laneRowIndex = udiv(laneId, i32_val(executionSize));
    laneRowIndex = mul(laneRowIndex, i32_val(opsPerChannel));
    laneColIndex = urem(laneId, i32_val(executionSize));
  } break;
  }

  auto multiDimBase =
      (opIdx == 0)
          ? SmallVector<Value>{add(laneRowIndex, warpOffset), laneColIndex}
          : SmallVector<Value>{laneRowIndex, add(laneColIndex, warpOffset)};

  return multiDimBase;
}

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

namespace mlir::triton::intel {

inline SmallVector<SmallVector<unsigned>>
emitOffsetForLayout(Attribute layout, RankedTensorType type);

inline SmallVector<SmallVector<unsigned>>
emitOffsetForSliceLayout(const SliceEncodingAttr &sliceLayout,
                         RankedTensorType type) {
  auto parentEncoding = sliceLayout.getParent();
  unsigned dim = sliceLayout.getDim();
  auto parentShape = sliceLayout.paddedShape(type.getShape());
  RankedTensorType parentTy =
      RankedTensorType::get(parentShape, type.getElementType(), parentEncoding);
  auto parentOffsets = ::intel::emitOffsetForLayout(parentEncoding, parentTy);
  if (parentOffsets.empty())
    return {};

  SmallVector<SmallVector<unsigned>> resultOffsets;
  std::set<SmallVector<unsigned>> uniqueOffsets;

  for (unsigned i = 0; i < parentOffsets.size(); ++i) {
    SmallVector<unsigned> offsets(parentOffsets[i].begin(),
                                  parentOffsets[i].end());
    offsets.erase(offsets.begin() + dim);
    if (auto [it, inserted] = uniqueOffsets.insert(offsets); inserted) {
      resultOffsets.push_back(offsets);
    }
  }

  // It can happen that after deduplicating elements above, resultOffsets has
  // fewer than getTotalElementsPerThread() elements.  In that case repeat the
  // sequence.
  int elemsPerThread = triton::gpu::getTotalElemsPerThread(type);
  assert(resultOffsets.size() > 0);
  assert(elemsPerThread % resultOffsets.size() == 0);
  int numRepeats = elemsPerThread / resultOffsets.size();
  SmallVector<SmallVector<unsigned>> ret;
  for (int i = 0; i < numRepeats; ++i) {
    for (unsigned j = 0; j < resultOffsets.size(); ++j) {
      ret.push_back(SmallVector<unsigned>(resultOffsets[j]));
    }
  }
  return ret;
}

//

// -----------------------------------------------------------------------
// Get offsets / indices for any layout
// -----------------------------------------------------------------------

inline SmallVector<Value>
emitBaseIndexForLayoutImpl(Location loc, RewriterBase &rewriter,
                           const TargetInfoBase &target, Attribute layout,
                           RankedTensorType type, bool withCTAOffset) {
  auto shape = type.getShape();

  SmallVector<Value> baseIndex;
  RewriterBase::InsertionGuard guard(rewriter);
  SmallVector<Value> result;
  if (auto dpasLayout = dyn_cast<DpasEncodingAttr>(layout)) {
    result = emitBaseIndexForDpasLayout(loc, rewriter, dpasLayout, type);
  } else if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(type.getShape());
    RankedTensorType parentTy =
        RankedTensorType::get(parentShape, type.getElementType(), parentLayout);
    result = ::intel::emitBaseIndexForLayoutImpl(
        loc, rewriter, target, parentLayout, parentTy, withCTAOffset);
    result.erase(result.begin() + sliceLayout.getDim());
    // CTAOffset has been added in emitBaseIndexForLayout of parentLayout
    return result;
  } else if (auto dotLayout = dyn_cast<DotOperandEncodingAttr>(layout)) {
    result = emitBaseIndexForDotOpLayout(loc, rewriter, dotLayout, type);
  } else {
    return mlir::emitBaseIndexForLayoutImpl(loc, rewriter, target, layout, type,
                                            withCTAOffset);
  }
  if (withCTAOffset) {
    auto CTAOffset =
        emitCTAOffsetForLayout(loc, rewriter, target, layout, shape);
    assert(CTAOffset.size() == result.size() && "Rank mismatch");
    for (unsigned k = 0; k < result.size(); ++k) {
      // Individual elements of `result` may be null.  In the caller
      // (emitBaseIndexForLayout), we assert that all such dimensions are sliced
      // off.
      if (!result[k])
        continue;
      result[k] = add(result[k], CTAOffset[k]);
    }
  }
  return result;
}

inline SmallVector<Value>
emitBaseIndexForLayout(Location loc, RewriterBase &rewriter,
                       const TargetInfoBase &target, Attribute layout,
                       RankedTensorType type, bool withCTAOffset) {
  SmallVector<Value> idx = ::intel::emitBaseIndexForLayoutImpl(
      loc, rewriter, target, layout, type, withCTAOffset);

  // Check that any null values were sliced out.
  for (Value v : idx) {
    if (!v) {
      llvm::errs() << "Failed to generate indexing code, possibly due to bad "
                      "#mma layout.  Please rerun your program with "
                      "MLIR_ENABLE_DUMP=1 and file a bug."
                   << "\nloc: " << loc << "\nlayout: " << layout
                   << "\ntype: " << type << "\nwithCTAOffset: " << withCTAOffset
                   << "\n";
      llvm::report_fatal_error("Failed to generate indexing code");
    }
  }

  return idx;
}

inline SmallVector<SmallVector<unsigned>>
emitOffsetForLayout(Attribute layout, RankedTensorType type) {
  if (auto dpasLayout = dyn_cast<DpasEncodingAttr>(layout))
    return emitOffsetForDpasLayout(dpasLayout, type);
  if (auto dotLayout = dyn_cast<DotOperandEncodingAttr>(layout))
    return emitOffsetForDotOpLayout(dotLayout, type);
  if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout))
    return ::intel::emitOffsetForSliceLayout(sliceLayout, type);
  return mlir::emitOffsetForLayout(layout, type);
}

// Emit indices calculation within each ConversionPattern, and returns a
// [elemsPerThread X rank] index matrix.
inline SmallVector<SmallVector<Value>>
emitIndices(Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
            Attribute layout, RankedTensorType type, bool withCTAOffset) {
  MLIRContext *ctx = rewriter.getContext();
  auto shape = type.getShape();
  std::optional<LinearLayout> ll = triton::gpu::toLinearLayout(shape, layout);
  if (ll.has_value())
    return mlir::emitIndices(loc, rewriter, target, layout, type,
                             withCTAOffset);

  // step 1, delinearize threadId to get the base index
  auto multiDimBase = ::intel::emitBaseIndexForLayout(
      loc, rewriter, target, layout, type, withCTAOffset);
  // step 2, get offset of each element
  auto offset = intel::emitOffsetForLayout(layout, type);
  // step 3, add offset to base, and reorder the sequence
  // of indices to guarantee that elems in the same
  // sizePerThread are adjacent in order
  unsigned rank = shape.size();
  unsigned elemsPerThread = offset.size();
  SmallVector<SmallVector<Value>> multiDimIdx(elemsPerThread,
                                              SmallVector<Value>(rank));
  for (unsigned n = 0; n < elemsPerThread; ++n)
    for (unsigned k = 0; k < rank; ++k)
      multiDimIdx[n][k] = add(multiDimBase[k], i32_val(offset[n][k]));

  return multiDimIdx;
}

/* ---------------- */
/* ---------------- */
inline DenseMap<unsigned, Value> getSwizzledSharedPtrs(
    Location loc, const TargetInfoBase &target, unsigned inVec,
    RankedTensorType srcTy, triton::gpu::SharedEncodingAttr resSharedLayout,
    Type resElemTy, const SharedMemoryObject &shrMemObj, RewriterBase &rewriter,
    SmallVectorImpl<Value> &offsetVals, SmallVectorImpl<Value> &srcStrides) {
  // This utility computes the pointers for accessing the provided swizzled
  // shared memory layout `resSharedLayout`. More specifically, it computes,
  // for all indices (row, col) of `srcEncoding` such that idx % inVec = 0,
  // the pointer: ptr[(row, col)] = base + (rowOff * strides[ord[1]] +
  // colOff) where :
  //   phase = (row // perPhase) % maxPhase
  //   rowOff = row
  //   colOff = colOffSwizzled + colOffOrdered
  //     colOffSwizzled = ((col // outVec) ^ phase) * outVec
  //     colOffOrdered = (col % outVec) // minVec * minVec
  //
  // Note 1:
  // -------
  // Because swizzling happens at a granularity of outVec, we need to
  // decompose the offset into a swizzled factor and a non-swizzled
  // (ordered) factor
  //
  // Note 2:
  // -------
  // If we have x, y, z of the form:
  // x = 0b00000xxxx
  // y = 0byyyyy0000
  // z = 0b00000zzzz
  // then (x + y) XOR z = 0byyyyxxxx XOR 0b00000zzzz = (x XOR z) + y
  // This means that we can use some immediate offsets for shared memory
  // operations.
  auto dstPtrTy = shrMemObj.base.getType();
  auto dstOffset = dot(rewriter, loc, offsetVals, shrMemObj.strides);
  Value dstPtrBase = gep(dstPtrTy, resElemTy, shrMemObj.base, dstOffset);

  auto srcEncoding = srcTy.getEncoding();
  auto srcShape = srcTy.getShape();
  auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
  // swizzling params as described in TritonGPUAttrDefs.td
  unsigned outVec = resSharedLayout.getVec();
  unsigned perPhase = resSharedLayout.getPerPhase();
  unsigned maxPhase = resSharedLayout.getMaxPhase();
  // Order
  auto inOrder = triton::gpu::getOrder(srcEncoding);
  auto outOrder = triton::gpu::getOrder(resSharedLayout);
  assert(maxPhase == 1 ||
         outVec * maxPhase <= srcShape[outOrder[0]] &&
             "Swizzling would generate out of bounds memory accesses");
  // Tensor indices held by the current thread, as LLVM values
  auto srcIndices = ::intel::emitIndices(loc, rewriter, target, srcEncoding,
                                         srcTy, /*withCTAOffset=*/false);
  // Swizzling with leading offsets (e.g. Hopper GMMA)
  unsigned swizzlingByteWidth = 0;
  if (resSharedLayout.getHasLeadingOffset()) {
    if (perPhase == 4 && maxPhase == 2)
      swizzlingByteWidth = 32;
    else if (perPhase == 2 && maxPhase == 4)
      swizzlingByteWidth = 64;
    else if (perPhase == 1 && maxPhase == 8)
      swizzlingByteWidth = 128;
    else
      llvm::report_fatal_error("Unsupported shared layout.");
  }
  unsigned numElemsPerSwizzlingRow =
      swizzlingByteWidth * 8 / resElemTy.getIntOrFloatBitWidth();
  Value numElemsPerSwizzlingRowVal = i32_val(numElemsPerSwizzlingRow);
  unsigned leadingDimOffset;
  if (outOrder.size() >= 2) {
    leadingDimOffset = numElemsPerSwizzlingRow * srcShapePerCTA[outOrder[1]];
  } else {
    leadingDimOffset = numElemsPerSwizzlingRow;
  }

  Value leadingDimOffsetVal = i32_val(leadingDimOffset);
  // Return values
  DenseMap<unsigned, Value> ret;
  // cache for non-immediate offsets
  DenseMap<unsigned, Value> cacheCol, cacheRow;
  unsigned minVec = std::min(outVec, inVec);
  Value strideRow = outOrder.size() >= 2 ? srcStrides[outOrder[1]] : i32_val(0);
  Value strideCol = srcStrides[outOrder[0]];
  LDBG("getSwizzledSharedPtrs: perPhase = "
       << perPhase << " maxPhase = " << maxPhase << " minVec = " << minVec
       << " inVec = " << inVec << " outVec = " << outVec << " strideRow "
       << strideRow << " strideCol " << strideCol);
  for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
    Value offset = i32_val(0);
    // Extract multi dimensional index for current element
    auto idx = srcIndices[elemIdx];
    Value idxCol = idx[outOrder[0]]; // contiguous dimension
    Value idxRow;
    if (outOrder.size() >= 2) {
      idxRow = idx[outOrder[1]]; // discontiguous dimension
    } else {
      idxRow = i32_val(0);
    }
    // compute phase = (row // perPhase) % maxPhase
    Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
    // extract dynamic/static offset for immediate offsetting
    unsigned immedateOffCol = 0;
    unsigned immedateOffRow = 0;
    if (leadingDimOffset) {
      // hopper
      offset =
          mul(udiv(idxCol, numElemsPerSwizzlingRowVal), leadingDimOffsetVal);
      // Shrink by swizzling blocks
      idxCol = urem(idxCol, numElemsPerSwizzlingRowVal);
      strideRow = numElemsPerSwizzlingRowVal;
    }
    if (auto add = idxCol.getDefiningOp<LLVM::AddOp>()) {
      if (auto _cst = add.getRhs().getDefiningOp<LLVM::ConstantOp>()) {
        unsigned cst =
            cast<IntegerAttr>(_cst.getValue()).getValue().getSExtValue();
        unsigned key = cst % (outVec * maxPhase);
        cacheCol.insert({key, idxCol});
        idxCol = cacheCol[key];
        immedateOffCol = cst / (outVec * maxPhase) * (outVec * maxPhase);
      }
    }
    if (auto add = idxRow.getDefiningOp<LLVM::AddOp>()) {
      if (auto _cst = add.getRhs().getDefiningOp<LLVM::ConstantOp>()) {
        unsigned cst =
            cast<IntegerAttr>(_cst.getValue()).getValue().getSExtValue();
        unsigned key = cst % (perPhase * maxPhase);
        cacheRow.insert({key, idxRow});
        idxRow = cacheRow[key];
        immedateOffRow = cst / (perPhase * maxPhase) * (perPhase * maxPhase);
      }
    }
    // row offset is simply row index
    Value rowOff = mul(idxRow, strideRow);
    // because swizzling happens at a granularity of outVec, we need to
    // decompose the offset into a swizzled factor and a non-swizzled
    // (ordered) factor: colOffSwizzled = ((col // outVec) ^ phase) * outVec
    // colOffOrdered = (col % outVec) // minVec * minVec
    Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
    colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
    Value colOffOrdered = urem(idxCol, i32_val(outVec));
    colOffOrdered = udiv(colOffOrdered, i32_val(minVec));
    colOffOrdered = mul(colOffOrdered, i32_val(minVec));
    Value colOff = add(colOffSwizzled, colOffOrdered);
    // compute non-immediate offset
    if (outOrder.size() == 3)
      offset = add(offset, mul(idx[outOrder[2]], srcStrides[outOrder[2]]));
    offset = add(offset, add(rowOff, mul(colOff, strideCol)));
    Value currPtr = gep(dstPtrTy, resElemTy, dstPtrBase, offset);
    // compute immediate offset
    Value immediateOff;
    if (outOrder.size() >= 2) {
      immediateOff =
          add(mul(i32_val(immedateOffRow), strideRow), i32_val(immedateOffCol));
    } else {
      immediateOff = i32_val(immedateOffCol);
    }

    ret[elemIdx] = gep(dstPtrTy, resElemTy, currPtr, immediateOff);
  }
  return ret;
}

inline SmallVector<Value>
loadSharedToDistributed(RankedTensorType dstTy, MemDescType srcTy,
                        Type elemLlvmTy, SharedMemoryObject &memObj,
                        Location loc, RewriterBase &rewriter,
                        const TargetInfoBase &target) {
  SmallVector<Value> ret;
  bool success = emitTransferBetweenRegistersAndShared(
      dstTy, srcTy, elemLlvmTy, /*maxVecElems=*/std::nullopt, memObj.getBase(),
      memObj.getStrides(), loc, rewriter, target,
      [&](VectorType vecTy, Value vecAddr) {
        auto vecVal = load(vecTy, vecAddr);
        vecVal.setAlignment(vecTy.getNumElements() *
                            elemLlvmTy.getIntOrFloatBitWidth() / 8);
        for (int v = 0; v < vecTy.getNumElements(); v++) {
          ret.push_back(extract_element(elemLlvmTy, vecVal, i32_val(v)));
        }
      });
  if (!success)
    llvm::report_fatal_error("Failed to emit transfer from shared to register");

  return ret;
}

inline void storeDistributedToShared(MemDescType dstTy, RankedTensorType srcTy,
                                     Type elemLlvmTy, ArrayRef<Value> srcVals,
                                     Value smemBase, ArrayRef<Value> dstStrides,
                                     Location loc, RewriterBase &rewriter,
                                     const TargetInfoBase &target) {
  bool success = emitTransferBetweenRegistersAndShared(
      srcTy, dstTy, elemLlvmTy, /*maxVecElems=*/std::nullopt, smemBase,
      dstStrides, loc, rewriter, target, [&](VectorType vecTy, Value vecAddr) {
        ArrayRef<Value> vals = srcVals.take_front(vecTy.getNumElements());
        srcVals = srcVals.drop_front(vecTy.getNumElements());

        Value vec = undef(vecTy);
        for (int i = 0; i < vals.size(); i++) {
          vec = insert_element(vec, vals[i], i32_val(i));
        }
        store(vec, vecAddr)
            .setAlignment(vecTy.getNumElements() *
                          elemLlvmTy.getIntOrFloatBitWidth() / 8);
      });
  if (!success)
    llvm::report_fatal_error("Failed to emit transfer from register to shared");
}

Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        Value v);
Value convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        Value v, RoundingMode rounding);

} // namespace mlir::triton::intel

#endif
