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
  if (mod->getAttrOfType<IntegerAttr>("ttg.shared").getInt() == 0)
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

} // namespace mlir::LLVM::intel

using mlir::triton::gpu::intel::DpasEncodingAttr;

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
  size_t rank = repCluster.size();
  SmallVector<unsigned> sizePerDPASInst = {
      sizePerThreads[rank - 2] / repCluster[rank - 2],
      sizePerThreads[rank - 1] / repCluster[rank - 1]};

  unsigned rowsPerElem = dpasLayout.getSubGroupSize() / instShapeC[1];
  unsigned colsPerElem = 1;

  unsigned repNumber = product<unsigned>(repCluster);
  unsigned elemNumberPerRep = product<unsigned>(sizePerDPASInst);
  for (unsigned repId = 0; repId < repNumber; ++repId) {
    for (unsigned elemId = 0; elemId < elemNumberPerRep; ++elemId) {
      // Follows the C++ order for the dpas layout.
      SmallVector<unsigned> repOffset = {
          (repId / repCluster[rank - 1]) * instShapeC[0],
          (repId % repCluster[rank - 1]) * instShapeC[1]};

      SmallVector<unsigned> elemOffset = {
          (elemId / sizePerDPASInst[1]) * rowsPerElem,
          (elemId % sizePerDPASInst[1]) * colsPerElem};

      if (rank == 3)
        offsets.push_back({0, repOffset[0] + elemOffset[0] + ctaOffsetX,
                           repOffset[1] + elemOffset[1] + ctaOffsetY});
      else {
        assert((rank == 2) && "unexpected rank number for Dpas layout");
        offsets.push_back({repOffset[0] + elemOffset[0] + ctaOffsetX,
                           repOffset[1] + elemOffset[1] + ctaOffsetY});
      }
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

  unsigned rank = shape.size();
  unsigned numRowsPerPackedValue = 0u, numColsPerPackedValue = 0u;
  unsigned numColsPerLaneForPackedValue = 0u, numOpsPerPackedValue = 0u;
  switch (opIdx) {
  case 0: {
    assert((opsPerChannel == 4 || opsPerChannel == 2 || opsPerChannel == 1) &&
           "invalid opsPerChannel number.");
    SmallVector<unsigned> shapeA = dpasLayout.getShapeA();
    // Unlike the operand B, to pack the value to i16 for scalar bit width <=16.
    numOpsPerPackedValue = opsPerChannel == 4 ? 2 : 1;
    unsigned packedColNum = shapeA[rank - 1] / numOpsPerPackedValue;
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
  int64_t numRepOuter = numReps[opIdx ? 2 : 1];
  int64_t numRepK = numReps[opIdx ? 1 : 2];

  ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
  unsigned repClusterSize = repCluster[opIdx ? rank - 1 : rank - 2];

  for (unsigned repOuter = 0; repOuter < numRepOuter; ++repOuter)
    for (unsigned k = 0; k < numRepK; ++k)
      for (unsigned rep = 0; rep < repClusterSize; ++rep) {
        for (unsigned elemId = 0; elemId < numElemPerInstPerThread; ++elemId) {
          unsigned opsRowIndex =
              (opIdx == 0) ? 0 : elemId % numOpsPerPackedValue;
          unsigned opsColIndex =
              (opIdx == 0) ? elemId % numOpsPerPackedValue : 0;
          unsigned packedElemId = elemId / numOpsPerPackedValue;
          unsigned repRowIndex =
              shapePerCTATile[rank - 2] * (opIdx == 0 ? repOuter : k);
          unsigned repColIndex =
              shapePerCTATile[rank - 1] * (opIdx == 0 ? k : repOuter);
          unsigned repClusterRowIndex = opIdx == 0 ? rep * instShape[0] : 0;
          unsigned repClusterColIndex = opIdx == 0 ? 0 : rep * instShape[1];
          unsigned packedElemRowIndex =
              (packedElemId / numColsPerLaneForPackedValue) *
              numRowsPerPackedValue;
          unsigned packedElemColIndex =
              (packedElemId % numColsPerLaneForPackedValue) *
              numColsPerPackedValue;
          if (rank == 3)
            offsets.push_back({0,
                               repRowIndex + repClusterRowIndex +
                                   packedElemRowIndex + opsRowIndex,
                               repColIndex + repClusterColIndex +
                                   packedElemColIndex + opsColIndex});
          else {
            assert((rank == 2) && "unexpected rank number for Dot layout");
            offsets.push_back({repRowIndex + repClusterRowIndex +
                                   packedElemRowIndex + opsRowIndex,
                               repColIndex + repClusterColIndex +
                                   packedElemColIndex + opsColIndex});
          }
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
  size_t rank = shape.size();

  for (unsigned i = 0; i < shape[rank - 2]; i += shapePerCTA[rank - 2]) {
    for (unsigned j = 0; j < shape[rank - 1]; j += shapePerCTA[rank - 1]) {
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

  size_t rank = warpShape.size();
  assert(rank == shapePerCTA.size() && "Rank mismatch");
  Value warpIndex =
      (opIdx == 0) ? urem(multiDimWarpId[rank - 2],
                          i32_val(mlir::ceil<unsigned>(shapePerCTA[rank - 2],
                                                       warpShape[rank - 2])))
                   : urem(multiDimWarpId[rank - 1],
                          i32_val(mlir::ceil<unsigned>(shapePerCTA[rank - 1],
                                                       warpShape[rank - 1])));
  Value warpOffset =
      mul(warpIndex, i32_val(warpShape[opIdx ? rank - 1 : rank - 2]));

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
    unsigned packedColNum = shapeA[rank - 1] / packedOpsPerLane;
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
  default: {
    llvm::report_fatal_error("Only support opIdx 1 or 0 for DotOpLayout.");
  }
  }

  SmallVector<Value> multiDimBase(rank);
  if (rank == 3)
    multiDimBase[0] = multiDimWarpId[0];
  multiDimBase[rank - 2] =
      (opIdx == 0) ? add(laneRowIndex, warpOffset) : laneRowIndex;
  multiDimBase[rank - 1] =
      (opIdx == 0) ? laneColIndex : add(laneColIndex, warpOffset);

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

  size_t rank = type.getShape().size();
  auto warpsPerCTA = dpasLayout.getWarpsPerCTA();
  ArrayRef<int64_t> shape = type.getShape();

  auto order = triton::gpu::getOrder(dpasLayout);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  // Compute the 2-dim coordinates of the warp containing the tensor element
  // operated on by this thread.
  SmallVector<unsigned> warpShape = dpasLayout.getShapeC();
  Value rowWarpId =
      urem(multiDimWarpId[rank - 2],
           i32_val(mlir::ceil<unsigned>(shape[rank - 2], warpShape[rank - 2])));
  Value colWarpId =
      urem(multiDimWarpId[rank - 1],
           i32_val(mlir::ceil<unsigned>(shape[rank - 1], warpShape[rank - 1])));
  Value rowWarpOffset = mul(rowWarpId, i32_val(warpShape[rank - 2]));
  Value colWarpOffset = mul(colWarpId, i32_val(warpShape[rank - 1]));

  // Compute the 2-dim coordinates of the first element in the warp operated
  // on by this thread.
  SmallVector<unsigned> threadsPerWarp = getThreadsPerWarp(dpasLayout);
  SmallVector<Value> multiDimBase(rank);
  if (rank == 3)
    multiDimBase[0] = multiDimWarpId[0];
  multiDimBase[rank - 2] =
      add(udiv(laneId, i32_val(threadsPerWarp[rank - 1])), rowWarpOffset);
  multiDimBase[rank - 1] =
      add(urem(laneId, i32_val(threadsPerWarp[rank - 1])), colWarpOffset);
  return multiDimBase;
}

namespace mlir::triton::intel {

inline SmallVector<SmallVector<unsigned>>
emitOffsetForLayout(Attribute layout, RankedTensorType type);

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

Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        Value v);
Value convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        Value v, RoundingMode rounding);

} // namespace mlir::triton::intel

#endif // TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H
