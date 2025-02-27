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

LLVM::LLVMFuncOp getSpirvPrintfDeclaration(RewriterBase &rewriter);

static Value getModuleWarpSize(RewriterBase &rewriter, Location loc) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  return b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
}

Value convertFp32ToFp16(Location loc, ConversionPatternRewriter &rewriter,
                        const Value &v, triton::RoundingMode rounding);

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

  unsigned rowsPerElem =
      product<unsigned>(dpasLayout.getThreadsPerWarp()) / instShapeC[1];
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

  auto opIdx = static_cast<DpasEncodingAttr::OpIdx>(dotLayout.getOpIdx());
  SmallVector<int64_t> numReps =
      dpasLayout.getDPASRepetitions(shapePerCTA, opIdx);
  SmallVector<unsigned> warpShape = (opIdx == DpasEncodingAttr::OpIdx::OperandA)
                                        ? dpasLayout.getShapeA()
                                        : dpasLayout.getShapeB();
  SmallVector<unsigned> instShape = (opIdx == DpasEncodingAttr::OpIdx::OperandA)
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
  case DpasEncodingAttr::OpIdx::OperandA: {
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
  case DpasEncodingAttr::OpIdx::OperandB: {
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
  int64_t numRepOuter = numReps[unsigned(opIdx) ? 2 : 1];
  int64_t numRepK = numReps[unsigned(opIdx) ? 1 : 2];

  ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
  unsigned repClusterSize = repCluster[bool(opIdx) ? rank - 1 : rank - 2];

  for (unsigned repOuter = 0; repOuter < numRepOuter; ++repOuter)
    for (unsigned k = 0; k < numRepK; ++k)
      for (unsigned rep = 0; rep < repClusterSize; ++rep) {
        for (unsigned elemId = 0; elemId < numElemPerInstPerThread; ++elemId) {
          bool isOperandA = (opIdx == DpasEncodingAttr::OpIdx::OperandA);
          unsigned opsRowIndex = isOperandA ? 0 : elemId % numOpsPerPackedValue;
          unsigned opsColIndex = isOperandA ? elemId % numOpsPerPackedValue : 0;
          unsigned packedElemId = elemId / numOpsPerPackedValue;
          unsigned repRowIndex =
              shapePerCTATile[rank - 2] * (isOperandA ? repOuter : k);
          unsigned repColIndex =
              shapePerCTATile[rank - 1] * (isOperandA ? k : repOuter);
          unsigned repClusterRowIndex = isOperandA ? rep * instShape[0] : 0;
          unsigned repClusterColIndex = isOperandA ? 0 : rep * instShape[1];
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
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto dpasLayout = dyn_cast<DpasEncodingAttr>(dotLayout.getParent());
  if (!dpasLayout) {
    llvm::errs() << "dotLayout: " << dotLayout << "\n";
    llvm_unreachable(
        "unsupported parent layout in emitBaseIndexForDotOpLayout");
  }

  Value threadId = getThreadId(rewriter, loc);
  unsigned warpSize = triton::gpu::getWarpSize(dpasLayout);
  Value warpId = b.udiv(threadId, b.i32_val(warpSize));
  Value laneId = b.urem(threadId, b.i32_val(warpSize));

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
      (opIdx == 0)
          ? b.urem(multiDimWarpId[rank - 2],
                   b.i32_val(mlir::ceil<unsigned>(shapePerCTA[rank - 2],
                                                  warpShape[rank - 2])))
          : b.urem(multiDimWarpId[rank - 1],
                   b.i32_val(mlir::ceil<unsigned>(shapePerCTA[rank - 1],
                                                  warpShape[rank - 1])));
  Value warpOffset =
      b.mul(warpIndex, b.i32_val(warpShape[opIdx ? rank - 1 : rank - 2]));

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

    laneRowIndex = b.udiv(laneId, b.i32_val(packedColNum));
    laneColIndex = b.urem(laneId, b.i32_val(packedColNum));
    laneColIndex = b.mul(laneColIndex, b.i32_val(packedOpsPerLane));
  } break;
  case 1: {
    if (warpSize < executionSize)
      llvm::report_fatal_error(
          "DpasEncodingAttr sub-group size could not "
          "be smaller than the execution size for B operand.");

    laneRowIndex = b.udiv(laneId, b.i32_val(executionSize));
    laneRowIndex = b.mul(laneRowIndex, b.i32_val(opsPerChannel));
    laneColIndex = b.urem(laneId, b.i32_val(executionSize));
  } break;
  default: {
    llvm::report_fatal_error("Only support opIdx 1 or 0 for DotOpLayout.");
  }
  }

  SmallVector<Value> multiDimBase(rank);
  if (rank == 3)
    multiDimBase[0] = multiDimWarpId[0];
  multiDimBase[rank - 2] =
      (opIdx == 0) ? b.add(laneRowIndex, warpOffset) : laneRowIndex;
  multiDimBase[rank - 1] =
      (opIdx == 0) ? laneColIndex : b.add(laneColIndex, warpOffset);

  return multiDimBase;
}

static SmallVector<Value>
emitBaseIndexForDpasLayout(Location loc, RewriterBase &rewriter,
                           const DpasEncodingAttr &dpasLayout,
                           RankedTensorType type) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = b.i32_val(triton::gpu::getWarpSize(dpasLayout));
  Value warpId = b.udiv(threadId, warpSize);
  Value laneId = b.urem(threadId, warpSize);

  size_t rank = type.getShape().size();
  auto warpsPerCTA = dpasLayout.getWarpsPerCTA();
  ArrayRef<int64_t> shape = type.getShape();

  auto order = triton::gpu::getOrder(dpasLayout);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  // Compute the 2-dim coordinates of the warp containing the tensor element
  // operated on by this thread.
  SmallVector<unsigned> warpShape = dpasLayout.getShapeC();
  Value rowWarpId = b.urem(
      multiDimWarpId[rank - 2],
      b.i32_val(mlir::ceil<unsigned>(shape[rank - 2], warpShape[rank - 2])));
  Value colWarpId = b.urem(
      multiDimWarpId[rank - 1],
      b.i32_val(mlir::ceil<unsigned>(shape[rank - 1], warpShape[rank - 1])));
  Value rowWarpOffset = b.mul(rowWarpId, b.i32_val(warpShape[rank - 2]));
  Value colWarpOffset = b.mul(colWarpId, b.i32_val(warpShape[rank - 1]));

  // Compute the 2-dim coordinates of the first element in the warp operated
  // on by this thread.
  SmallVector<unsigned> threadsPerWarp = getThreadsPerWarp(dpasLayout);
  SmallVector<Value> multiDimBase(rank);
  if (rank == 3)
    multiDimBase[0] = multiDimWarpId[0];
  multiDimBase[rank - 2] =
      b.add(b.udiv(laneId, b.i32_val(threadsPerWarp[rank - 1])), rowWarpOffset);
  multiDimBase[rank - 1] =
      b.add(b.urem(laneId, b.i32_val(threadsPerWarp[rank - 1])), colWarpOffset);
  return multiDimBase;
}

namespace mlir::triton::intel {

Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        Value v);
Value convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        Value v, RoundingMode rounding);

} // namespace mlir::triton::intel

#endif // TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H
