//===- Utility.cpp - Triton Intel GPU utilities -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/Utility.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/Support/MathExtras.h"

#include <optional>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {

RankedTensorType getRankedTensorType(Type ptrTy) {
  return tt::isTensorPointerType(ptrTy)
             ? cast<RankedTensorType>(
                   cast<tt::PointerType>(ptrTy).getPointeeType())
             : dyn_cast<RankedTensorType>(ptrTy);
}

static bool isSingleValue(Value value) {
  // Don't consider load as expensive if it is loading a scalar.
  if (auto tensorTy = getRankedTensorType(value.getType()))
    return tensorTy.getNumElements() == 1;
  // TODO: Handle other cases.
  // For example, when ptr is a tensor of single value.
  // It means that ptr is a resultant of broadcast or generated through
  // a chain of broadcast and other operations.
  // Rematerialize it without considering contiguous memory access pattern is
  // fine.
  return true;
}

bool isDivisible(Value value, unsigned divisor) {
  // Case 1: Value is defined by a constant operation
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
    return integerAttr && integerAttr.getValue().getZExtValue() % divisor == 0;
  }

  // Case 2: Value is a block argument of the entry block
  if (value.getParentBlock()->isEntryBlock() && isa<BlockArgument>(value)) {
    BlockArgument blockArg = cast<BlockArgument>(value);
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto funcOp = dyn_cast<tt::FuncOp>(parentOp)) {
      auto divisibilityAttr = funcOp.getArgAttrOfType<IntegerAttr>(
          blockArg.getArgNumber(), "tt.divisibility");
      return divisibilityAttr &&
             divisibilityAttr.getValue().getZExtValue() % divisor == 0;
    }
  }

  // Case 3: Value is defined by a muli operation.
  if (auto mulIOp = value.getDefiningOp<arith::MulIOp>()) {
    return isDivisible(mulIOp->getOperand(0), divisor) ||
           isDivisible(mulIOp->getOperand(1), divisor);
  }

  // Case 4: Value is defined by arith::ExtSIOp, tt::AddPtrOp or
  // arith::AddIOp operation.
  if (auto *op = value.getDefiningOp()) {
    if (isa<arith::ExtSIOp, tt::AddPtrOp, arith::AddIOp>(op)) {
      return llvm::all_of(op->getOperands(), [&](Value operand) {
        return isDivisible(operand, divisor);
      });
    }
  }

  return false;
}

Attribute inferSrcEncoding(Operation *op, Attribute encoding) {
  if (auto makeTensorPtrOp = dyn_cast<MakeTensorPtrOp>(op))
    return encoding;
  if (auto advanceOp = dyn_cast<AdvanceOp>(op))
    return encoding;

  if (auto dotEnc = dyn_cast<DotOperandEncodingAttr>(encoding)) {
    if (auto parentEnc = dyn_cast<DpasEncodingAttr>(dotEnc.getParent())) {
      if (auto fp4ToFpOp = dyn_cast<gpu::Fp4ToFpOp>(op)) {
        // Dispatch DotEncoding + DPASEncoding to the
        // TritonIntelGPUInferLayoutInterface
        Attribute srcEnc;
        llvm::ArrayRef<int64_t> shape = fp4ToFpOp.getSrc().getType().getShape();
        if (succeeded(parentEnc.getDialect()
                          .getRegisteredInterface<DialectInferLayoutInterface>()
                          ->inferFp4ToFpOpEncoding(
                              shape, fp4ToFpOp.getAxis(), parentEnc, srcEnc,
                              /*fwdInference*/ false, std::nullopt)))
          return srcEnc;
        return {};
      }
    }
  }

  return mlir::inferSrcEncoding(op, encoding);
}

bool isExpensiveLoadOrStore(Operation *op) {
  assert((isa<tt::LoadOp>(op) || isa<tt::StoreOp>(op)) &&
         "Expecting Triton LoadOp or StoreOp");
  Value base = op->getOperand(0);

  // A size 1 tensor is not expensive since all threads will load the same
  // value.
  if (isSingleValue(base))
    return false;

  // Loads or stores that use a block pointer are expensive if they cannot be
  // lowered to 2D block read/write operations. Temporarily leverage the
  // "ttig.block_io" attribute to filter out inexpensive loads.
  Attribute blockIOAttr =
      op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
  if (blockIOAttr)
    return false;

  // Loads or stores that use more threads than elements can be presumed to have
  // a high hit-rate that makes them cheap.
  if (auto ptrType = getRankedTensorType(base.getType())) {
    int numWarps = ttg::lookupNumWarps(op);
    auto mod = op->getParentOfType<ModuleOp>();
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    return ptrType.getNumElements() >= numWarps * threadsPerWarp;
  }

  return false;
}

bool hasDotDpasEncoding(RankedTensorType tensorType) {
  if (!tensorType.getEncoding())
    return false;

  auto dotLayout =
      dyn_cast<ttg::DotOperandEncodingAttr>(tensorType.getEncoding());
  if (!dotLayout)
    return false;

  return isa<ttgi::DpasEncodingAttr>(dotLayout.getParent());
}

bool hasDpasEncoding(RankedTensorType tensorType) {
  return isa_and_nonnull<ttgi::DpasEncodingAttr>(tensorType.getEncoding());
}

std::optional<DotOperandEncodingAttr>
getDotEncoding(RankedTensorType tensorType) {
  if (!tensorType.getEncoding())
    return std::nullopt;

  auto dotLayout =
      dyn_cast<ttg::DotOperandEncodingAttr>(tensorType.getEncoding());
  if (!dotLayout)
    return std::nullopt;

  return dotLayout;
}

// Check if the convert will be performed by reordering registers.
static bool isFreeConvert(Operation *op) {
  auto convertOp = dyn_cast<ttg::ConvertLayoutOp>(op);
  if (!convertOp)
    return false;
  return cvtReordersRegisters(convertOp.getSrc().getType(),
                              convertOp.getType());
}

LogicalResult getConvertBackwardSlice(
    OpOperand &root, SetVector<Value> &slice, Attribute rootEncoding,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation,
    std::function<Value(OpOperand &, Attribute)> getExistingConversion) {
  DenseSet<std::pair<OpOperand *, Attribute>> seen;
  SmallVector<std::pair<OpOperand *, Attribute>> queue;

  std::optional<bool> enableForLoopSupport =
      mlir::triton::tools::isEnvValueBool(mlir::triton::tools::getStrEnv(
          "TRITON_INTEL_REMOVELAYOUTCONVERSION_SUPPORT_FOR_LOOP"));

  auto enqueue = [&](OpOperand &operand, Attribute encoding) {
    auto x = std::make_pair(&operand, encoding);
    if (!seen.insert(x).second) {
      return; // Already enqueued, skip
    }
    queue.push_back(x);
  };
  enqueue(root, rootEncoding);

  auto updateLayout = [&](Value value, Attribute encoding) {
    assert(isTensorOrTensorPointerType(value.getType()));

    if (RankedTensorType tensorType = getRankedTensorType(value.getType()))
      if (tensorType.getEncoding() == encoding)
        return success();

    slice.insert(value);
    Attribute &existing = layout[value];
    if (existing && existing != encoding)
      return failure();
    existing = encoding;
    return success();
  };

  while (!queue.empty()) {
    auto [currentValueUse, encoding] = queue.back();
    Value currentValue = currentValueUse->get();
    queue.pop_back();
    if (!isTensorOrTensorPointerType(currentValue.getType()))
      continue;

    // Skip propagating through for op results for now.
    // TODO: enable this based on needs.
    if (!enableForLoopSupport && currentValue.getDefiningOp<scf::ForOp>())
      return failure();

    if (failed(updateLayout(currentValue, encoding)))
      return failure();

    Value existing;
    if (getExistingConversion &&
        (existing = getExistingConversion(*currentValueUse, encoding))) {
      if (failed(updateLayout(existing, encoding)))
        return failure();
      currentValue = existing;
    }
    if (auto forOp = currentValue.getDefiningOp<scf::ForOp>()) {
      if (stopPropagation && stopPropagation(forOp))
        continue;

      auto loopRes = cast<OpResult>(currentValue);
      OpOperand *initOperand = forOp.getTiedLoopInit(loopRes);
      BlockArgument blockArg = forOp.getTiedLoopRegionIterArg(loopRes);
      OpOperand *yieldOperand = forOp.getTiedLoopYieldedValue(blockArg);

      enqueue(*initOperand, encoding);
      enqueue(*yieldOperand, encoding);

      continue;
    }
    if (auto ifOp = currentValue.getDefiningOp<scf::IfOp>()) {
      if (stopPropagation && stopPropagation(ifOp))
        continue;
      unsigned argIdx = mlir::cast<OpResult>(currentValue).getResultNumber();

      OpOperand &thenValue = ifOp.thenYield()->getOpOperand(argIdx);
      OpOperand &elseValue = ifOp.elseYield()->getOpOperand(argIdx);

      enqueue(thenValue, encoding);
      enqueue(elseValue, encoding);

      continue;
    }
    if (auto *definingOp = currentValue.getDefiningOp()) {
      // If the op has multiple results we need to update all results layout.
      for (Value result : definingOp->getResults()) {
        if (result == currentValue ||
            !isTensorOrTensorPointerType(result.getType()))
          continue;
        if (failed(updateLayout(result, encoding)))
          return failure();
      }
      if (isFreeConvert(definingOp)) {
        enqueue(definingOp->getOpOperand(0), encoding);
        continue;
      }
      if (canFoldIntoConversion(definingOp, encoding))
        continue;
      if (stopPropagation && stopPropagation(definingOp))
        continue;
      if (isa<triton::CatOp>(definingOp))
        return failure();
      if (auto gather = dyn_cast<GatherOp>(definingOp)) {
        // Specially handle gather since its transfer function only applies
        // between its index operand and result.
        auto srcEncoding = ttgi::inferSrcEncoding(gather, encoding);
        if (!srcEncoding)
          return failure();
        enqueue(gather.getIndicesMutable(), srcEncoding);
        continue;
      }
      for (auto [i, operand] : llvm::enumerate(definingOp->getOpOperands())) {
        auto srcEncoding = ttgi::inferSrcEncoding(definingOp, encoding);
        if (!srcEncoding)
          return failure();
        enqueue(operand, srcEncoding);
      }
      continue;
    }
    auto blockArg = cast<BlockArgument>(currentValue);
    Block *block = blockArg.getOwner();
    Operation *parentOp = block->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      OpOperand *initOperand = forOp.getTiedLoopInit(blockArg);
      OpOperand &yieldOperand = forOp.getBody()->getTerminator()->getOpOperand(
          blockArg.getArgNumber() - forOp.getNumInductionVars());
      enqueue(*initOperand, encoding);
      enqueue(yieldOperand, encoding);
      continue;
    }
    // TODO: add support for WhileOp and other region types.
    return failure();
  }
  return success();
}

LLVM::LLVMFuncOp lookupOrCreateSPIRVFn(Operation *symbolTable, StringRef name,
                                       ArrayRef<Type> paramTypes,
                                       Type resultType) {
  auto func = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(symbolTable, name));
  if (!func) {
    OpBuilder b(symbolTable->getRegion(0));
    func = LLVM::LLVMFuncOp::create(
        b, symbolTable->getLoc(), name,
        LLVM::LLVMFunctionType::get(resultType, paramTypes));
    func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  }
  return func;
}

LLVM::CallOp createSPIRVBuiltinCall(Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    LLVM::LLVMFuncOp func, ValueRange args) {
  auto call = LLVM::CallOp::create(rewriter, loc, func, args);
  call.setCConv(func.getCConv());
  return call;
}

SmallVector<unsigned> calculateDPASInstShapeA(unsigned repeatCount,
                                              unsigned systolicDepth,
                                              unsigned opsPerChannel) {
  return {repeatCount, systolicDepth * opsPerChannel};
}

SmallVector<unsigned> calculateDPASInstShapeB(unsigned systolicDepth,
                                              unsigned opsPerChannel,
                                              unsigned executionSize) {
  return {systolicDepth * opsPerChannel, executionSize};
}

SmallVector<unsigned> calculateDPASInstShapeC(unsigned repeatCount,
                                              unsigned executionSize) {
  return {repeatCount, executionSize};
}

SmallVector<unsigned> calculateShapeA(unsigned repeatCount,
                                      unsigned systolicDepth,
                                      unsigned opsPerChannel,
                                      ArrayRef<unsigned> repCluster) {
  SmallVector<unsigned> instShapeA =
      calculateDPASInstShapeA(repeatCount, systolicDepth, opsPerChannel);
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeA[0] * repCluster[rank - 2];
  resShape[rank - 1] = instShapeA[1];
  return resShape;
}

SmallVector<unsigned> calculateShapeB(unsigned systolicDepth,
                                      unsigned opsPerChannel,
                                      unsigned executionSize,
                                      ArrayRef<unsigned> repCluster) {
  SmallVector<unsigned> instShapeB =
      calculateDPASInstShapeB(systolicDepth, opsPerChannel, executionSize);
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeB[0];
  resShape[rank - 1] = instShapeB[1] * repCluster[rank - 1];
  return resShape;
}

SmallVector<unsigned> calculateShapeC(unsigned repeatCount,
                                      unsigned executionSize,
                                      ArrayRef<unsigned> repCluster) {
  SmallVector<unsigned> instShapeC =
      calculateDPASInstShapeC(repeatCount, executionSize);
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeC[0] * repCluster[rank - 2];
  resShape[rank - 1] = instShapeC[1] * repCluster[rank - 1];
  return resShape;
}

SmallVector<unsigned> calculateWarpsPerTile(unsigned capRepeatCount,
                                            unsigned capExecutionSize,
                                            const ArrayRef<int64_t> shape,
                                            unsigned numWarps) {
  size_t rank = shape.size();
  SmallVector<unsigned> ret(rank, 1);

  if (rank == 3) {
    int batchWarp = numWarps;
    while (batchWarp > shape[0])
      batchWarp /= 2;
    ret[0] = batchWarp;
    numWarps /= batchWarp;
  }

  // Try to find a proper tiling shape for the dot operation.
  // It doubles the warp number in col or row in each time based on column to
  // width ratio.
  // By this, we can minimize the duplication of the dot operands A and B.
  SmallVector<int64_t> shapePerWarp{capRepeatCount, capExecutionSize};
  uint32_t rowColRatio = llvm::divideCeil(capRepeatCount, capExecutionSize);
  uint32_t colRowRatio = llvm::divideCeil(capExecutionSize, capRepeatCount);

  int rowDim = rank - 2, colDim = rank - 1;
  do {
    if (ret[rowDim] * ret[colDim] >= numWarps)
      break;
    if (shape[rowDim] / (shapePerWarp[0] * colRowRatio) / ret[rowDim] >=
        shape[colDim] / (shapePerWarp[1] * rowColRatio) / ret[colDim]) {
      if (ret[rowDim] < shape[rowDim] / shapePerWarp[0])
        ret[rowDim] *= 2;
      else
        ret[colDim] *= 2;
    } else {
      ret[colDim] *= 2;
    }
  } while (true);

  return ret;
}

SmallVector<unsigned>
calculateRepCluster(unsigned capRepeatCount, unsigned capSystolicDepth,
                    unsigned capExecutionSize, unsigned opsPerChan,
                    ArrayRef<int64_t> retShape, unsigned threadsPerWarp,
                    unsigned int a_bitwidth, bool is_FP8,
                    ArrayRef<int64_t> a_shape, ArrayRef<int64_t> b_shape,
                    SmallVector<unsigned> warpsPerTile) {
  size_t rank = retShape.size();
  SmallVector<unsigned> repCluster(rank, 1);

  unsigned repeatCount =
      std::min(capRepeatCount, (unsigned)retShape[rank - 2] /*M*/);
  unsigned numElemsPerRowForA =
      opsPerChan == 1 ? capSystolicDepth
                      : capSystolicDepth * 2; // A is packed to i16 or i32.
  unsigned minM = llvm::divideCeil(threadsPerWarp, numElemsPerRowForA);
  repeatCount = std::max(repeatCount, minM);

  if (capExecutionSize == 16) {
    unsigned dpasElemBitWidths = a_bitwidth;

    // We are upcasting FP8 to FP16
    if (is_FP8)
      dpasElemBitWidths = 2 * dpasElemBitWidths;

    // Enlarge the repCluster size to use the large 2D load for A and B
    // operands.
    constexpr unsigned PVC_2D_LOAD_MAXIMUM_NUMBER_OF_ROWS = 32;
    constexpr unsigned PVC_2D_LOAD_MAXIMUM_BYTES_OF_COLS = 64;

    unsigned maxRepClusterM =
        PVC_2D_LOAD_MAXIMUM_NUMBER_OF_ROWS / capRepeatCount;
    SmallVector<int64_t> repA = calculateDPASRepetitions(
        a_shape, static_cast<ttgi::DpasEncodingAttr::OpIdx>(0), warpsPerTile,
        repCluster, repeatCount, capSystolicDepth, capExecutionSize,
        opsPerChan);

    unsigned repClusterDimM =
        std::min(maxRepClusterM, static_cast<unsigned>(repA[1]));

    unsigned maxRepClusterN = PVC_2D_LOAD_MAXIMUM_BYTES_OF_COLS /
                              ((dpasElemBitWidths / 8) * capExecutionSize);
    SmallVector<int64_t> repB = calculateDPASRepetitions(
        b_shape, static_cast<ttgi::DpasEncodingAttr::OpIdx>(1), warpsPerTile,
        repCluster, repeatCount, capSystolicDepth, capExecutionSize,
        opsPerChan);

    unsigned repClusterDimN =
        std::min(maxRepClusterN, static_cast<unsigned>(repB[2]));
    repCluster[rank - 2] = repClusterDimM;
    repCluster[rank - 1] = repClusterDimN;
  }

  return repCluster;
}

SmallVector<int64_t>
calculateDPASRepetitions(ArrayRef<int64_t> shape, DpasEncodingAttr::OpIdx opIdx,
                         ArrayRef<unsigned> warpsPerCTA,
                         ArrayRef<unsigned> repCluster, unsigned repeatCount,
                         unsigned systolicDepth, unsigned executionSize,
                         unsigned opsPerChannel) {
  // Always return a 3D shape repetitions for the ease of value handling, same
  // to mma.
  size_t rank = shape.size();
  SmallVector<int64_t> rep(3, 1);

  switch (opIdx) {
  case DpasEncodingAttr::OpIdx::OperandA: {
    SmallVector<unsigned> shapePerWarp =
        calculateShapeA(repeatCount, systolicDepth, opsPerChannel, repCluster);

    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / (shapePerWarp[rank - 2] *
                                                    warpsPerCTA[rank - 2])),
            std::max<int64_t>(1, shape[rank - 1] / shapePerWarp[rank - 1])};
  } break;
  case DpasEncodingAttr::OpIdx::OperandB: {
    SmallVector<unsigned> shapePerWarp = calculateShapeB(
        systolicDepth, opsPerChannel, executionSize, repCluster);

    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / shapePerWarp[rank - 2]),
            std::max<int64_t>(1, shape[rank - 1] / (shapePerWarp[rank - 1] *
                                                    warpsPerCTA[rank - 1]))};
  } break;
  case DpasEncodingAttr::OpIdx::OperandC: {
    SmallVector<unsigned> shapePerWarp =
        calculateShapeC(repeatCount, executionSize, repCluster);

    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / (shapePerWarp[rank - 2] *
                                                    warpsPerCTA[rank - 2])),
            std::max<int64_t>(1, shape[rank - 1] / (shapePerWarp[rank - 1] *
                                                    warpsPerCTA[rank - 1]))};
  } break;
  }

  llvm_unreachable("unexpected opIdx");
}

} // namespace mlir::triton::gpu::intel
