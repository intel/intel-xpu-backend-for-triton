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
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

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
  if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op))
    return encoding;
  if (auto advanceOp = dyn_cast<tt::AdvanceOp>(op))
    return encoding;
  // Dispatch DotEncoding + DPASEncoding to the
  // TritonIntelGPUInferLayoutInterface
  if (auto dotEncoding = dyn_cast<ttg::DotOperandEncodingAttr>(encoding)) {
    auto parentEnc = dyn_cast<ttgi::DpasEncodingAttr>(dotEncoding.getParent());
    assert(
        parentEnc &&
        "Intel inferSrcEncoding requires DpasEncoding for DotOperandEncoding");
    Attribute srcEnc;
    if (auto fp4ToFpOp = dyn_cast<tt::gpu::Fp4ToFpOp>(op)) {
      llvm::ArrayRef<int64_t> shape = fp4ToFpOp.getSrc().getType().getShape();
      if (succeeded(
              parentEnc.getDialect()
                  .getRegisteredInterface<triton::DialectInferLayoutInterface>()
                  ->inferFp4ToFpOpEncoding(
                      shape, fp4ToFpOp.getAxis(), parentEnc, srcEnc,
                      /*fwdInference*/ false, std::nullopt)))
        return srcEnc;
      return {};
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

  // Loads that use a block pointer are expensive if they cannot be lowered to
  // 2D block read operations. Temporarily leverage the
  // "triton_intel_gpu.block_io" attribute to filter out inexpensive loads.
  Attribute blockIOAttr =
      op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
  if (blockIOAttr)
    return false;

  // Loads that use more threads than elements can be presumed to have a high
  // hit-rate that makes them cheap to load.
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
    if (currentValue.getDefiningOp<scf::ForOp>())
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
    func = b.create<LLVM::LLVMFuncOp>(
        symbolTable->getLoc(), name,
        LLVM::LLVMFunctionType::get(resultType, paramTypes));
    func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  }
  return func;
}

LLVM::CallOp createSPIRVBuiltinCall(Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    LLVM::LLVMFuncOp func, ValueRange args) {
  auto call = rewriter.create<LLVM::CallOp>(loc, func, args);
  call.setCConv(func.getCConv());
  return call;
}

} // namespace mlir::triton::gpu::intel
