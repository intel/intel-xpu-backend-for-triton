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

static bool isSingleValue(Value value) {
  // Don't consider load as expensive if it is loading a scalar.
  if (auto tensorTy = dyn_cast<RankedTensorType>(value.getType()))
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

  // Case 3: Value is defined by a sign extension operation
  if (auto extSIOp = value.getDefiningOp<arith::ExtSIOp>())
    return isDivisible(extSIOp->getOperand(0), divisor);

  // hacking for TMA.
  if (auto llvmLoad = value.getDefiningOp<LLVM::LoadOp>())
    return true;

  return false;
}

std::optional<Attribute> inferSrcEncoding(Operation *op, Attribute encoding) {
  if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op))
    return encoding;
  if (auto advanceOp = dyn_cast<tt::AdvanceOp>(op))
    return encoding;

  return mlir::inferSrcEncoding(op, encoding);
}

bool isExpensiveLoadOrStore(Operation *op) {
  assert((isa<tt::LoadOp>(op) || isa<tt::StoreOp>(op)) &&
         "Expecting Triton LoadOp or StoreOp");
  Value base = op->getOperand(0);

  // Case 1: A size 1 tensor is not expensive since all threads will load the
  // same
  if (isSingleValue(base))
    return false;

  // Case 2: Tensor of pointers has more threads than elements
  // we can presume a high hit-rate that makes it cheap to load
  if (auto ptrType = dyn_cast<RankedTensorType>(base.getType())) {
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
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

LogicalResult
getConvertBackwardSlice(Value root, SetVector<Value> &slice,
                        Attribute rootEncoding,
                        DenseMap<Value, Attribute> &layout,
                        std::function<bool(Operation *)> stopPropagation) {
  DenseSet<std::pair<Value, Attribute>> seen;
  SmallVector<std::pair<Value, Attribute>> queue;

  auto enqueue = [&](Value operand, Attribute encoding) {
    auto x = std::make_pair(operand, encoding);
    if (!seen.insert(x).second) {
      return; // Already enqueued, skip
    }
    queue.push_back(x);
  };
  enqueue(root, rootEncoding);

  while (!queue.empty()) {
    auto [currentValue, encoding] = queue.back();
    queue.pop_back();
    if (!isTensorOrTensorPointerType(currentValue.getType()))
      continue;
    slice.insert(currentValue);
    if (layout.find(currentValue) != layout.end()) {
      if (layout[currentValue] != encoding)
        return failure();
    }
    layout[currentValue] = encoding;

    if (auto ifOp = currentValue.getDefiningOp<scf::IfOp>()) {
      auto results = ifOp.getResults();
      unsigned argIdx = mlir::cast<OpResult>(currentValue).getResultNumber();

      auto thenValue = ifOp.thenYield().getOperand(argIdx);
      auto elseValue = ifOp.elseYield().getOperand(argIdx);

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
        enqueue(result, encoding);
      }
      if (isFreeConvert(definingOp)) {
        enqueue(definingOp->getOperand(0), encoding);
        continue;
      }
      if (canFoldIntoConversion(definingOp, encoding))
        continue;
      if (stopPropagation && stopPropagation(definingOp))
        continue;
      if (isa<triton::CatOp>(definingOp))
        return failure();
      for (Value operand : definingOp->getOperands()) {
        auto srcEncoding = ttgi::inferSrcEncoding(definingOp, encoding);
        if (!srcEncoding)
          return failure();
        enqueue(operand, *srcEncoding);
      }
      continue;
    }
    auto blockArg = cast<BlockArgument>(currentValue);
    Block *block = blockArg.getOwner();
    Operation *parentOp = block->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      OpOperand *initOperand = forOp.getTiedLoopInit(blockArg);
      Value yieldOperand = forOp.getBody()->getTerminator()->getOperand(
          blockArg.getArgNumber() - forOp.getNumInductionVars());
      enqueue(initOperand->get(), encoding);
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

static std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (ofr.is<Attribute>() && isa<IntegerAttr>(ofr.get<Attribute>()))
    return cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
  return std::nullopt;
}

std::optional<int64_t> getFoldedConstantValue(Operation *op) {
  SmallVector<OpFoldResult> results;
  if (failed(op->fold(results))) {
    return std::nullopt;
  }

  // If fold succeeded but `results` is empty, we give a second try, after the
  // operands have been switched during the first call to `fold()`.
  if (results.empty()) {
    if (failed(op->fold(results))) {
      return std::nullopt;
    }
  }

  if (results.size() != 1) {
    return std::nullopt;
  }

  auto intAttr = getIntAttr(results[0]);
  if (intAttr.has_value()) {
    return intAttr.value();
  }

  auto val = cast<Value>(results[0]);
  auto constOp = val.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;

  return getIntAttr(constOp.getValue());
}

bool isConstant(Value val, const unsigned expected) {
  auto defOp = val.getDefiningOp();
  if (!defOp)
    return false;
  return (getFoldedConstantValue(defOp) == expected);
}

} // namespace mlir::triton::gpu::intel
