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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <optional>

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {

bool supportDPAS(DotOp op, DeviceArch arch) {
  if (arch == DeviceArch::UNKNOWN)
    return false;

  auto mod = op->getParentOfType<mlir::ModuleOp>();
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);

  if (arch == DeviceArch::PVC && threadsPerWarp != 16) {
    // Only support threadsPerWarp 16 for PVC now.
    return false;
  }

  if (arch == DeviceArch::ATS && threadsPerWarp != 8) {
    // Only support threadsPerWarp 8 for ATS now.
    return false;
  }

  DPASEngineType dpasType = getDPASType(op);

  if (dpasType == DPASEngineType::FP32_FP32_TF32_TF32) {
    // Only PVC support TF32.
    return op.getInputPrecision() == InputPrecision::TF32 &&
           arch == DeviceArch::PVC;
  }

  return dpasType != DPASEngineType::NOT_APPLICABLE;
}

DPASEngineType getDPASType(DotOp op) {
  // d = a * b + c
  auto aTy = cast<RankedTensorType>(op.getA().getType());
  auto bTy = cast<RankedTensorType>(op.getB().getType());
  auto cTy = cast<RankedTensorType>(op.getC().getType());
  auto dTy = cast<RankedTensorType>(op.getD().getType());

  if (aTy.getElementType() != bTy.getElementType() ||
      cTy.getElementType() != dTy.getElementType())
    return DPASEngineType::NOT_APPLICABLE;

  // TODO: add more dpas supported data type.
  if (dTy.getElementType().isIntOrIndex()) {
    // Integer
    if (dTy.getElementType().getIntOrFloatBitWidth() == 32) {
      if (aTy.getElementType().getIntOrFloatBitWidth() == 8 &&
          bTy.getElementType().getIntOrFloatBitWidth() == 8)
        return dTy.getElementType().isSignedInteger()
                   ? DPASEngineType::S32_S32_S8_S8
                   : DPASEngineType::U32_U32_U8_U8;
    }
  } else {
    // floating.
    if (dTy.getElementType().isF32()) {
      if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
        return DPASEngineType::FP32_FP32_FP16_FP16;
      if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
        return DPASEngineType::FP32_FP32_BF16_BF16;
      if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
          op.getInputPrecision() == InputPrecision::TF32)
        return DPASEngineType::FP32_FP32_TF32_TF32;
    } else if (dTy.getElementType().isF16()) {
      if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
        return DPASEngineType::FP16_FP16_FP16_FP16;
    } else if (dTy.getElementType().isBF16()) {
      if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
        return DPASEngineType::BF16_BF16_BF16_BF16;
    }
  }

  return DPASEngineType::NOT_APPLICABLE;
}

std::optional<Attribute> inferSrcEncoding(Operation *op, Attribute encoding) {
  if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op))
    return encoding;
  if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op))
    return encoding;

  return mlir::inferSrcEncoding(op, encoding);
}

bool isExpensiveLoadOrStore(Operation *op) {
  assert((isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) &&
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
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
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

// Check if the convert will be a no-op in codegen.
static bool isFreeConvert(Operation *op) {
  if (auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op))
    return isMmaToMmaShortcut(convertOp.getSrc().getType(),
                              convertOp.getType());
  return false;
}

LogicalResult
getConvertBackwardSlice(Value root, SetVector<Value> &slice,
                        Attribute rootEncoding,
                        DenseMap<Value, Attribute> &layout,
                        std::function<bool(Operation *)> stopPropagation) {
  DenseSet<Value> visited;
  SmallVector<std::pair<Value, Attribute>> queue = {{root, rootEncoding}};
  while (!queue.empty()) {
    auto [currentValue, encoding] = queue.back();
    queue.pop_back();
    if (!visited.insert(currentValue).second)
      continue;
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
      unsigned argIdx = cast<OpResult>(currentValue).getResultNumber();

      auto thenValue = ifOp.thenYield().getOperand(argIdx);
      auto elseValue = ifOp.elseYield().getOperand(argIdx);

      queue.push_back({thenValue, encoding});
      queue.push_back({elseValue, encoding});

      continue;
    }
    if (auto *definingOp = currentValue.getDefiningOp()) {
      // If the op has multiple results we need to update all results layout.
      for (Value result : definingOp->getResults()) {
        if (result == currentValue ||
            !isTensorOrTensorPointerType(result.getType()))
          continue;
        if (layout.find(result) != layout.end()) {
          if (layout[result] != encoding)
            return failure();
          continue;
        }
        layout[result] = encoding;
      }
      if (!isFreeConvert(definingOp) &&
          canFoldIntoConversion(definingOp, encoding))
        continue;
      if (stopPropagation && stopPropagation(definingOp))
        continue;
      if (isa<triton::CatOp>(definingOp))
        return failure();
      for (Value operand : definingOp->getOperands()) {
        auto srcEncoding = ttgi::inferSrcEncoding(definingOp, encoding);
        if (!srcEncoding)
          return failure();
        if (slice.count(operand) == 0)
          queue.push_back({operand, *srcEncoding});
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
      queue.push_back({initOperand->get(), encoding});
      queue.push_back({yieldOperand, encoding});
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
