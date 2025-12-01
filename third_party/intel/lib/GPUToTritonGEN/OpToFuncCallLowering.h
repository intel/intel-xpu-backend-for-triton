//===- OpToFuncCallLowering.h - GPU ops lowering to custom calls *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_GPUTOGEN_OPTOFUNCCALLLOWERING_H
#define TRITON_CONVERSION_GPUTOGEN_OPTOFUNCCALLLOWERING_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"

namespace mlir {

/// Rewriting that replace SourceOp with a CallOp to `f32Func` or `f64Func`
/// depending on the element type that Op operates upon. The function
/// declaration is added in case it was not added before.
///
/// If the input values are of f16 type, the value is first casted to f32, the
/// function called and then the result casted back.
///
/// Example:
///   %exp_f32 = math.exp %arg_f32 : f32
///
/// will be transformed into a call to a vendor specific math library.
template <typename SourceOp>
struct OpToFuncCallLowering : public ConvertOpToLLVMPattern<SourceOp> {
public:
  explicit OpToFuncCallLowering(LLVMTypeConverter &lowering, StringRef f32Func,
                                StringRef f64Func)
      : ConvertOpToLLVMPattern<SourceOp>(lowering), f32Func(f32Func),
        f64Func(f64Func) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    static_assert(std::is_base_of<OpTrait::SameOperandsAndResultType<SourceOp>,
                                  SourceOp>::value,
                  "expected op with same operand and result types");

    SmallVector<Value, 1> castedOperands;
    for (Value operand : adaptor.getOperands())
      castedOperands.push_back(maybeCast(operand, rewriter));

    SmallVector<Type> parameters(ValueRange(castedOperands).getTypes());
    Type resultType = parameters.front();
    StringRef funcName = getFunctionName(resultType);
    if (funcName.empty())
      return failure();

    auto moduleOp = op->template getParentWithTrait<OpTrait::SymbolTable>();
    auto funcOp = triton::gpu::intel::lookupOrCreateSPIRVFn(
        moduleOp, funcName, parameters, resultType);
    auto callOp = triton::gpu::intel::createSPIRVBuiltinCall(
        op->getLoc(), rewriter, funcOp, castedOperands);

    if (resultType == adaptor.getOperands().front().getType()) {
      rewriter.replaceOp(op, {callOp.getResult()});
      return success();
    }

    Value truncated = LLVM::FPTruncOp::create(
        rewriter, op->getLoc(), adaptor.getOperands().front().getType(),
        callOp.getResult());
    rewriter.replaceOp(op, {truncated});
    return success();
  }

private:
  Value maybeCast(Value operand, PatternRewriter &rewriter) const {
    Type type = operand.getType();
    if (!isa<Float16Type>(type))
      return operand;

    return LLVM::FPExtOp::create(rewriter, operand.getLoc(),
                                 Float32Type::get(rewriter.getContext()),
                                 operand);
  }

  StringRef getFunctionName(Type type) const {
    if (isa<Float32Type>(type))
      return f32Func;
    if (isa<Float64Type>(type))
      return f64Func;
    return "";
  }

  const std::string f32Func;
  const std::string f64Func;
};

} // namespace mlir

#endif // TRITON_CONVERSION_GPUTOGEN_OPTOFUNCCALLLOWERING_H
