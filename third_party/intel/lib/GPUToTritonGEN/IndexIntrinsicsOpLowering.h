//===- IndexIntrinsicsOpLowering.h - GPU IndexOps Lowering class *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_GPUTOGEN_INDEXINTRINSICSOPLOWERING_H
#define TRITON_CONVERSION_GPUTOGEN_INDEXINTRINSICSOPLOWERING_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {

template <typename SourceOp, typename TargetOp>
class SingleDimLaunchConfigLowering : public ConvertOpToLLVMPattern<SourceOp> {
private:
  unsigned indexBitwidth;

public:
  explicit SingleDimLaunchConfigLowering(LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    const unsigned resBitWidth = 32;
    Operation *newOp =
        rewriter.create<TargetOp>(loc, IntegerType::get(context, resBitWidth));

    if (indexBitwidth > resBitWidth) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    } else if (indexBitwidth < resBitWidth) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp->getResult(0));
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace mlir

#endif // TRITON_CONVERSION_GPUTOGEN_INDEXINTRINSICSOPLOWERING_H
