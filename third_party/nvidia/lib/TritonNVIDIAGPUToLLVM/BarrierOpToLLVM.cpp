/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct BarrierOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<mlir::gpu::BarrierOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      mlir::gpu::BarrierOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (op->hasAttr("bar_id")) {
      // llvm.nvvm.barrier0 doesn't support bar_id and num_threads attributes,
      // so we have to lower it to ptx manually.
      auto barId = op->getAttrOfType<IntegerAttr>("bar_id").getInt();
      auto numThreads = op->getAttrOfType<IntegerAttr>("num_threads").getInt();
      barSync(rewriter, op, barId, numThreads);
      rewriter.eraseOp(op);
      return success();
    }
    // Otherwise we let the default lowering handle it
    return failure();
  }
};

struct FenceAsyncSharedOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::FenceAsyncSharedOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::FenceAsyncSharedOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::FenceAsyncSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.replaceOpWithNewOp<triton::nvgpu::FenceAsyncSharedOp>(
        op, adaptor.getBCluster());
    return success();
  }
};
} // namespace

void mlir::triton::populateBarrierOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit) {
  patterns.add<BarrierOpConversion>(typeConverter, target, benefit);
  patterns.add<FenceAsyncSharedOpConversion>(typeConverter, target, benefit);
}
