//===- TritonGENToSPIRVPass.cpp - TritonGEN to SPIRV dialect conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/TritonGENToSPIRV/TritonGENToSPIRVPass.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONGENTOSPIRV
#include "intel/include/TritonGENToSPIRV/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;

namespace {

static spirv::Scope getSpirvScope(TritonGEN::MemScope scope) {
  switch (scope) {
  case TritonGEN::MemScope::WORK_GROUP:
    return spirv::Scope::Workgroup;
  case TritonGEN::MemScope::SUB_GROUP:
    return spirv::Scope::Subgroup;
  default:
    llvm_unreachable("unexpected MemScope value");
  }
}

struct TritonGENBarrierLowering
    : public OpConversionPattern<TritonGEN::BarrierOp> {
  using OpConversionPattern<TritonGEN::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = getContext();
    auto scope = spirv::Scope::Workgroup;
    spirv::MemorySemantics memorySemantics;
    switch (op.getMemFence()) {
    case TritonGEN::MemFence::LOCAL:
      memorySemantics = spirv::MemorySemantics::AcquireRelease |
                        spirv::MemorySemantics::WorkgroupMemory;
      break;
    case TritonGEN::MemFence::GLOBAL:
      memorySemantics = spirv::MemorySemantics::SequentiallyConsistent |
                        spirv::MemorySemantics::CrossWorkgroupMemory;
      break;
    case TritonGEN::MemFence::LOCAL_AND_GLOBAL:
      memorySemantics = spirv::MemorySemantics::SequentiallyConsistent |
                        spirv::MemorySemantics::WorkgroupMemory |
                        spirv::MemorySemantics::CrossWorkgroupMemory;
      break;
    default:
      llvm_unreachable("unexpected MemFence value");
    };
    rewriter.replaceOpWithNewOp<spirv::ControlBarrierOp>(op, scope, scope,
                                                         memorySemantics);
    return success();
  }
};

struct TritonGENSplitBarrierArriveLowering
    : public OpConversionPattern<TritonGEN::SplitBarrierArriveOp> {
  using OpConversionPattern<
      TritonGEN::SplitBarrierArriveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SplitBarrierArriveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::INTELControlBarrierArriveOp>(
        op, getSpirvScope(op.getExecutionScope()),
        getSpirvScope(op.getMemoryScope()), spirv::MemorySemantics::None);
    return success();
  }
};

struct TritonGENSplitBarrierWaitLowering
    : public OpConversionPattern<TritonGEN::SplitBarrierWaitOp> {
  using OpConversionPattern<TritonGEN::SplitBarrierWaitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SplitBarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::INTELControlBarrierWaitOp>(
        op, getSpirvScope(op.getExecutionScope()),
        getSpirvScope(op.getMemoryScope()), spirv::MemorySemantics::None);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

class GENToSPIRVConversionTarget : public ConversionTarget {
public:
  explicit GENToSPIRVConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<spirv::SPIRVDialect>();
    addIllegalOp<TritonGEN::BarrierOp>();
  }
};

struct ConvertTritonGENToSPIRV
    : public triton::impl::ConvertTritonGENToSPIRVBase<
          ConvertTritonGENToSPIRV> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet pattern(ctx);
    GENToSPIRVConversionTarget target(*ctx);

    populateTritonGENToSPIRVConversionPatterns(pattern);

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(pattern))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::triton::populateTritonGENToSPIRVConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TritonGENBarrierLowering, TritonGENSplitBarrierArriveLowering,
               TritonGENSplitBarrierWaitLowering>(patterns.getContext());
}
