//===- PipelineManager.h - TritonIntelGPU pipeline manager ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pipeline manager for the TritonIntelGPU -> LLVM pass.
//
//===----------------------------------------------------------------------===//

#ifndef TRITONINTELGPUTOLLVM_PIPELINEMANAGER_H
#define TRITONINTELGPUTOLLVM_PIPELINEMANAGER_H

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/PatternMatch.h"

#include "intel/include/GPUToTritonGEN/GPUToTritonGENPass.h"
#include "intel/include/TritonGENToLLVM/TritonGENToLLVMPass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "PatternTritonGPUOpToLLVM.h"

namespace mlir {

FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

namespace mlir::triton::intel {

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), numWarps(numWarps) {}

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {

    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  triton::FuncOp amendFuncOp(triton::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const {
    // Push back a variable that indicates the current stack pointer of shared
    // memory to the function arguments.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    // 1. Modify the function type to add the new argument.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    amendedInputTy.push_back(ptrTy);
    auto amendedFuncTy = FunctionType::get(funcTy.getContext(), amendedInputTy,
                                           funcTy.getResults());
    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    auto amendedArgAttrs = llvm::to_vector<4>(funcOp.getAllArgAttrs());
    amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    amendedAttrs.push_back(rewriter.getNamedAttr(
        funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(amendedArgAttrs)));
    // 3. Add a new argument to the region
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();
    region.addArgument(ptrTy, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = funcOp;
    if (!LLVM::isKernel(funcOp))
      amendedFuncOp = amendFuncOp(funcOp, rewriter);

    LLVM::LLVMFuncOp newFuncOp = *mlir::convertFuncOpToLLVMFuncOp(
        amendedFuncOp, rewriter, *getTypeConverter());
    if (!newFuncOp)
      return failure();

    MLIRContext *ctx = funcOp->getContext();
    auto mod = funcOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (LLVM::isKernel(funcOp)) {
      newFuncOp.setCConv(LLVM::CConv::SPIR_KERNEL);
      newFuncOp.setLinkage(LLVM::Linkage::External);
    }

    NamedAttrList attrs;
    attrs.append(TritonGEN::TritonGENDialect::getMaxWorkGroupSizeAttrName(),
                 rewriter.getI32ArrayAttr({threadsPerWarp * numWarps, 1, 1}));
    attrs.append(TritonGEN::TritonGENDialect::getReqdSubGroupSizeAttrName(),
                 rewriter.getI32ArrayAttr({threadsPerWarp}));
    newFuncOp->setDialectAttrs(attrs);

    if (!LLVM::isKernel(funcOp)) {
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
      rewriter.eraseOp(amendedFuncOp);
    }

    // required by AxisInfoAnalysis
    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
};

struct AddSPIRVEnvPattern : public mlir::OpRewritePattern<ModuleOp> {

  using mlir::OpRewritePattern<ModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModuleOp op,
                                PatternRewriter &rewriter) const override {
    if (spirv::lookupTargetEnv(op)) {
      return failure();
    }

    int subgroupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(op);

    auto resourceLimit = spirv::getDefaultResourceLimits(rewriter.getContext());
    auto newResourceLimit = rewriter.getAttr<spirv::ResourceLimitsAttr>(
        resourceLimit.getMaxComputeSharedMemorySize(),
        resourceLimit.getMaxComputeWorkgroupInvocations(),
        resourceLimit.getMaxComputeWorkgroupSize(), subgroupSize,
        resourceLimit.getMinSubgroupSize(), resourceLimit.getMaxSubgroupSize(),
        resourceLimit.getCooperativeMatrixPropertiesKhr(),
        resourceLimit.getCooperativeMatrixPropertiesNv());
    auto triple = spirv::VerCapExtAttr::get(
        spirv::Version::V_1_2,
        {spirv::Capability::GroupNonUniform, spirv::Capability::Addresses,
         spirv::Capability::Float16Buffer, spirv::Capability::Int64,
         spirv::Capability::Int16, spirv::Capability::Int8,
         spirv::Capability::Kernel, spirv::Capability::Linkage,
         spirv::Capability::Vector16, spirv::Capability::GenericPointer,
         spirv::Capability::Groups, spirv::Capability::Float64},
        {}, rewriter.getContext());
    auto newTargetEnv = spirv::TargetEnvAttr::get(triple, newResourceLimit);
    rewriter.modifyOpInPlace(op, [op, newTargetEnv] {
      op->setAttr(spirv::getTargetEnvAttrName(), newTargetEnv);
    });
    return success();
  }
};

/// Manages TritonIntelGPU --> LLVM the conversion pipeline.
/// Currently the conversion pipeline depends on whether the kernel contains
/// block pointers or not.
class TritonGPUToLLVMPipelineManager {
public:
  TritonGPUToLLVMPipelineManager(ModuleOp &mod, MLIRContext *ctx)
      : mod(mod), ctx(ctx),
        isAdvancedPathEnabled(
            mod->hasAttr(gpu::intel::TritonIntelGPUDialect::
                             getSupportSG2DBlockAttrName()) &&
            mod->hasAttr(
                gpu::intel::TritonIntelGPUDialect::getSupportDPASAttrName()) &&
            mlir::triton::tools::getBoolEnv("TRITON_INTEL_ADVANCED_PATH")) {}

  /// FIXME: remove once the block ptr conversion path is capable of handling
  ///        shared memory.
  bool skipSharedMemoryAllocation() const { return isAdvancedPathEnabled; }

  /// Populate the conversion pipeline for function operations.
  void populateFunctionConversionPatterns(
      RewritePatternSet &funcPatterns,
      TritonIntelGPUToLLVMTypeConverter &typeConverter, int numWarps) const {
    funcPatterns.add<FuncOpConversion>(typeConverter, numWarps,
                                       /*benefit=*/1);
    if (!isAdvancedPathEnabled)
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
  }

  /// Populate the conversion pipeline for various operations.
  void
  populateConversionPatterns(RewritePatternSet &patterns,
                             ModuleAxisInfoAnalysis &axisInfoAnalysis,
                             TritonIntelGPUToLLVMTypeConverter &typeConverter,
                             TargetInfo &targetInfo, int benefit) const {
    using namespace mlir;
    using namespace mlir::triton;

    // should run before other patterns that need the SPIRV-ENV attr
    // (e.g. patterns that output triton_gen.sub_group_reduce)
    patterns.add<AddSPIRVEnvPattern>(&typeConverter.getContext(),
                                     patternBenefitAddSPIRVEnv);

    if (isAdvancedPathEnabled) {
      intel::populateArithOpsToLLVMPatterns(typeConverter, patterns, benefit);
      intel::populateBF16CastsLLVMPatterns(typeConverter, patterns, benefit);
      intel::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                benefit);
      intel::populateTritonOpsToLLVMPatterns(typeConverter, patterns, benefit);
    } else {
      intel::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                   patterns, benefit);
      intel::populateDotOpToLLVMPatterns(typeConverter, patterns, benefit);
      intel::populateElementwiseOpToLLVMPatterns(
          typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
      intel::populateLoadStoreOpToLLVMPatterns(
          typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);
      intel::populateReduceOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                            benefit);
      intel::populateScanOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                          benefit);
      intel::populateViewOpToLLVMPatterns(typeConverter, patterns, benefit);

      intel::populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns,
                                                benefit);
      intel::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
      intel::populatePrintOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                          benefit);
      populateAssertOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                    benefit);
      intel::populateMemoryOpToLLVMPattern(typeConverter, targetInfo, patterns,
                                           benefit);
      intel::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                benefit);
      intel::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                              patterns, benefit);
    }

    intel::populateSPMDOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                       benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    triton::populateTritonGENToLLVMConversionPatterns(typeConverter, patterns);
    triton::populateGPUToTritonGENConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  }

private:
  ModuleOp &mod;
  MLIRContext *ctx;

  /// Selects which conversion pipeline to use.
  /// FIXME: this is temporary and should be removed once we have an analysis to
  /// determine whether a kernel uses block pointers.
  bool isAdvancedPathEnabled = false;
};

} // namespace mlir::triton::intel

#endif // TRITONINTELGPUTOLLVM_PIPELINEMANAGER_H
