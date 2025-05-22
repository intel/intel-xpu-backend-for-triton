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
#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/IR/PatternMatch.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Utils.h"
#include "intel/include/GPUToTritonGEN/GPUToTritonGENPass.h"
#include "intel/include/TritonGENToLLVM/TritonGENToLLVMPass.h"
#include "intel/include/TritonGENToSPIRV/TritonGENToSPIRVPass.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "third_party/proton/dialect/include/TritonProtonToLLVM/PatternTritonProtonOpToLLVM.h"

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
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo),
        numWarps(numWarps) {}

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
                             ConversionPatternRewriter &rewriter,
                             const TargetInfoBase &targetInfo) const {
    // Push back two new arguments that indicate the current pointer to shared
    // memory and global scratch memory.
    Location loc = funcOp.getLoc();
    MLIRContext *ctx = funcOp->getContext();
    auto sharedPtrTy =
        LLVM::LLVMPointerType::get(ctx, targetInfo.getSharedAddressSpace());
    Type globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);

    // 1. Modify the function type to add the new arguments.
    FunctionType funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    bool isKernel = LLVM::isKernel(funcOp);
    if (!isKernel)
      amendedInputTy.push_back(sharedPtrTy);

    amendedInputTy.push_back(globalPtrTy);
    auto amendedFuncTy =
        FunctionType::get(ctx, amendedInputTy, funcTy.getResults());
    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    if (auto argAttrs = funcOp.getAllArgAttrs()) {
      SmallVector<mlir::Attribute> amendedArgAttrs(argAttrs.begin(),
                                                   argAttrs.end());
      while (amendedArgAttrs.size() < amendedInputTy.size())
        amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));

      amendedAttrs.push_back(
          rewriter.getNamedAttr(funcOp.getArgAttrsAttrName(),
                                rewriter.getArrayAttr(amendedArgAttrs)));
    }

    // 3. Add the new arguments to the region
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    Region &region = funcOp.getBody();
    if (!isKernel)
      region.addArgument(sharedPtrTy, loc);

    region.addArgument(globalPtrTy, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = amendFuncOp(funcOp, rewriter, targetInfo);

    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp)) {
      return failure();
    }

    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;

    MLIRContext *ctx = funcOp->getContext();
    auto mod = funcOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (LLVM::isKernel(funcOp)) {
      newFuncOp.setCConv(LLVM::CConv::SPIR_KERNEL);
      newFuncOp.setLinkage(LLVM::Linkage::External);
    } else {
      newFuncOp.setCConv(LLVM::CConv::SPIR_FUNC);
    }

    newFuncOp->setAttr(
        TritonGEN::TritonGENDialect::getMaxWorkGroupSizeAttrName(),
        rewriter.getDenseI32ArrayAttr({threadsPerWarp * numWarps, 1, 1}));
    newFuncOp.setIntelReqdSubGroupSize(threadsPerWarp);

    if (!LLVM::isKernel(funcOp)) {
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
    }

    // required by AxisInfoAnalysis
    rewriter.eraseOp(funcOp);
    rewriter.eraseOp(amendedFuncOp);
    return success();
  }

private:
  int numWarps{0};
  const TargetInfoBase &targetInfo;
};

/// Manages TritonIntelGPU --> LLVM the conversion pipeline.
/// Currently the conversion pipeline depends on whether the kernel contains
/// block pointers or not.
class TritonGPUToLLVMPipelineManager {
public:
  TritonGPUToLLVMPipelineManager(ModuleOp &mod, MLIRContext *ctx, bool advanced,
                                 bool oneMatrixPerLoadForBT,
                                 bool useTileLoadLinearLayout)
      : mod(mod), ctx(ctx), isAdvancedPathEnabled(advanced),
        oneMatrixPerLoadForBT(oneMatrixPerLoadForBT),
        useTileLoadLinearLayout(useTileLoadLinearLayout) {}

  /// FIXME: remove once the block ptr conversion path is capable of handling
  ///        shared memory.
  bool skipSharedMemoryAllocation() const { return isAdvancedPathEnabled; }

  /// Populate the conversion pipeline for function operations.
  void populateFunctionConversionPatterns(
      RewritePatternSet &funcPatterns,
      TritonIntelGPUToLLVMTypeConverter &typeConverter, int numWarps,
      const TargetInfoBase &targetInfo) const {
    funcPatterns.add<FuncOpConversion>(typeConverter, numWarps, targetInfo,
                                       /*benefit=*/1);
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

    if (isAdvancedPathEnabled) {
      intel::populateArithOpsToLLVMPatterns(typeConverter, patterns, benefit);
      intel::populateBF16CastsLLVMPatterns(typeConverter, patterns, benefit);
      intel::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
      intel::populateTritonOpsToLLVMPatterns(typeConverter, patterns, benefit);
    } else {
      intel::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                   patterns, benefit);
      intel::populateDotOpToLLVMPatterns(typeConverter, patterns, benefit);
      intel::populateElementwiseOpToLLVMPatterns(
          typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
      intel::populateLoadStoreOpToLLVMPatterns(
          typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit,
          oneMatrixPerLoadForBT, useTileLoadLinearLayout);
      intel::populateReduceOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                            benefit);
      mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
      mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                   targetInfo, benefit);
      mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                                 benefit);

      intel::populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns,
                                                benefit);
      intel::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
      intel::populatePrintOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                          benefit);
      mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
      populateAssertOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                    benefit);
      mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                   patterns, benefit);
      mlir::triton::proton::populateRecordOpToLLVMPattern(
          typeConverter, patterns, targetInfo, benefit);
      intel::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
      mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                     patterns, benefit);
      intel::populateFp4ToFpToLLVMPatterns(typeConverter, patterns, benefit);
    }

    intel::populateSPMDOpToLLVMPattern(typeConverter, patterns, targetInfo,
                                       benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    triton::populateGPUToTritonGENConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateGpuToLLVMSPVConversionPatterns(typeConverter, patterns);
    populateSPIRVToLLVMConversionPatterns(typeConverter, patterns,
                                          spirv::ClientAPI::OpenCL);
  }

private:
  ModuleOp &mod;
  MLIRContext *ctx;

  /// Selects which conversion pipeline to use.
  /// FIXME: this is temporary and should be removed once we have an analysis to
  /// determine whether a kernel uses block pointers.
  bool isAdvancedPathEnabled = false;
  bool oneMatrixPerLoadForBT = false;
  bool useTileLoadLinearLayout = true;
};

} // namespace mlir::triton::intel

#endif // TRITONINTELGPUTOLLVM_PIPELINEMANAGER_H
