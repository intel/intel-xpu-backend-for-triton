#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "TargetInfo.h"
#include "TritonGPUToLLVMBase.h"
#include "intel/include/Analysis/AxisInfo.h"
#include "intel/include/TritonIntelGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir::triton::intel {

/* Specialized common patterns start */

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       const TargetInfoBase &targetInfo,
                                       PatternBenefit benefit);

void populateReduceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfo &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

void populateControlFlowOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const TargetInfoBase &targetInfo,
                                        PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit);

void populatePrintOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  const TargetInfoBase &targetInfo,
                                  PatternBenefit benefit);

/* Specialized common patterns end */

/* Third party patterns start */

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

void populateFp4ToFpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);

void populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, const ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit);

void populateTensorPtrOpsToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

/* Third party patterns end */

} // namespace mlir::triton::intel

#endif
