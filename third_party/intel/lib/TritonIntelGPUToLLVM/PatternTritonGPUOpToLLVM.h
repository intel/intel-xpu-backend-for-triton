#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "TritonGPUToLLVMBase.h"

namespace mlir {
namespace triton {

void populateBarrierOpToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateClusterOpsToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateDotOpToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateElementwiseOpToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    int computeCapability, Target target, PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateLoadStoreOpToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    Target target, PatternBenefit benefit);

void populateReduceOpToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, int computeCapability, Target target,
    PatternBenefit benefit);
void populateScanOpToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateTensorPtrOpsToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateTritonGPUToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateViewOpToLLVMPatterns(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populatePrintOpToLLVMPattern(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateAssertOpToLLVMPattern(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateMemoryOpToLLVMPattern(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateControlFlowOpToLLVMPattern(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateMakeRangeOpToLLVMPattern(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(
    intel::TritonGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

} // namespace triton
} // namespace mlir

#endif
