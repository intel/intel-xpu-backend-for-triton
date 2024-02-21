#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "TritonGPUToLLVMBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {
namespace intel {

void populateBarrierOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit);

void populateClusterOpsToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit);

void populateDotOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, Target target,
                                 PatternBenefit benefit);

void populateElementwiseOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    Target target, PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit);

void populateLoadStoreOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
    PatternBenefit benefit);

void populateReduceOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    int computeCapability, Target target,
                                    PatternBenefit benefit);
void populateScanOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns, Target target,
                                  PatternBenefit benefit);

void populateTensorPtrOpsToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit);

void populateTritonGPUToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit);

void populateViewOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns, Target target,
                                  PatternBenefit benefit);

void populatePrintOpToLLVMPattern(TritonGPUToLLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns, Target target,
                                  PatternBenefit benefit);

void populateAssertOpToLLVMPattern(TritonGPUToLLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns, Target target,
                                   PatternBenefit benefit);

void populateMemoryOpToLLVMPattern(TritonGPUToLLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns, Target target,
                                   PatternBenefit benefit);

void populateControlFlowOpToLLVMPattern(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit);

void populateMakeRangeOpToLLVMPattern(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(TritonGPUToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, Target target,
                                 PatternBenefit benefit);

} // namespace intel
} // namespace triton
} // namespace mlir

#endif
