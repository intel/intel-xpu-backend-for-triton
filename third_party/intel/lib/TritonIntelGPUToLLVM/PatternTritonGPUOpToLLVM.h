#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "TritonGPUToLLVMBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {
namespace intel {

void populateBarrierOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateClusterOpsToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateDotOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateElementwiseOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    int computeCapability, Target target, PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateLoadStoreOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    Target target, PatternBenefit benefit);

void populateReduceOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, int computeCapability, Target target,
    PatternBenefit benefit);
void populateScanOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateTensorPtrOpsToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateTritonGPUToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populatePrintOpToLLVMPattern(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateAssertOpToLLVMPattern(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateMemoryOpToLLVMPattern(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateControlFlowOpToLLVMPattern(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateMakeRangeOpToLLVMPattern(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit);

} // namespace intel
} // namespace triton
} // namespace mlir

#endif
