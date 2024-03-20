#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "TargetInfo.h"
#include "TritonGPUToLLVMBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {
namespace intel {

void populateBarrierOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

void populateClusterOpsToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

void populateDotOpToLLVMPatterns(TritonGPUToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

void populateElementwiseOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

void populateLoadStoreOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit);

void populateReduceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit);
void populateScanOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

void populateTensorPtrOpsToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

void populateTritonGPUToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

void populatePrintOpToLLVMPattern(TritonGPUToLLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

void populateAssertOpToLLVMPattern(TritonGPUToLLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);

void populateMemoryOpToLLVMPattern(TritonGPUToLLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);

void populateControlFlowOpToLLVMPattern(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

void populateMakeRangeOpToLLVMPattern(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(TritonGPUToLLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

} // namespace intel
} // namespace triton
} // namespace mlir

#endif
