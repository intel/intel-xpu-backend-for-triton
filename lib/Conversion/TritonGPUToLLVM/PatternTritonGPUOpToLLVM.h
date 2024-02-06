#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Target/PTX/TmaMetadata.h"

typedef llvm::DenseMap<mlir::Operation *, mlir::triton::MakeTensorPtrOp>
    TensorPtrMapT;

namespace mlir {
namespace triton {

<<<<<<< HEAD
void populateBarrierOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
    PatternBenefit benefit);

void populateClusterOpsToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
    PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
    PatternBenefit benefit);
=======
void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns, int numWarps,
                                     ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                     PatternBenefit benefit);

void populateClusterOpsToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns, int numWarps,
                                      ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                      PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit);
>>>>>>> 2dd9d74527f431e5e822b8e67c01900e4d0bfef3

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns, int numWarps,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 Target target, PatternBenefit benefit);

void populateElementwiseOpToLLVMPatterns(
<<<<<<< HEAD
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    int computeCapability, Target target, PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
=======
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
>>>>>>> 2dd9d74527f431e5e822b8e67c01900e4d0bfef3
    PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       int numWarps,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);

void populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis,
    mlir::triton::gpu::TMAMetadataTy *tmaMetadata,
    const TensorPtrMapT *tensorPtrMap, Target target, PatternBenefit benefit);

void populateReduceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns, int numWarps,
                                    ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                    int computeCapability, Target target,
                                    PatternBenefit benefit);

void populateRegReallocOpToLLVMPatterns(
<<<<<<< HEAD
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
    PatternBenefit benefit);
=======
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit);
>>>>>>> 2dd9d74527f431e5e822b8e67c01900e4d0bfef3

void populateScanOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns, int numWarps,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  Target target, PatternBenefit benefit);

void populateTensorPtrOpsToLLVMPatterns(
<<<<<<< HEAD
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
    PatternBenefit benefit);

void populateTritonGPUToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
    PatternBenefit benefit);
=======
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit);

void populateTritonGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns, int numWarps,
                                     ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                     PatternBenefit benefit);
>>>>>>> 2dd9d74527f431e5e822b8e67c01900e4d0bfef3

void populateViewOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns, int numWarps,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  Target target, PatternBenefit benefit);

} // namespace triton
} // namespace mlir

#endif
