#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCE_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCE_OP_H

#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

namespace AMD{
void populateReduceOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    int computeCapability, PatternBenefit benefit);
}

#endif
