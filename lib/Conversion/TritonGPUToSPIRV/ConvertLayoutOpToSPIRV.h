#ifndef TRITON_CONVERSION_TRITONGPU_TO_SPIRV_CONVERT_LAYOUT_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_SPIRV_CONVERT_LAYOUT_OP_H

#include "TritonGPUToSPIRVBase.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;

void populateConvertLayoutOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);

#endif
