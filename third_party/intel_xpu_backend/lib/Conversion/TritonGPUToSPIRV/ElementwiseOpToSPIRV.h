#ifndef TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_SPIRV_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_SPIRV_OP_H

#include "TritonGPUToSPIRVBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateElementwiseOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation *allocation,
    Value smem, PatternBenefit benefit, bool supportBFConvOp);

#endif
