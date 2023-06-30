#ifndef TRITON_CONVERSION_TRITONGPU_TO_SPIRV_DOT_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_SPIRV_DOT_OP_H

#include "TritonGPUToSPIRVBase.h"
#include "triton/Analysis/Membar.h"

using namespace mlir;
using namespace mlir::triton;

void populateDotOpToSPIRVPatterns(TritonGPUToSPIRVTypeConverter &typeConverter,
                                  mlir::MLIRContext *context,
                                  RewritePatternSet &patterns, int numWarps,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  ModuleAllocation &allocation,
                                  PatternBenefit benefit);

#endif
