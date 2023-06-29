#ifndef TRITON_VIEWOPTOSPIRV_H
#define TRITON_VIEWOPTOSPIRV_H

#include "TritonGPUToSPIRVBase.h"
#include "triton/Analysis/Membar.h"

using namespace mlir;
using namespace mlir::triton;

void populateViewOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    mlir::RewritePatternSet &patterns, int numWarps,
    mlir::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    mlir::ModuleAllocation *allocation, mlir::Value smem,
    mlir::PatternBenefit benefit);

#endif
