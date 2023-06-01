#ifndef TRITON_TRITONGPUTOSPIRV_H
#define TRITON_TRITONGPUTOSPIRV_H

#include "TritonGPUToSPIRVBase.h"

void populateTritonGPUToSPIRVPatterns(
        TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
        mlir::RewritePatternSet &patterns, int numWarps,
        mlir::ModuleAxisInfoAnalysis &axisInfoAnalysis,
        mlir::ModuleAllocation &allocation,
        ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
        mlir::PatternBenefit benefit);

#endif
