#ifndef TRITON_TRITONGPUTOVC_H
#define TRITON_TRITONGPUTOVC_H

#include "TritonGPUToSPIRVBase.h"

void populateTritonGPUToVCPatterns(TritonGPUToSPIRVTypeConverter &typeConverter,
                                   mlir::MLIRContext *context,
                                   mlir::RewritePatternSet &patterns,
                                   int numWarps, mlir::PatternBenefit benefit);

#endif

