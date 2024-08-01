#ifndef TRITON_INTEL_ANALYSIS_UTILITY_H
#define TRITON_INTEL_ANALYSIS_UTILITY_H

#include "triton/Analysis/Utility.h"

namespace mlir::triton::gpu::intel {

bool isDpasToDotShortcut(RankedTensorType dpasTy, RankedTensorType dotTy);

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_UTILITY_H
