#ifndef TRITON_INTEL_ANALYSIS_UTILITY_H
#define TRITON_INTEL_ANALYSIS_UTILITY_H

#include "triton/Analysis/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

bool isDpasToDotShortcut(RankedTensorType srcTy, RankedTensorType dstTy);

}
} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_INTEL_ANALYSIS_UTILITY_H
