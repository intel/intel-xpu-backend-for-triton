#include "triton/Analysis/Utility.h"
#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

bool isDpasToDotShortcut(RankedTensorType srcTy, RankedTensorType dstTy) {
  auto dpasLayout = dyn_cast<DpasEncodingAttr>(srcTy.getEncoding());
  auto dotOperandLayout = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  // dpas -> dot_operand conversion when:
  if (dpasLayout && dotOperandLayout &&
      dotOperandLayout.getParent() == dpasLayout) {
    if (dpasLayout.getExecutionSize() == 16 &&
        dpasLayout.getSystolicDepth() == 8 &&
        dpasLayout.getOpsPerChannel() == 2 && /* PVC Half precision. */
        dotOperandLayout.getOpIdx() == 0 &&   /* A operands. */
        dpasLayout.getWarpsPerCTA().back() ==
            1 /* The warpsPerCTA is [..., 1]. */
    )
      return true;
  }

  return false;
}

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
