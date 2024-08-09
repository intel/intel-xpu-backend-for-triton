#include "triton/Analysis/Utility.h"
#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace mlir::triton::gpu::intel {

bool isDpasToDotShortcut(RankedTensorType dpasTy, RankedTensorType dotTy) {
  auto dpasLayout = dyn_cast<DpasEncodingAttr>(dpasTy.getEncoding());
  auto dotOperandLayout = dyn_cast<DotOperandEncodingAttr>(dotTy.getEncoding());
  // dpas -> dot_operand conversion when:
  if (dpasLayout && dotOperandLayout &&
      dotOperandLayout.getParent() == dpasLayout) {
    SmallVector<unsigned> shapeC = dpasLayout.getDPASInstShapeC();
    SmallVector<unsigned> shapeA = dpasLayout.getDPASInstShapeA();
    if (dotOperandLayout.getOpIdx() == 0 && /* A operands. */
        dpasLayout.getWarpsPerCTA().back() ==
            1 && /* The warpsPerCTA is [..., 1]. */
        shapeA[0] == shapeC[0] &&
        shapeA[1] == shapeC[1] /* C shape is equal to A shape */
    )
      return true;
  }

  return false;
}

bool cvtNeedsSharedMemory(RankedTensorType srcTy, RankedTensorType dstTy) {
  MLIRContext *ctx = srcTy.getContext();
  std::optional<LinearLayout> srcLayout =
      triton::gpu::intel::toLinearLayout(srcTy.getShape(), srcTy.getEncoding());
  std::optional<LinearLayout> dstLayout =
      triton::gpu::intel::toLinearLayout(dstTy.getShape(), dstTy.getEncoding());
  if (srcLayout.has_value() && dstLayout.has_value()) {
    // comp describes the layout function for converting from src to dst.
    LinearLayout comp = srcLayout->invertAndCompose(*dstLayout);
    StringAttr kLane = StringAttr::get(ctx, "lane");
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    StringAttr kBlock = StringAttr::get(ctx, "block");
    // In principle, there's no need for shared memory if there's no
    // communication between warps.  However, right now we only have implemented
    // the shortcut case where there's no communication between *threads*.
    //
    // TODO(jlebar): Remove the kLane layout once we add support for
    // shuffle-based layout conversions in ConvertLayoutToLLVM.
    if (comp.divideRight(LinearLayout::identity1D(comp.getInDimSize(kLane),
                                                  kLane, kLane) *
                         LinearLayout::identity1D(comp.getInDimSize(kWarp),
                                                  kWarp, kWarp) *
                         LinearLayout::identity1D(comp.getInDimSize(kBlock),
                                                  kBlock, kBlock))
            .has_value()) {
      return false;
    }
  }

  // TODO(jlebar): Remove these special cases once they're fully subsumed by the
  // linear-layout check above.
  return !triton::gpu::intel::isDpasToDotShortcut(srcTy, dstTy) &&
         !isMmaToMmaShortcut(srcTy, dstTy) &&
         !isMmaToDotShortcut(srcTy, dstTy) &&
         !isMfmaToDotShortcut(srcTy, dstTy);
}

} // namespace mlir::triton::gpu::intel
