#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUREDUCEDATADUPLICATION
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

/// Return the op after which the shared-memory staging buffer that replaces
/// \p cvtOp should be allocated, or nullptr to allocate it at \p cvtOp.
///
/// A conversion can sit in a loop while its source is defined outside it (an
/// earlier pass chose to keep the conversion in the loop, e.g. to limit the
/// dot-operand's register live range; only the local_load needs to stay
/// there). When that happens, the store into shared memory is loop
/// invariant. Allocating right after the source performs the store once and
/// ends the source's register live range early, without relying on
/// TritonGPUReorderInstructions to hoist the allocation later.
Operation *getLoopInvariantAllocAnchor(ConvertLayoutOp cvtOp) {
  auto loop = cvtOp->getParentOfType<LoopLikeOpInterface>();
  if (!loop)
    return nullptr;
  Operation *srcDef = cvtOp.getSrc().getDefiningOp();
  if (!srcDef || loop->isAncestor(srcDef))
    return nullptr;
  // Staging a scalar-derived value for the whole loop costs shared memory for
  // no benefit; TritonGPUReorderInstructions skips these for the same reason.
  if (isa<arith::ConstantOp, triton::SplatOp>(srcDef))
    return nullptr;
  return srcDef;
}

class TritonIntelGPUReduceDataDuplicationPass
    : public intel::impl::TritonIntelGPUReduceDataDuplicationBase<
          TritonIntelGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
      if (!cvtNeedsSharedMemory(cvtOp))
        return;
      auto srcOrder = triton::gpu::getOrder(srcType);
      auto rank = srcOrder.size(); // TODO: maybe we can use upstream code.
      if (auto srcDpasEncoding =
              dyn_cast<triton::gpu::intel::DpasEncodingAttr>(srcEncoding)) {
        auto opIdx =
            static_cast<intel::DpasEncodingAttr::OpIdx>(dstDotOp.getOpIdx());
        if ((opIdx == intel::DpasEncodingAttr::OpIdx::OperandA /* Operand A */
             && dstDotOp.getParent() == srcDpasEncoding &&
             srcDpasEncoding.getWarpsPerCTA()[rank - 1] ==
                 1 /* No parallel on N dim */) ||
            (opIdx == intel::DpasEncodingAttr::OpIdx::OperandB /* Operand B */
             && dstDotOp.getParent() == srcDpasEncoding &&
             srcDpasEncoding.getWarpsPerCTA()[rank - 2] ==
                 1 /* No parallel on M dim */))
          /* The destination dot layout has no duplication. */
          return;
      }
      SmallVector<unsigned> sharedOrder;
      if (rank == 3) {
        // add all elements except the element that is zero
        for (unsigned i = 0; i < rank; ++i)
          if (srcOrder[i] != 0)
            sharedOrder.emplace_back(srcOrder[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = std::move(srcOrder);
      }
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto tmpType = triton::gpu::MemDescType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SwizzledSharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(), sharedOrder,
              triton::gpu::getCGALayout(srcEncoding), srcType.getElementType()),
          sharedMemorySpace);
      if (Operation *allocAnchor = getLoopInvariantAllocAnchor(cvtOp))
        builder.setInsertionPointAfter(allocAnchor);
      auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(),
                                                   tmpType, cvtOp.getSrc());
      builder.setInsertionPoint(cvtOp);
      auto newConvert = triton::gpu::LocalLoadOp::create(
          builder, cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

} // namespace
