#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELSPILLTENSORTOSLM
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace ttgi = mlir::triton::gpu::intel;
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

class TritonIntelSpillTensorPass
    : public intel::impl::TritonIntelSpillTensorToSLMBase<
          TritonIntelSpillTensorPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](mlir::scf::ForOp forOp) -> void {
      // We cannot use forOp.walk(...) here because we only want to visit the
      // dot op in the loop body block. Nested blocks are handled separately.
      llvm::SmallVector<triton::DotOp> dotOps;
      for (Operation &opInFor : forOp) {
        if (auto dotOp = dyn_cast<triton::DotOp>(opInFor)) {
          auto A = dotOp.getA();
          if (auto *definingOp = A.getDefiningOp()) {
            auto parentOp = definingOp->getParentOp();
            if (parentOp != forOp) {
              dotOps.push_back(dotOp);
            }
          }
        }
      }

      for (auto dotOp : dotOps) {
        auto A = dotOp.getA();
        OpBuilder builder(A.getContext());
        builder.setInsertionPointAfter(A.getDefiningOp());
        auto srcType = cast<RankedTensorType>(A.getType());
        auto dstDotOp = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
            srcType.getEncoding());

        auto srcOrder = triton::gpu::getOrder(dstDotOp);
        auto rank = srcOrder.size();
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

        ArrayRef<int64_t> shape = srcType.getShape();
        triton::gpu::BlockedEncodingAttr blockEncoding =
            getDefaultBlockedEncoding(
                srcType.getContext(), shape,
                triton::gpu::TritonGPUDialect::getNumWarps(mod),
                triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod),
                triton::gpu::TritonGPUDialect::getNumCTAs(mod));
        auto cvtType = RankedTensorType::get(
            srcType.getShape(), srcType.getElementType(), blockEncoding);
        auto cvtOp = builder.create<triton::gpu::ConvertLayoutOp>(
            A.getDefiningOp()->getLoc(), cvtType, A);

        auto sharedMemorySpace =
            triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
        auto tmpType = triton::MemDescType::get(
            srcType.getShape(), srcType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(), sharedOrder,
                triton::gpu::getCTALayout(dstDotOp), srcType.getElementType()),
            sharedMemorySpace);
        auto tmp = builder.create<triton::gpu::LocalAllocOp>(
            A.getDefiningOp()->getLoc(), tmpType, cvtOp);
        auto newConvert = builder.create<triton::gpu::LocalLoadOp>(
            A.getDefiningOp()->getLoc(), srcType, tmp);
        dotOp.setOperand(0, newConvert);
      }
    });
  }
};

} // namespace
