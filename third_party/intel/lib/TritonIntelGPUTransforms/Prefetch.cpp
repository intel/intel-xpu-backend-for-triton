//===----------------------------------------------------------------------===//
//
// This pass tries to prefetch operands (a and b) of tt.dot.
// Those ConvertLayoutOps will be lowered to shared memory loads.
//
// For example:
// %a: tensor<128x32xf16, #enc>
// scf.for %iv = ... iter_args(%a_arg = %a, ...) {
//   %d = tt.dot %a_arg, %b, %c
//   ...
//   scf.yield %a_next, ...
// }
//
// will be translated to
//
// %a: tensor<128x32xf16, #enc>
// %a_tmp = tensor.subview %a[0, 0] [128, 16]
// %a_prefetch = ttg.local_load %a_tmp
// scf.for %iv = ... iter_args(%a_buf = %a, ..., %a_prefetch_arg = %a_prefetch)
// {
//   %x = tt.dot %a_prefetch_arg, %b, %c
//   %a_tmp_rem = tensor.subview %a_buf[0, 16] [128, 16]
//   %a_prefetch_next = ttg.local_load %a_tmp_rem
//   ...
//   scf.yield %next_a, ..., %a_prefetch_next
// }
//===----------------------------------------------------------------------===//

#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUPREFETCH
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-prefetch"

namespace {

/// Create a prefetch operation for the given load operation.
static void createPrefetchOp(tt::LoadOp loadOp, Value ptr) {
  Operation *defOp = loadOp->getOperand(0).getDefiningOp();
  OpBuilder builder(defOp);
  // TODO: Add prefetchOp after last dependency (between ptr and mask)
  builder.setInsertionPointAfter(defOp);
  auto prefetchOp = builder.create<ttgi::PrefetchOp>(
      loadOp->getLoc(), ptr, loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());

  // inherit attributes from the load operation
  auto attrs = loadOp->getAttrDictionary();
  prefetchOp->setAttrs(attrs);
}

static bool addPrefetch(scf::ForOp forOp) {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<RankedTensorType>(v.getType()).getEncoding();
  };

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // Only accepts dotOps encoded as DPAS MMA
      // TODO: Investigate if BlockPointer is a condition to efficient prefetch?
      auto dstDpasEnc = dyn_cast<ttg::intel::DpasEncodingAttr>(
          getEncoding(dotOp.getResult()));
      if (!dstDpasEnc)
        // Don't rewrite if any other type is found.
        return false;
      dotsInFor.push_back(dotOp);
    }

  if (dotsInFor.empty())
    return false;

  // returns source of cvt
  auto getPrefetchSrc = [](Value v) -> std::optional<triton::LoadOp> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundOutOfLoopLoad = false;
    SmallVector<Value> rets;
    rets.push_back(op->getResult(0));
    LLVM_DEBUG(llvm::dbgs() << "Prefetch src: " << *op);
    while (op) {
      if (op->getNumOperands() != 1)
        break;
      rets.push_back(op->getOperand(0));
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        Operation *parentOp = op->getBlock()->getParentOp();
        if (!isa<scf::ForOp>(parentOp))
          return loadOp;
        break;
      }
      op = op->getOperand(0).getDefiningOp();
    }
    return std::nullopt;
  };

  for (triton::DotOp dot : dotsInFor) {
    auto aVals = getPrefetchSrc(dot.getA());
    auto bVals = getPrefetchSrc(dot.getB());

    if (aVals) {
      tt::LoadOp loadOp = aVals.value();
      createPrefetchOp(loadOp, loadOp.getPtr());
    }
    if (bVals) {
      tt::LoadOp loadOp = bVals.value();
      createPrefetchOp(loadOp, loadOp.getPtr());
    }
  }

  return true;
}

class PrefetchPass
    : public triton::gpu::intel::impl::TritonIntelGPUPrefetchBase<
          PrefetchPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUPrefetchBase<
      PrefetchPass>::TritonIntelGPUPrefetchBase;

  void runOnOperation() override {

    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    getOperation()->walk([&](scf::ForOp forOp) {
      if (addPrefetch(forOp))
        return;
    });
  }
};

} // namespace
