//===----------------------------------------------------------------------===//
//
// This pass tries to reduce the liveness of variable.
// For e.g: by reducing the distance between load ops and
// the operation using the variable.
//===----------------------------------------------------------------------===//

#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "mlir/Analysis/Liveness.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>
#include <optional>

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUREDUCEVARIABLELIVENESS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

using TensorValue = TypedValue<RankedTensorType>;

#define DEBUG_TYPE "tritonintelgpu-reduce-variable-liveness"

namespace {

/// Return true if the lifespan of the V value is considered long.
static bool isLongLifeSpanVariable(Value v,
                                   const LivenessBlockInfo *livenessBlockInfo) {
  // Case 1: Variable liveness expend before the dot block.
  //         e.g. used in a block - loaded in another block
  return livenessBlockInfo->isLiveIn(v);
}

/// Create a prefetch operation for the given load operation.
static void createPrefetchOp(tt::LoadOp loadOp) {
  Operation *op = loadOp.getPtr().getDefiningOp();
  OpBuilder builder(op);
  // TODO: Add prefetchOp after last dependency (between ptr and mask when PR
  // #3634 is merged)
  builder.setInsertionPointAfter(op);
  auto prefetchOp = builder.create<ttgi::PrefetchOp>(
      loadOp->getLoc(), loadOp.getPtr(), loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());

  // inherit attributes from the load operation
  auto attrs = loadOp->getAttrDictionary();
  prefetchOp->setAttrs(attrs);
}

/// Investigate opportunities for the reducing register pressure by moving DotOp
/// operands.
static bool optimizeDotOperands(scf::ForOp forOp,
                                SmallVector<Value> &prefetchedValue,
                                Liveness &livenessAnalysis) {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<RankedTensorType>(v.getType()).getEncoding();
  };

  // returns loadOp that loads the value v.
  auto getLoad = [](Value v) -> std::optional<triton::LoadOp> {
    // walk back to Load operation
    Operation *op = v.getDefiningOp();
    while (op) {
      if (op->getNumOperands() != 1)
        break;
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        return loadOp;
      }
      op = op->getOperand(0).getDefiningOp();
    }
    return std::nullopt;
  };

  // Prefetch the dotOp operand and move it closer to dotOp.
  auto moveOperand = [&prefetchedValue](uint8_t opId, triton::DotOp dotOp,
                                        tt::LoadOp loadOp) {
    OpBuilder b(dotOp);
    TensorValue tensorV = opId == 0 ? dotOp.getA() : dotOp.getB();
    auto tensorType = cast<RankedTensorType>(tensorV.getType());
    Type elType = tensorType.getElementType();
    Type loadType =
        cast<RankedTensorType>(loadOp.getResult().getType()).getElementType();
    // Types mismatch => Skip this case to avoid inserting to
    // many addtional operations in the loop.
    if (elType != loadType)
      return;
    // Only pointer to tensor are moved
    if (!mlir::triton::isTensorPointerType(loadOp.getPtr().getType()))
      return;
    if (std::find(prefetchedValue.begin(), prefetchedValue.end(),
                  loadOp.getPtr()) == prefetchedValue.end()) {
      createPrefetchOp(loadOp);
      prefetchedValue.push_back(loadOp.getPtr());
    }
    b.setInsertionPoint(dotOp);
    auto newLoad = cast<tt::LoadOp>(b.clone(*loadOp.getOperation()));
    auto newCvt = b.create<ttg::ConvertLayoutOp>(tensorV.getLoc(), tensorType,
                                                 newLoad.getResult());
    dotOp.setOperand(opId, newCvt.getResult());
  };

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // Only accepts dotOps encoded as DPAS MMA
      auto dstDpasEnc = dyn_cast<ttg::intel::DpasEncodingAttr>(
          getEncoding(dotOp.getResult()));
      if (!dstDpasEnc)
        // Don't rewrite if any other type is found.
        return false;
      dotsInFor.push_back(dotOp);
    }

  if (dotsInFor.empty())
    return false;

  for (triton::DotOp dot : dotsInFor) {
    auto aVals = getLoad(dot.getA());
    auto bVals = getLoad(dot.getB());
    auto livenessBlockInfo = livenessAnalysis.getLiveness(dot->getBlock());

    if (isLongLifeSpanVariable(dot.getA(), livenessBlockInfo) && aVals) {
      tt::LoadOp loadOp = aVals.value();
      moveOperand(0, dot, loadOp);
    }
    if (isLongLifeSpanVariable(dot.getB(), livenessBlockInfo) && bVals) {
      tt::LoadOp loadOp = bVals.value();
      moveOperand(1, dot, loadOp);
    }
  }
  return true;
}

class ReduceVariableLivenessPass
    : public triton::gpu::intel::impl::TritonIntelGPUReduceVariableLivenessBase<
          ReduceVariableLivenessPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUReduceVariableLivenessBase<
      ReduceVariableLivenessPass>::TritonIntelGPUReduceVariableLivenessBase;

  void runOnOperation() override {

    // Canonicalize convert ops to make the pattern matching easier.
    SmallVector<Value> prefetchedValue;
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }

    Operation *rootOperation = getOperation();
    Liveness livenessAnalysis(rootOperation);
    rootOperation->walk([&](scf::ForOp forOp) {
      if (optimizeDotOperands(forOp, prefetchedValue, livenessAnalysis))
        return;
    });
  }
};

} // namespace
