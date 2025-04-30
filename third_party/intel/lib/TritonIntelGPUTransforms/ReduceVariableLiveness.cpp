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

#include "intel/include/Analysis/Liveness.h"
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

#include <iostream>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

using TensorValue = TypedValue<RankedTensorType>;

#define DEBUG_TYPE "tritonintelgpu-reduce-variable-liveness"

namespace {

#define TOTAL_BLOCK_SIZE_THRESHOLD_IN_BYTES 32768
#define LARGE_TENSOR_SIZE_THRESHOLD_IN_BYTES 8192

static unsigned getSizeInBytes(RankedTensorType &tensorType) {
  unsigned elTypeBitWidth = tensorType.getElementType().getIntOrFloatBitWidth();
  unsigned totalNumElement = 1;
  for (int64_t dim : tensorType.getShape()) {
    totalNumElement *= dim;
  }
  return totalNumElement * (elTypeBitWidth / 8);
}

static unsigned
getBlockLiveInSizeInBytes(const LivenessBlockInfo *livenessBlockInfo) {
  unsigned blockInSize = 0;
  for (Value liveVal : livenessBlockInfo->in()) {
    // Only tensors are taken into account as other variables do not count much
    // in the total number of registers required.
    if (TensorValue tensorV = dyn_cast<TensorValue>(liveVal)) {
      auto tensorType = dyn_cast<RankedTensorType>(tensorV.getType());
      blockInSize += getSizeInBytes(tensorType);
    }
  }
  return blockInSize;
}

/// Return true if the lifespan of the \p v value is considered long.
static bool isLongLifeSpanVariable(Value v,
                                   const LivenessBlockInfo *livenessBlockInfo,
                                   unsigned LiveInSizeInBytes) {
  // The variable is considered as a long life span elected for being moved if:
  // The live-in variables of the forOp consist in a large amount of bytes and
  // The variable defined by `v` is a large tensor and
  // The variable liveness of `v` expends before the dot block.
  // i.e. used in a block - loaded in another block
  if (TensorValue tensorV = dyn_cast<TensorValue>(v)) {
    auto tensorType = cast<RankedTensorType>(tensorV.getType());
    return (
        (LiveInSizeInBytes > TOTAL_BLOCK_SIZE_THRESHOLD_IN_BYTES) &&
        (getSizeInBytes(tensorType) > LARGE_TENSOR_SIZE_THRESHOLD_IN_BYTES) &&
        livenessBlockInfo->isLiveIn(v));
  }
  return false;
}

static bool isLoadCandidate(tt::LoadOp loadOp, TensorValue tensorV,
                            Operation *forOp) {
  // Only pointer to tensor are considered to be moved
  if (!mlir::triton::isTensorPointerType(loadOp.getPtr().getType()))
    return false;
  auto tensorType = cast<RankedTensorType>(tensorV.getType());
  Type elType = tensorType.getElementType();
  Type loadElType =
      cast<RankedTensorType>(loadOp.getResult().getType()).getElementType();
  // Types mismatch => Skip this case to avoid inserting too
  // many addtional operations in the loop.
  if (elType != loadElType)
    return false;
  Attribute blockIOAttr = loadOp->getAttr(
      mlir::triton::gpu::intel::TritonIntelGPUDialect::getBlockIOAttrName());
  if (!blockIOAttr)
    return false;
  // Only tensor with rank = 2 are considered to be moved
  if (tensorType.getShape().size() != 2)
    return false;
  // Only loadOp out of the for loop body are considered to be moved
  if (loadOp->getParentOp() == forOp)
    return false;
  return true;
}

/// Create a prefetch operation for the given load operation.
static void createPrefetchOp(tt::LoadOp loadOp) {
  Operation *op = loadOp.getPtr().getDefiningOp();
  OpBuilder builder(op);
  // TODO: Add prefetchOp after last dependency between ptr and mask,
  // if this support is extended to tensor of pointers.
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
      if (auto loadOp = dyn_cast<triton::LoadOp>(op))
        return loadOp;
      if (!isa<ttg::ConvertLayoutOp>(op))
        break;
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
    unsigned LiveInSizeInBytes = getBlockLiveInSizeInBytes(livenessBlockInfo);

    if (aVals && isLongLifeSpanVariable(aVals.value(), livenessBlockInfo,
                                        LiveInSizeInBytes)) {
      tt::LoadOp loadOp = aVals.value();
      if (isLoadCandidate(loadOp, dot.getA(), forOp))
        moveOperand(0, dot, loadOp);
    }
    if (bVals && isLongLifeSpanVariable(bVals.value(), livenessBlockInfo,
                                        LiveInSizeInBytes)) {
      tt::LoadOp loadOp = bVals.value();
      if (isLoadCandidate(loadOp, dot.getB(), forOp))
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
