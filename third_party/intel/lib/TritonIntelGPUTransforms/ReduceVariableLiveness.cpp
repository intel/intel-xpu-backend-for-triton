#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Analysis/Liveness.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

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

constexpr uint32_t TOTAL_BLOCK_SIZE_THRESHOLD_IN_BYTES = 32768;
constexpr uint32_t LARGE_TENSOR_MINOR_SHAPE_THRESHOLD = 128;
constexpr uint32_t LARGE_TENSOR_MAJOR_SHAPE_THRESHOLD = 128;
constexpr uint32_t LARGE_TENSOR_SIZE_THRESHOLD_IN_BYTES =
    LARGE_TENSOR_MAJOR_SHAPE_THRESHOLD * LARGE_TENSOR_MINOR_SHAPE_THRESHOLD * 2;

/// Return the total size of the tensor type /p tensorType in bytes.
/// If the tensor element type is not a int or a float, this function return 0.
unsigned getSizeInBytes(RankedTensorType &tensorType) {
  Type elType = tensorType.getElementType();
  if (!elType.isIntOrFloat())
    return 0;
  unsigned elTypeBitWidth = elType.getIntOrFloatBitWidth();
  unsigned totalNumElement = 1;
  for (int64_t dim : tensorType.getShape()) {
    totalNumElement *= dim;
  }
  return totalNumElement * (elTypeBitWidth / 8);
}

/// Return the size in bytes of all the variables that have been identified as
/// live-in variable for the block.
unsigned getBlockLiveInSizeInBytes(const LivenessBlockInfo *livenessBlockInfo) {
  unsigned blockInSize = 0;
  for (Value liveVal : livenessBlockInfo->in()) {
    Type liveValTy = liveVal.getType();
    if (TensorValue tensorV = dyn_cast<TensorValue>(liveVal)) {
      auto tensorType = dyn_cast<RankedTensorType>(tensorV.getType());
      blockInSize += getSizeInBytes(tensorType);
    } else if (liveValTy.isIntOrFloat()) {
      blockInSize += liveValTy.getIntOrFloatBitWidth() / 8;
    }
  }
  return blockInSize;
}

/// Return true if the lifespan of the \p v value is considered long.
bool isLongLifeSpanVariable(Value v, const LivenessBlockInfo *livenessBlockInfo,
                            unsigned LiveInSizeInBytes) {
  // The variable is considered as a long life span elected for being moved if:
  // the live-in variables of the forOp consist in a large amount of bytes and
  // the variable defined by `v` is a large tensor (with large amount of element
  // in the minor dimenssion) and the variable liveness of `v` expends before
  // the dot block. i.e. used in a block - loaded in another block
  TensorValue tensorV = dyn_cast<TensorValue>(v);
  if (!tensorV)
    return false;

  auto tensorType = cast<RankedTensorType>(tensorV.getType());
  auto tensorOrder = ttg::getOrder(tensorType);
  return (
      (tensorOrder.size() == 2) &&
      (getSizeInBytes(tensorType) >= LARGE_TENSOR_SIZE_THRESHOLD_IN_BYTES) &&
      (tensorType.getShape()[tensorOrder[1]] >=
       LARGE_TENSOR_MINOR_SHAPE_THRESHOLD) &&
      (LiveInSizeInBytes > TOTAL_BLOCK_SIZE_THRESHOLD_IN_BYTES) &&
      livenessBlockInfo->isLiveIn(v));
}

/// Return true if the \p loadOp is suitable to be moved.
/// \p expectedElementType is the element type expected for the load to be a
/// candidate,
/// \p forOp operation to which we want to move the loadOp
bool isLoadCandidate(tt::LoadOp loadOp, Type expectedElementType,
                     Operation *forOp) {
  if (!tt::isTensorOrTensorPointerType(loadOp.getPtr().getType()))
    return false;
  // LoadOps with non-null mask are not candidates.
  if (loadOp.getMask())
    return false;
  auto loadType = cast<RankedTensorType>(loadOp.getResult().getType());
  Type loadElType = loadType.getElementType();
  // Types mismatch => Skip this case to avoid inserting too
  // many addtional operations in the loop.
  if (expectedElementType != loadElType)
    return false;
  Attribute blockIOAttr =
      loadOp->getAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
  if (!blockIOAttr)
    return false;
  // Only tensor with rank = 2 are considered to be moved
  if (loadType.getShape().size() != 2)
    return false;
  // Only loadOp out of the for loop body are considered to be moved
  if (loadOp->getParentOp() == forOp)
    return false;
  // Multiple users
  if (any_of(loadOp->getUsers(), [&](Operation *user) {
        return ((user->getBlock() == forOp->getBlock()) &&
                user->isBeforeInBlock(forOp));
      }))
    return false;
  // We skip the load if the defining op is not is the same region.
  // To avoid prefetching this data in another region
  // (as the prefetch is added after the defining op).
  if (!loadOp.getPtr().getDefiningOp())
    return false;
  return true;
}

/// Create a prefetch operation for the given load operation.
void createPrefetchOp(tt::LoadOp loadOp) {
  Operation *op = loadOp.getPtr().getDefiningOp();
  OpBuilder builder(op);
  // TODO: Add prefetchOp after last dependency between ptr and mask,
  // if this support is extended to support masks.
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
/// Returns `true` if at least one operand has been moved.
bool optimizeDotOperands(scf::ForOp forOp, SmallVector<Value> &prefetchedValue,
                         Liveness &livenessAnalysis) {
  Block *loop = forOp.getBody();
  bool opMoved = false;

  auto getEncoding = [](Value v) {
    return cast<RankedTensorType>(v.getType()).getEncoding();
  };

  // returns loadOp that loads the value v.
  auto getLoad = [](Value v) -> std::optional<tt::LoadOp> {
    // walk back to Load operation
    Operation *op = v.getDefiningOp();
    while (op) {
      if (auto loadOp = dyn_cast<tt::LoadOp>(op))
        return loadOp;
      if (!isa<ttg::ConvertLayoutOp>(op))
        break;
      op = op->getOperand(0).getDefiningOp();
    }
    return std::nullopt;
  };

  // Prefetch the dotOp operand and move it closer to dotOp.
  auto moveOperand = [&prefetchedValue, &opMoved](uint8_t opId, tt::DotOp dotOp,
                                                  tt::LoadOp loadOp) {
    assert(opId < 2 && "opId must be 0 or 1");
    OpBuilder b(dotOp);
    TensorValue tensorV = opId == 0 ? dotOp.getA() : dotOp.getB();
    auto tensorType = cast<RankedTensorType>(tensorV.getType());
    Operation *insertBeforeOp = dotOp;
    SmallVector<Operation *> usesInSameLoop;
    // Other use(s) in the same loop
    for (Operation *user : loadOp->getUsers()) {
      if (user == dotOp)
        continue;
      if (user->getParentOp() == dotOp->getParentOp()) {
        usesInSameLoop.push_back(user);
        if (user->isBeforeInBlock(insertBeforeOp))
          insertBeforeOp = user;
      }
    }

    if (std::find(prefetchedValue.begin(), prefetchedValue.end(),
                  loadOp.getPtr()) == prefetchedValue.end()) {
      createPrefetchOp(loadOp);
      prefetchedValue.push_back(loadOp.getPtr());
    }
    b.setInsertionPoint(insertBeforeOp);
    auto newLoad = cast<tt::LoadOp>(b.clone(*loadOp.getOperation()));
    auto newCvt = b.create<ttg::ConvertLayoutOp>(tensorV.getLoc(), tensorType,
                                                 newLoad.getResult());
    dotOp.setOperand(opId, newCvt.getResult());

    // Update other user in the same loop if any
    for (Operation *user : usesInSameLoop)
      user->replaceUsesOfWith(loadOp.getResult(), newLoad.getResult());

    // Multiple users:
    // Note that if other users come before the loop, the loadOp is not a
    // candidate for being moved.
    if (!loadOp->use_empty()) {
      b.setInsertionPointAfter(dotOp->getParentOp());
      auto copyLoad = cast<tt::LoadOp>(b.clone(*loadOp.getOperation()));
      loadOp->replaceAllUsesWith(copyLoad);
    }
    opMoved = true;
  };

  SmallVector<tt::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
      // Only accepts dotOps encoded as DPAS MMA
      if (!ttgi::hasDpasEncoding(dotOp.getResult().getType()))
        // Don't rewrite if any other type is found.
        return false;
      dotsInFor.push_back(dotOp);
    }

  if (dotsInFor.empty())
    return false;

  for (tt::DotOp dot : dotsInFor) {
    std::optional<tt::LoadOp> aVals = getLoad(dot.getA());
    std::optional<tt::LoadOp> bVals = getLoad(dot.getB());

    auto livenessBlockInfo = livenessAnalysis.getLiveness(dot->getBlock());
    unsigned LiveInSizeInBytes = getBlockLiveInSizeInBytes(livenessBlockInfo);

    if (aVals && isLongLifeSpanVariable(aVals.value(), livenessBlockInfo,
                                        LiveInSizeInBytes)) {
      tt::LoadOp loadOp = aVals.value();
      auto tensorType = cast<RankedTensorType>(dot.getA().getType());
      if (isLoadCandidate(loadOp, tensorType.getElementType(), forOp))
        moveOperand(0, dot, loadOp);
    }
    if (bVals && isLongLifeSpanVariable(bVals.value(), livenessBlockInfo,
                                        LiveInSizeInBytes)) {
      tt::LoadOp loadOp = bVals.value();
      auto tensorType = cast<RankedTensorType>(dot.getB().getType());
      if (isLoadCandidate(loadOp, tensorType, forOp))
        moveOperand(1, dot, loadOp);
    }
  }
  return opMoved;
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
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                      &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }

    Operation *rootOperation = getOperation();
    Liveness livenessAnalysis(rootOperation);
    // TODO: extend the pass to handle `while` loops.
    rootOperation->walk([&](scf::ForOp forOp) {
      if (optimizeDotOperands(forOp, prefetchedValue, livenessAnalysis)) {
        // The liveness analysis must be re-performed before the processing of
        // each "for loop" given that the liveness of variables may have changed
        // as a result of the code, and specifically `LoadOps`, being modified
        // by the pass.
        livenessAnalysis = Liveness(rootOperation);
        return;
      }
    });
  }
};

} // namespace
