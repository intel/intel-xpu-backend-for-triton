#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUREORDERFORLIVENESS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define DEBUG_TYPE "tritonintelgpu-reorder-for-liveness"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Blocks smaller than this cannot benefit: with few values live at once the
/// ordering barely matters and the analysis is pure overhead.
constexpr unsigned MinOpsToConsider = 32;

/// Estimated peak per-lane register footprint, in bytes, above which IGC starts
/// to spill straight-line code. A hardware thread has 256 x 64 B = 16 KB of GRF
/// in large-GRF mode, i.e. 512 B per lane at 32 lanes. The trigger is set at
/// half of that because immediates, address computation and IGC's own
/// temporaries need headroom: the #7782 kernel peaked at an estimated
/// 368 B/lane and still spilled 384 B at 256 GRF.
constexpr unsigned PressureTriggerBytesPerLane = 256;

/// Smallest reduction in estimated peak footprint, in bytes per lane, that is
/// worth reordering a block for. 32 B/lane is 16 GRF at 32 lanes; below that
/// the estimate is not accurate enough for the churn to be justified.
constexpr unsigned MinImprovementBytesPerLane = 32;

/// Marks an operation that no memory operation depends on.
constexpr unsigned NoAnchor = std::numeric_limits<unsigned>::max();

/// Size in bytes of one element of \p ty, or 0 if it is not a type we account
/// for.
unsigned getElementSizeInBytes(Type ty) {
  if (isa<tt::PointerType>(ty))
    return 8;
  if (ty.isIntOrFloat())
    return llvm::divideCeil(ty.getIntOrFloatBitWidth(), 8);
  return 0;
}

/// Per-lane register footprint of \p v in bytes: how many elements of \p v this
/// work-item holds, times the element size. Values whose type we cannot account
/// for contribute 0 -- they cost the same in every ordering, so they cannot
/// change which ordering wins.
unsigned getFootprintInBytes(Value v) {
  Type ty = v.getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
    // Only a distributed layout tells us how many elements a work-item holds.
    // An unset or not yet resolved encoding (e.g. Gluon's auto encoding) does
    // not.
    if (!isa_and_nonnull<ttg::DistributedEncodingTrait>(tensorTy.getEncoding()))
      return 0;
    unsigned elemSize = getElementSizeInBytes(tensorTy.getElementType());
    return elemSize ? ttg::getTotalElemsPerThread(tensorTy) * elemSize : 0;
  }
  return getElementSizeInBytes(ty);
}

/// Greedy list scheduler over the operations of a single block that minimizes
/// the peak per-lane footprint of simultaneously live values.
///
/// Operations are indexed by their original position. Operations with memory
/// effects ("anchors") are kept in their original relative order by a chain of
/// dependence edges, and are scheduled as soon as their operands allow, so
/// loads stay as early as they were and stores still free their operands as
/// early as they did. Only effect-free operations are free to move.
class BlockScheduler {
public:
  BlockScheduler(Block &block) : block(block) {
    Operation *prevAnchor = nullptr;
    for (Operation &op : block.without_terminator()) {
      index[&op] = ops.size();
      ops.push_back(&op);
      // An op with a memory effect, or with no results to keep alive, is left
      // where it is relative to the other such ops.
      bool isAnchor = !isMemoryEffectFree(&op) || op.getNumResults() == 0;
      anchor.push_back(isAnchor);
      anchorPred.push_back(isAnchor ? prevAnchor : nullptr);
      if (isAnchor)
        prevAnchor = &op;
    }

    unsigned numOps = ops.size();
    dataPreds.resize(numOps);
    preds.resize(numOps);
    succs.resize(numOps);
    weight.assign(numOps, 0);
    dataUses.assign(numOps, 0);
    escapes.assign(numOps, false);

    for (unsigned i = 0; i < numOps; ++i) {
      Operation *op = ops[i];
      for (Value result : op->getResults()) {
        weight[i] += getFootprintInBytes(result);
        // A result used by the terminator, or from another block, stays live to
        // the end of this block and can never be killed by the schedule.
        for (Operation *user : result.getUsers())
          if (!index.contains(user))
            escapes[i] = true;
      }
      // Distinct in-block producers of this op's operands. These are the edges
      // that carry a value, and hence the only ones that keep one alive.
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        auto it = def ? index.find(def) : index.end();
        if (it != index.end() && !llvm::is_contained(dataPreds[i], it->second))
          dataPreds[i].push_back(it->second);
      }
      for (unsigned p : dataPreds[i])
        ++dataUses[p];

      // Scheduling edges are the data edges plus the anchor chain, which orders
      // the memory effects without consuming any value.
      preds[i] = dataPreds[i];
      if (Operation *chained = anchorPred[i]) {
        unsigned chainedIdx = index.lookup(chained);
        if (!llvm::is_contained(preds[i], chainedIdx))
          preds[i].push_back(chainedIdx);
      }
    }

    for (unsigned i = 0; i < numOps; ++i)
      for (unsigned p : preds[i])
        succs[p].push_back(i);

    // Longest path from each op to a sink, used to break ties depth-first: it
    // is better to finish a chain that is already started (and free its inputs)
    // than to start a new one.
    height.assign(numOps, 0);
    for (unsigned i = numOps; i-- > 0;)
      for (unsigned s : succs[i])
        height[i] = std::max(height[i], height[s] + 1);

    // Position in the anchor chain of the first memory operation that depends
    // on each operation. Scheduling by this first means the address
    // computation of a load is issued ahead of unrelated arithmetic, so the
    // load itself is not sunk towards its uses.
    feedsAnchor.assign(numOps, NoAnchor);
    unsigned rank = 0;
    for (unsigned i = 0; i < numOps; ++i)
      if (anchor[i])
        feedsAnchor[i] = rank++;
    for (unsigned i = numOps; i-- > 0;)
      for (unsigned s : succs[i])
        feedsAnchor[i] = std::min(feedsAnchor[i], feedsAnchor[s]);
  }

  /// Whether reordering this block is allowed at all.
  bool isEligible() const {
    if (ops.size() < MinOpsToConsider)
      return false;
    for (Operation *op : ops) {
      // Structured control flow is handled by code sinking, not here.
      if (op->getNumRegions() != 0)
        return false;
      // Never perturb a matmul schedule.
      if (isa<tt::DotOpInterface>(op))
        return false;
    }
    return true;
  }

  /// Peak simultaneous live footprint, in bytes per lane, of \p order.
  unsigned getPeakBytes(ArrayRef<unsigned> order) const {
    SmallVector<unsigned> remainingUses = dataUses;
    unsigned live = 0, peak = 0;
    for (unsigned i : order) {
      // At the defining operation the result and all of its operands are live
      // at the same time.
      live += weight[i];
      peak = std::max(peak, live);
      for (unsigned p : dataPreds[i])
        if (--remainingUses[p] == 0 && !escapes[p])
          live -= weight[p];
      if (dataUses[i] == 0 && !escapes[i])
        live -= weight[i];
    }
    return peak;
  }

  SmallVector<unsigned> getOriginalOrder() const {
    return llvm::to_vector(llvm::seq(0u, unsigned(ops.size())));
  }

  /// A topological order chosen greedily to keep the live footprint small.
  SmallVector<unsigned> getScheduledOrder() const {
    unsigned numOps = ops.size();
    SmallVector<unsigned> remainingUses = dataUses;
    SmallVector<unsigned> unmetDeps(numOps);
    SmallVector<unsigned> ready, order;
    order.reserve(numOps);

    for (unsigned i = 0; i < numOps; ++i) {
      unmetDeps[i] = preds[i].size();
      if (unmetDeps[i] == 0)
        ready.push_back(i);
    }

    while (!ready.empty()) {
      unsigned bestPos = 0;
      // The anchor chain admits at most one ready anchor; take it immediately
      // to hold loads and stores at their original point in the stream.
      auto anchorIt =
          llvm::find_if(ready, [&](unsigned i) { return anchor[i]; });
      if (anchorIt != ready.end()) {
        bestPos = std::distance(ready.begin(), anchorIt);
      } else {
        unsigned bestAnchor = 0;
        int64_t bestDelta = 0, bestNegHeight = 0;
        for (unsigned pos = 0; pos < ready.size(); ++pos) {
          unsigned i = ready[pos];
          // Change in live footprint from issuing `i` now: its result becomes
          // live unless it dies immediately, and every operand whose last use
          // this is dies.
          int64_t delta = (dataUses[i] == 0 && !escapes[i]) ? 0 : weight[i];
          for (unsigned p : dataPreds[i])
            if (remainingUses[p] == 1 && !escapes[p])
              delta -= weight[p];
          // What the earliest pending memory operation needs comes first, then
          // the smallest growth in live footprint, then the deepest chain so
          // that a chain already started is finished before another is opened.
          int64_t negHeight = -int64_t(height[i]);
          if (pos == 0 || std::tie(feedsAnchor[i], delta, negHeight) <
                              std::tie(bestAnchor, bestDelta, bestNegHeight)) {
            bestAnchor = feedsAnchor[i];
            bestDelta = delta;
            bestNegHeight = negHeight;
            bestPos = pos;
          }
        }
      }

      unsigned i = ready[bestPos];
      LDBG("pick " << ops[i]->getName() << " #" << i << " anchor "
                   << feedsAnchor[i] << " height " << height[i] << " ready "
                   << ready.size());
      ready.erase(ready.begin() + bestPos);
      order.push_back(i);
      for (unsigned p : dataPreds[i])
        --remainingUses[p];
      for (unsigned s : succs[i])
        if (--unmetDeps[s] == 0)
          ready.push_back(s);
    }

    assert(order.size() == numOps && "dependence graph must be acyclic");
    return order;
  }

  /// Rewrite the block to follow \p order.
  void apply(ArrayRef<unsigned> order) {
    Operation *terminator = block.getTerminator();
    for (unsigned i : order)
      ops[i]->moveBefore(terminator);
  }

private:
  Block &block;
  SmallVector<Operation *> ops;
  DenseMap<Operation *, unsigned> index;
  /// Operations defining an operand of each operation.
  SmallVector<SmallVector<unsigned>> dataPreds;
  /// Scheduling dependences: data edges plus the anchor chain, and their
  /// transpose.
  SmallVector<SmallVector<unsigned>> preds, succs;
  /// Per-lane footprint of each operation's results, number of in-block
  /// consumers of them, longest path to a sink, and the first memory operation
  /// depending on it.
  SmallVector<unsigned> weight, dataUses, height, feedsAnchor;
  SmallVector<bool> anchor, escapes;
  SmallVector<Operation *> anchorPred;
};

class TritonIntelGPUReorderForLivenessPass
    : public triton::gpu::intel::impl::TritonIntelGPUReorderForLivenessBase<
          TritonIntelGPUReorderForLivenessPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUReorderForLivenessBase<
      TritonIntelGPUReorderForLivenessPass>::
      TritonIntelGPUReorderForLivenessBase;

  void runOnOperation() override {
    getOperation()->walk([](Block *block) {
      if (!block->mightHaveTerminator())
        return;

      BlockScheduler scheduler(*block);
      if (!scheduler.isEligible())
        return;

      unsigned originalPeak =
          scheduler.getPeakBytes(scheduler.getOriginalOrder());
      if (originalPeak <= PressureTriggerBytesPerLane) {
        LDBG("skip (peak " << originalPeak << " B/lane is within budget)");
        return;
      }

      SmallVector<unsigned> order = scheduler.getScheduledOrder();
      unsigned scheduledPeak = scheduler.getPeakBytes(order);
      if (scheduledPeak + MinImprovementBytesPerLane > originalPeak) {
        LDBG("skip (peak " << originalPeak << " B/lane, best order found is "
                           << scheduledPeak << " B/lane)");
        return;
      }

      LDBG("reorder: peak " << originalPeak << " -> " << scheduledPeak
                            << " B/lane");
      scheduler.apply(order);
    });
  }
};

} // namespace
