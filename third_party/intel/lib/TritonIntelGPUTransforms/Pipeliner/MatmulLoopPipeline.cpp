#include "Schedule.h"
#include "include/triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <cstdlib>
#include <cstring>
#include <limits>

#define DEBUG_TYPE "tritonintelgpu-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

static ttg::DotOperandEncodingAttr allTransitiveUsesHaveDotEncoding(Value val);

namespace {

/// A load operation eligible for prefetching. Only tt::LoadOp and
/// tt::DescriptorLoadOp are valid. Static predicates centralize the
/// candidacy logic.
struct PrefetchCandidate {
  explicit PrefetchCandidate(Operation *op) : op(op) {
    assert(
        (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op)) &&
        "only tt::LoadOp, tt::DescriptorLoadOp and tt::DescriptorGatherOp can "
        "be prefetched");
  }
  Operation *op;

  /// Whether \p op has the block_io attribute and rank >= 2 result type,
  /// making it eligible for 2D block prefetching.
  static bool isCandidate(Operation *op) {
    if (!isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
      return false;
    if (!op->getAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName()))
      return false;
    auto resultTy = dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
    return resultTy && resultTy.getRank() >= 2;
  }

  /// Whether all transitive uses of \p op's result feed a dot operation.
  static bool feedsDot(Operation *op) {
    return allTransitiveUsesHaveDotEncoding(op->getResult(0)) != nullptr;
  }
};

/// Return the multi-buffered byte cost of a load's result tensor.
/// Each prefetched load is kept live for \p numStages iterations.
/// Returns UINT_MAX when the tensor is non-ranked or has dynamic
/// dimensions, so callers that compare against a byte budget skip the
/// load conservatively. Uses ceil-div on bits/8 to avoid under-counting
/// sub-byte types, and widens arithmetic to int64_t to avoid overflow
/// on large tensors before saturating back to unsigned.
static unsigned getTileBytes(Operation *op, int numStages) {
  auto tensorType = dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
  if (!tensorType || !tensorType.hasStaticShape())
    return std::numeric_limits<unsigned>::max();
  unsigned bits = tensorType.getElementType().getIntOrFloatBitWidth();
  int64_t numElems = tensorType.getNumElements();
  int64_t bytesPerElem = llvm::divideCeil(bits, 8u);
  int64_t totalBytes = numElems * bytesPerElem * numStages;
  return (totalBytes > std::numeric_limits<unsigned>::max())
             ? std::numeric_limits<unsigned>::max()
             : static_cast<unsigned>(totalBytes);
}

/// Collect all prefetch candidates from the loop body. Dot-feeding loads
/// are always collected; elementwise loads are only collected on architectures
/// with 256B prefetch support (Xe3P+), where software prefetch can outpace the
/// hardware prefetcher.
static SmallVector<PrefetchCandidate>
collectPrefetchCandidates(scf::ForOp forOp, int numStages) {
  constexpr unsigned kMaxElementwisePrefetchOps = 4;
  constexpr unsigned kMaxPerLoadPrefetchBytes = 16384;

  auto moduleOp = forOp->getParentOfType<ModuleOp>();
  bool enableElementwisePrefetch = moduleOp->hasAttr(
      ttgi::TritonIntelGPUDialect::getSupportPrefetch256BAttrName());

  unsigned numElementwisePrefetch = 0;

  SmallVector<PrefetchCandidate> candidates;
  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block.
  for (Operation &op : forOp) {
    if (!PrefetchCandidate::isCandidate(&op))
      continue;

    if (PrefetchCandidate::feedsDot(&op)) {
      candidates.emplace_back(&op);
    } else if (enableElementwisePrefetch &&
               numElementwisePrefetch < kMaxElementwisePrefetchOps) {
      unsigned tileBytes = getTileBytes(&op, numStages);
      if (tileBytes > kMaxPerLoadPrefetchBytes) {
        LDBG("Skipping elementwise prefetch: per-load budget exceeded ("
             << tileBytes << " > " << kMaxPerLoadPrefetchBytes << " bytes)");
        continue;
      }
      candidates.emplace_back(&op);
      ++numElementwisePrefetch;
    }
  }
  return candidates;
}

/// Return true iff upstream loop scheduling annotations are present on \p forOp
/// and cover every prefetch candidate, so Intel's pipeliner can validate the
/// upstream contract. The check is observational: callers do not branch on it
/// today, but a `false` return signals that we cannot rely on the annotations
/// being a faithful description of the candidate set.
///
/// All of the following must hold:
///   1. The opt-in env var TRITON_INTEL_ANNOTATE_LATENCIES is "1" (matching
///      the activation in third_party/intel/backend/compiler.py).
///   2. tt::CoarseSchedule::deSerialize succeeds, which requires
///      `tt.scheduled_max_stage` on the for-op.
///   3. Every candidate carries `loop.stage`. The only sanctioned exception is
///      the Xe3P+ elementwise carve-out (loads that do not feed a dot), which
///      upstream cannot annotate because the upstream scheduler keys on dot
///      operands -- see collectPrefetchCandidates.
static bool annotationsAreUsable(scf::ForOp forOp,
                                 ArrayRef<PrefetchCandidate> candidates) {
  // (1) Env-var gate. Read with std::getenv to match compiler.py's
  // `os.environ.get(...) == "1"` semantics and avoid the assertIsRecognized
  // allowlist in mlir::triton::tools::getBoolEnv.
  const char *envVal = std::getenv("TRITON_INTEL_ANNOTATE_LATENCIES");
  if (!envVal || std::strcmp(envVal, "1") != 0) {
    LDBG(
        "annotationsAreUsable: env-off (TRITON_INTEL_ANNOTATE_LATENCIES != 1)");
    return false;
  }

  // (2) `tt.scheduled_max_stage` must be on the for-op. deSerialize returns
  // failure() when the attr is absent.
  tt::CoarseSchedule upstream;
  if (failed(upstream.deSerialize(forOp, /*normalizeClusterId=*/true))) {
    LDBG("annotationsAreUsable: no tt.scheduled_max_stage on for-op");
    return false;
  }

  // (3) Each candidate must carry loop.stage, except the Xe3P+ elementwise
  // carve-out (a candidate that does not feed a dot, on a module that supports
  // 256B prefetch -- see collectPrefetchCandidates lines 100-110).
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  bool enableElementwisePrefetch = moduleOp->hasAttr(
      ttgi::TritonIntelGPUDialect::getSupportPrefetch256BAttrName());
  for (const PrefetchCandidate &c : candidates) {
    if (c.op->hasAttr(tt::kLoopStageAttrName))
      continue;
    bool isElementwiseCarveOut =
        enableElementwisePrefetch && !PrefetchCandidate::feedsDot(c.op);
    if (isElementwiseCarveOut)
      continue;
    LDBG("annotationsAreUsable: candidate without loop.stage: " << *c.op);
    return false;
  }

  return true;
}

} // namespace

/// Replace the ForOp's yield with a new one with the given operands appended.
static void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  assert(!newOperands.empty() && "Expecting at least one operand");

  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands().begin(),
                              yieldOp->getOperands().end());
  operands.append(newOperands.begin(), newOperands.end());

  OpBuilder builder(yieldOp);
  scf::YieldOp::create(builder, yieldOp->getLoc(), operands);
  yieldOp->erase();
}

static ttg::DotOperandEncodingAttr getDotEncodingFromUser(Operation *user) {
  if (user->getNumResults() != 1)
    return nullptr;

  OpResult res = user->getResult(0);
  auto tensorType = dyn_cast<RankedTensorType>(res.getType());
  if (!tensorType)
    return nullptr;

  Attribute layout = tensorType.getEncoding();
  return isa<ttg::SharedEncodingTrait, ttg::BlockedEncodingAttr>(layout)
             ? allTransitiveUsesHaveDotEncoding(res)
             : llvm::dyn_cast_or_null<ttg::DotOperandEncodingAttr>(layout);
}

/// If all the transitive uses of the given value are used by a convert to the
/// same dot operand encoding, return the encoding. Otherwise return nullptr.
static ttg::DotOperandEncodingAttr allTransitiveUsesHaveDotEncoding(Value val) {
  ttg::DotOperandEncodingAttr attr{nullptr};
  LDBG("Checking users of " << val);
  for (Operation *user : val.getUsers()) {
    ttg::DotOperandEncodingAttr dotAttr =
        isa<triton::DotOp>(user)
            ? dyn_cast<ttg::DotOperandEncodingAttr>(
                  cast<RankedTensorType>(val.getType()).getEncoding())
            : getDotEncodingFromUser(user);
    if (!dotAttr || (attr != nullptr && attr != dotAttr)) {
      LDBG("no dot attribute found for user: " << *user);
      return nullptr;
    }
    attr = dotAttr;
  }
  return attr;
}

/// Create a prefetch operation for the given load operation.
static void createPrefetchOp(scf::ForOp &forOp, tt::LoadOp loadOp,
                             bool useAnnotations) {
  OpBuilder builder(forOp);
  builder.setInsertionPoint(loadOp);
}

template <class> inline constexpr bool always_false_v = false;

/// Create a prefetch operation for the given load operation.
template <class OpType>
void createPrefetchOp(scf::ForOp &forOp, OpType loadOp, bool useAnnotations) {
  OpBuilder builder(forOp);
  builder.setInsertionPoint(loadOp);

  Operation *prefetchOp = nullptr;
  if constexpr (std::is_same_v<OpType, tt::DescriptorLoadOp>) {
    prefetchOp = ttgi::DescriptorPrefetchOp::create(
        builder, loadOp->getLoc(), loadOp.getDesc(), loadOp.getIndices(),
        loadOp.getCache(), loadOp.getEvict());
  } else if constexpr (std::is_same_v<OpType, tt::DescriptorGatherOp>) {
    prefetchOp = ttgi::DescriptorGatherPrefetchOp::create(
        builder, loadOp->getLoc(), loadOp.getDesc(), loadOp.getXOffsets(),
        loadOp.getYOffset(), triton::CacheModifier::CA,
        triton::EvictionPolicy::NORMAL);
  } else if constexpr (std::is_same_v<OpType, tt::LoadOp>) {
    prefetchOp = ttgi::PrefetchOp::create(
        builder, loadOp->getLoc(), loadOp.getPtr(), loadOp.getMask(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
  } else {
    static_assert(always_false_v<OpType> &&
                  "Unsupported OpType for createPrefetchOp");
  }

  // inherit attributes from the load operation
  auto attrs = loadOp->getAttrDictionary();
  prefetchOp->setAttrs(attrs);
  // Drop upstream scheduling attrs that may have been attached by
  // `tritongpu-assign-latencies` / `tritongpu-schedule-loops`. The Intel
  // pipeliner owns scheduling for prefetch ops and would otherwise produce
  // self-inconsistent IR (legacy stage assignment + upstream stage attr).
  // `tt.latency` was already consumed by the upstream passes.
  prefetchOp->removeAttr(tt::kLoopStageAttrName);
  prefetchOp->removeAttr(tt::kLoopClusterAttrName);

  // When the upstream annotation contract is in effect, copy the source
  // load's `loop.stage` / `loop.cluster` onto the new prefetch op so that
  // `createSchedule` can read the upstream-picked stage instead of the
  // hardcoded legacy value (0). The load itself still carries the attrs;
  // the strip above only cleared the dictionary copy on the prefetch op.
  if (useAnnotations) {
    if (Attribute s = loadOp->getAttr(tt::kLoopStageAttrName)) {
      prefetchOp->setAttr(tt::kLoopStageAttrName, s);
      // Warn if upstream picked a non-zero stage (the legacy hardcoded
      // value is 0; any other value means this PR shifts the prefetch
      // off the legacy stage).
      if (auto stageAttr = dyn_cast<IntegerAttr>(s)) {
        int upstreamStage = stageAttr.getInt();
        if (upstreamStage != 0)
          LDBG("annotation-driven prefetch deviates from legacy stage 0: "
               "stage="
               << upstreamStage << " for load=" << *loadOp);
      }
    }
    if (Attribute c = loadOp->getAttr(tt::kLoopClusterAttrName))
      prefetchOp->setAttr(tt::kLoopClusterAttrName, c);
  }
}

/// Create prefetch operations for the given load candidates.
static void createPrefetchOps(scf::ForOp &forOp,
                              ArrayRef<PrefetchCandidate> candidates,
                              bool useAnnotations) {
  assert(!candidates.empty() && "Expecting at least one candidate");
  for (const PrefetchCandidate &candidate : candidates) {
    TypeSwitch<Operation *>(candidate.op)
        .Case<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(
            [&](auto loadOp) {
              createPrefetchOp(forOp, loadOp, useAnnotations);
            })
        .Default([](Operation *op) {
          llvm_unreachable("Unsupported load operation type");
        });
  }
}

/// Function to mask operations during scheduling.
static Operation *predicateOp(RewriterBase &rewriter, Operation *op,
                              Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;

  return TypeSwitch<Operation *, Operation *>(op)
      .Case<tt::LoadOp, ttgi::PrefetchOp>([&](auto op) {
        rewriter.setInsertionPoint(op);
        Value mask = tt::getPredMask(rewriter, op.getPtr().getType(),
                                     op.getMask(), pred);
        op.getMaskMutable().assign(mask);
        return op;
      })
      .Default([](auto op) { return op; });
}

/// Helper to get the defining operation of a value.
static Operation *getDefOp(Value v, Operation *op, bool includeArg) {
  llvm::SmallDenseSet<Value> seen;
  while (auto arg = dyn_cast<BlockArgument>(v)) {
    if (!includeArg)
      break;
    if (!seen.insert(v).second)
      break;
    if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
      Operation *termOp = op->getBlock()->getTerminator();
      if (auto yieldOp = dyn_cast<scf::YieldOp>(termOp)) {
        v = yieldOp->getOperand(arg.getArgNumber() - 1);
        continue;
      }
      break;
    }
    break;
  }
  return v.getDefiningOp();
}

/// Helper to recursively add dependencies to the same stage.
static void addDep(Operation *op, DenseSet<Operation *> &deps,
                   bool includeArg = true,
                   DenseSet<Operation *> *filter = nullptr) {
  if (filter && filter->count(op))
    return;
  if (!deps.insert(op).second)
    return;

  for (Value operand : op->getOperands()) {
    Operation *defOp = getDefOp(operand, op, includeArg);
    if (defOp && defOp->getBlock() == op->getBlock())
      addDep(defOp, deps, includeArg, filter);
  }
}

// Add operations to the schedule with the given stage based on the filter
// function.
static void addOps(scf::ForOp forOp, int stage,
                   std::vector<std::pair<Operation *, unsigned>> &schedule,
                   std::function<bool(Operation *)> filter) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!filter(&op))
      continue;
    schedule.emplace_back(&op, stage);
  }
}

/// Create the schedule for a matmul loop. This is ad hoc based on how we know
/// matmul loops should be pipelined and is not a generic scheduler.
static std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp forOp, int numStages) {
  SmallVector<Operation *> prefetchOps;
  SmallVector<Operation *> loadOps;
  // Find the prefetch/load ops that will go respectively in stage 0 and stage
  // `numStages - 1`. All the other operations will go in stage `numStages - 1`.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttgi::PrefetchOp, ttgi::DescriptorPrefetchOp,
            ttgi::DescriptorGatherPrefetchOp>(op))
      prefetchOps.emplace_back(&op);
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      // Loads that are neither tensors nor pointers to tensor are not
      // prefetched and could be used by prefetchOp dependencies
      // (typically `advanceOp`).
      // As prefetchOp dependencies are assigned to stage 0, this type of loads
      // must not be explicitely assigned to stage `numStages - 1`.
      if (isa<RankedTensorType>(loadOp.getPtr().getType()))
        loadOps.emplace_back(&op);
    }
    if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
      loadOps.emplace_back(&op);
  }

  DenseSet<Operation *> prefetchAndDeps;
  for (Operation *op : prefetchOps)
    addDep(op, prefetchAndDeps, false);

  // Find depenencies with distance of 1.
  SmallVector<Operation *> distanceOneUsers;
  for (Operation *op : prefetchAndDeps) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = getDefOp(operand, op, true);
      if (defOp)
        distanceOneUsers.push_back(defOp);
    }
  }

  // For the rest of the ops we can move then into stage 1 so that they can be
  // closer to their uses.
  DenseSet<Operation *> stage1deps;
  for (Operation *op : distanceOneUsers)
    addDep(op, stage1deps, true, &prefetchAndDeps);

  DenseSet<Operation *> loadAndDeps;
  for (Operation *op : loadOps)
    addDep(op, loadAndDeps, false, &prefetchAndDeps);

  std::vector<std::pair<Operation *, unsigned>> schedule;

  // Schedule some dependencies with distance of 1 into stage 1 to reduce
  // pressure.
  addOps(forOp, 1, schedule,
         [&](Operation *op) { return stage1deps.count(op); });

  // Then Schedule stage 0.
  // Prefetch ops: stage from `loop.stage` attr (set by `createPrefetchOp`
  // when `useAnnotations` is true) or 0 (legacy hardcoded value, also the
  // value upstream picks for direct-feed-to-dot loads in 3-stage matmul
  // loops). Non-prefetch members of `prefetchAndDeps` (the backward
  // dependencies) stay at stage 0.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!prefetchAndDeps.count(&op))
      continue;
    unsigned stage = 0;
    if (isa<ttgi::PrefetchOp, ttgi::DescriptorPrefetchOp>(&op)) {
      if (auto stageAttr =
              op.getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName))
        stage = stageAttr.getInt();
    }
    schedule.emplace_back(&op, stage);
  }

  // Schedule stage `numStage - 1` first.
  // Finally schedule the dot ops in stage `numStage - 1` so that they get
  // pre-fetched and play well with pretech pass.
  addOps(forOp, numStages - 1, schedule,
         [&](Operation *op) { return loadAndDeps.count(op); });

  addOps(forOp, numStages - 1, schedule, [&](Operation *op) {
    return prefetchAndDeps.count(op) == 0 && stage1deps.count(op) == 0 &&
           loadAndDeps.count(op) == 0;
  });

  return schedule;
}

bool ttgi::preProcessLoopAndGetSchedule(scf::ForOp &forOp, int numStages,
                                        mlir::scf::PipeliningOption &options) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  SmallVector<PrefetchCandidate> candidates =
      collectPrefetchCandidates(forOp, numStages);
  if (candidates.empty()) {
    LDBG("No loads to pipeline");
    return false;
  }

  LLVM_DEBUG({
    DBGS() << "Loads to pipeline:\n";
    unsigned prefetchBytes = 0;
    for (const PrefetchCandidate &candidate : candidates) {
      Operation *op = candidate.op;
      RankedTensorType tensorType =
          dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
      if (tensorType) {
        ArrayRef<int64_t> shape = tensorType.getShape();
        auto numElems = product<int64_t>(shape);
        prefetchBytes +=
            numElems * tensorType.getElementType().getIntOrFloatBitWidth() / 8;
      }
      DBGS() << "  " << *op << "\n";
    }
    prefetchBytes *= numStages;
    constexpr unsigned BYTES_PER_KB = 1024;
    DBGS() << "Total number of bytes to prefetch: "
           << (prefetchBytes > BYTES_PER_KB
                   ? std::to_string(prefetchBytes / BYTES_PER_KB) + " KB"
                   : std::to_string(prefetchBytes) + " B")
           << " in " << numStages << " stages\n";
  });

  // When the upstream annotation pipeline has run and covers every
  // candidate, drive the prefetch op's stage from upstream's
  // `loop.stage` / `loop.cluster` attrs instead of hardcoding stage 0.
  // Other ops keep the stages Intel picks today.
  bool useAnnotations = annotationsAreUsable(forOp, candidates);
  LDBG("Pipeline path: " << (useAnnotations ? "annotation-driven" : "legacy"));

  // 2. Create the prefetching operations for the loads collected.
  createPrefetchOps(forOp, candidates, useAnnotations);

  // 3. Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages);

  // 4. Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  // Qualify with `::` so unqualified lookup doesn't pick up
  // `mlir::triton::predicateOp` (a different overload declared in
  // PipeliningUtility.h that would crash on `ttig.descriptor_prefetch`).
  options.predicateFn = ::predicateOp;
  options.supportDynamicLoops = true;
  options.annotateFn = [](Operation *op,
                          mlir::scf::PipeliningOption::PipelinerPart part,
                          unsigned iteration) {};

  return true;
}
