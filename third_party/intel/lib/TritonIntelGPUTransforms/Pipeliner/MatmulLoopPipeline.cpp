#include "Schedule.h"
#include "include/triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

// TODO: We can extra some helpers into common utilities once we add more
// schedules.

namespace {

struct LoadDotOperand {
  LoadDotOperand(tt::LoadOp load,
                 ttg::DotOperandEncodingAttr dotOperandEncoding,
                 bool needTrans = false)
      : load(load), dotOperandEncoding(dotOperandEncoding) {}
  tt::LoadOp load;
  ttg::DotOperandEncodingAttr dotOperandEncoding;
};

} // namespace

/// Replace the ForOp's yield with a new one with the given operands appended.
static void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  assert(!newOperands.empty() && "Expecting at least one operand");

  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands().begin(),
                              yieldOp->getOperands().end());
  operands.append(newOperands.begin(), newOperands.end());

  OpBuilder builder(yieldOp);
  builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();
}

static ttg::DotOperandEncodingAttr allTransitiveUsesHaveDotEncoding(Value val);

static ttg::DotOperandEncodingAttr getEncodingFromUser(Operation *user,
                                                       Value val) {
  assert(user->getNumResults() == 1 && "Expecting a single result");

  OpResult res = user->getResult(0);
  auto tensorType = dyn_cast<RankedTensorType>(res.getType());
  if (!tensorType)
    return nullptr;

  if (isa<ttg::SharedEncodingAttr>(tensorType.getEncoding())) {
    return allTransitiveUsesHaveDotEncoding(res);
  } else if (auto convertLayout = dyn_cast<ttg::ConvertLayoutOp>(user)) {
    auto tensorType =
        dyn_cast<RankedTensorType>(convertLayout.getResult().getType());
    if (!tensorType)
      return nullptr;
    return dyn_cast<ttg::DotOperandEncodingAttr>(tensorType.getEncoding());
  } else if (auto dotOp = dyn_cast<tt::DotOp>(user)) {
    auto tensorType = dyn_cast<RankedTensorType>(val.getType());
    if (!tensorType)
      return nullptr;
    return dyn_cast<ttg::DotOperandEncodingAttr>(tensorType.getEncoding());
  }
  return nullptr;
}

/// If all the transitive uses of the given value have are used by a convert to
/// the same dot operand encoding, return the encoding. Otherwise return
/// nullptr.
static ttg::DotOperandEncodingAttr allTransitiveUsesHaveDotEncoding(Value val) {
  ttg::DotOperandEncodingAttr attr{nullptr};
  for (Operation *user : val.getUsers()) {
    if (user->getNumResults() != 1)
      return nullptr;

    ttg::DotOperandEncodingAttr tempAttr = getEncodingFromUser(user, val);
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return nullptr;
    attr = tempAttr;
  }
  return attr;
}

/// Create a prefetch operation for the given load operation.
static void createPrefetchOp(scf::ForOp &forOp, tt::LoadOp loadOp, Value ptr) {
  OpBuilder builder(forOp);
  builder.setInsertionPoint(loadOp);
  builder.create<ttgi::PrefetchOp>(loadOp->getLoc(), ptr, loadOp.getCache(),
                                   loadOp.getEvict(), loadOp.getIsVolatile());
}

/// Create prefetch operations for the given loads.
static void createPrefetchOps(scf::ForOp &forOp,
                              ArrayRef<LoadDotOperand> loads) {
  assert(!loads.empty() && "Expecting at least one load operation");
  for (const LoadDotOperand &loadOperand : loads) {
    tt::LoadOp loadOp = loadOperand.load;
    createPrefetchOp(forOp, loadOp, loadOp.getPtr());
  }
}

/// Return the transitive use of the load which is a dot operand.
static std::optional<LoadDotOperand> loadDotOperand(tt::LoadOp loadOp) {
  if (ttg::DotOperandEncodingAttr attr =
          allTransitiveUsesHaveDotEncoding(loadOp.getResult()))
    return LoadDotOperand(loadOp, attr);
  return std::nullopt;
}

/// Collect loads to pipeline. Return success if we can pipeline this loop.
static void collectOpsToPipeline(scf::ForOp forOp,
                                 SmallVectorImpl<LoadDotOperand> &loadOps) {
  assert(loadOps.empty() && "Expecting an empty list of load operations");

  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  mlir::triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block.
  for (Operation &op : forOp) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(&op)) {
      std::optional<LoadDotOperand> loadWithDotOperand = loadDotOperand(loadOp);
      if (loadWithDotOperand.has_value())
        loadOps.push_back(loadWithDotOperand.value());
    }
  }
}

/// Combine the current mask with the given predicate.
static Value getPredMask(RewriterBase &rewriter, Type typeLike,
                         Value currentMask, Value pred) {
  Location loc = pred.getLoc();
  Value mask = pred;
  Type maskType = tt::getI1SameShape(typeLike);

  if (isa<RankedTensorType>(maskType))
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);

  return currentMask ? rewriter.create<arith::AndIOp>(loc, mask, currentMask)
                     : mask;
}

/// Function to mask operations during scheduling.
static Operation *predicateOp(RewriterBase &rewriter, Operation *op,
                              Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op) || isa<ttgi::PrefetchOp>(op))
    return op;

  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    rewriter.setInsertionPoint(loadOp);
    Value mask = getPredMask(rewriter, loadOp.getPtr().getType(),
                             loadOp.getMask(), pred);
    loadOp.getMaskMutable().assign(mask);
    return op;
  }

  llvm_unreachable("don't know how to predicate this operation");
}

/// Helper to get the defining operation of a value.
static Operation *getDefOp(Value v, Operation *op, bool includeArg) {
  llvm::SmallDenseSet<Value> seen;
  while (auto arg = v.dyn_cast<BlockArgument>()) {
    if (!includeArg)
      break;
    if (!seen.insert(v).second)
      break;
    if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
      auto yieldOp = op->getBlock()->getTerminator();
      v = yieldOp->getOperand(arg.getArgNumber() - 1);
      continue;
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

// Add operations to the shedule with the given stage based on the filter
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
    if (isa<ttgi::PrefetchOp>(op))
      prefetchOps.emplace_back(&op);
    if (isa<tt::LoadOp>(op))
      loadOps.emplace_back(&op);
  }

  DenseSet<Operation *> prefetchAndDeps;
  for (Operation *op : prefetchOps)
    addDep(op, prefetchAndDeps, false);

  // Find depenencies with distance of 1.
  SmallVector<Operation *> distanceOneUsers;
  for (Operation *op : prefetchAndDeps) {
    for (Value operand : op->getOperands()) {
#if 1
      Operation *defOp = getDefOp(operand, op, true);
      if (defOp)
        distanceOneUsers.push_back(defOp);
#else
      if (auto arg = operand.dyn_cast<BlockArgument>()) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
          auto yieldOp = op->getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp)
            distanceOneUsers.push_back(defOp);
        }
      }
#endif
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
  addOps(forOp, 0, schedule,
         [&](Operation *op) { return prefetchAndDeps.count(op); });

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
  SmallVector<LoadDotOperand> loads;
  collectOpsToPipeline(forOp, loads);
  if (loads.empty())
    return false;

  // 2. Create the prefetching operations for the loads collected.
  createPrefetchOps(forOp, loads);

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
  options.predicateFn = predicateOp;
  options.supportDynamicLoops = true;
  options.annotateFn = [](Operation *op,
                          mlir::scf::PipeliningOption::PipelinerPart part,
                          unsigned iteration) {};

  return true;
}
