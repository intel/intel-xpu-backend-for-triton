#include "Schedule.h"
#include "include/triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonintelgpu-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace {

/// Represent a candidate load operation which is used by operations that
/// convert its layout to a 'dot' layout (e.g. ttg.convert_layout).
struct LoadDotOperand {
  LoadDotOperand(tt::LoadOp load,
                 ttg::DotOperandEncodingAttr dotOperandEncoding)
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
static void createPrefetchOp(scf::ForOp &forOp, tt::LoadOp loadOp) {
  OpBuilder builder(forOp);
  builder.setInsertionPoint(loadOp);
  auto prefetchOp = builder.create<ttgi::PrefetchOp>(
      loadOp->getLoc(), loadOp.getPtr(), loadOp.getMask(), loadOp.getCache(),
      loadOp.getEvict(), loadOp.getIsVolatile());

  // inherit attributes from the load operation
  auto attrs = loadOp->getAttrDictionary();
  prefetchOp->setAttrs(attrs);
}

/// Create prefetch operations for the given loads.
static void createPrefetchOps(scf::ForOp &forOp,
                              ArrayRef<LoadDotOperand> loads) {
  assert(!loads.empty() && "Expecting at least one load operation");
  for (const LoadDotOperand &loadOperand : loads) {
    tt::LoadOp loadOp = loadOperand.load;
    createPrefetchOp(forOp, loadOp);
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
      // In order to avoid polluting the cache, do not prefetch loads unless the
      // memory they reference is densely structured.
      Attribute blockIOAttr =
          loadOp->getAttr(mlir::triton::gpu::intel::TritonIntelGPUDialect::
                              getBlockIOAttrName());
      if (!blockIOAttr) {
        LDBG("Skipping LoadOp without block_io attribute" << *loadOp);
        continue;
      }

      // Currently we can only prefetch 2D loads.
      if (cast<RankedTensorType>(loadOp.getType()).getRank() != 2) {
        LDBG("Skipping LoadOp with non 2D tensor type" << *loadOp);
        continue;
      }

      std::optional<LoadDotOperand> loadWithDotOperand = loadDotOperand(loadOp);
      if (loadWithDotOperand.has_value())
        loadOps.push_back(loadWithDotOperand.value());
    }
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
        Value ptr = op.getPtr();
        if (mlir::triton::isTensorPointerType(ptr.getType())) {
          // Work around: prefech op with scalar bool is reverted.
          // Block pointer has been protected by boundary.
          return op;
        }
        rewriter.setInsertionPoint(op);
        Value mask = tt::getPredMask(rewriter, op.getPtr().getType(),
                                     op.getMask(), pred);
        op.getMaskMutable().assign(mask);
        return op;
      });
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
    if (isa<ttgi::PrefetchOp>(op))
      prefetchOps.emplace_back(&op);
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      // Loads that are neither tensors nor pointers to tensor are not
      // prefetched and could be used by prefetchOp dependencies
      // (typically `advanceOp`).
      // As prefetchOp dependencies are assigned to stage 0, this type of loads
      // must not be explicitely assigned to stage `numStages - 1`.
      if (mlir::triton::isTensorOrTensorPointerType(loadOp.getPtr().getType()))
        loadOps.emplace_back(&op);
    }
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
  if (loads.empty()) {
    LDBG("No loads to pipeline");
    return false;
  }

  LLVM_DEBUG({
    DBGS() << "Loads to pipeline:\n";
    unsigned prefetchBytes = 0;
    for (LoadDotOperand &load : loads) {
      tt::LoadOp &op = load.load;
      if (auto tensorType =
              dyn_cast<RankedTensorType>(op.getResult().getType())) {
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
