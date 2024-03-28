#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

// TODO: We can extra some helpers into common utilities once we add more
// schedules.

/// Replace the yield with a new one with the given operands appended.
static void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  // Fix up the yield op.
  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands().begin(),
                              yieldOp->getOperands().end());
  operands.append(newOperands.begin(), newOperands.end());
  OpBuilder builder(yieldOp);
  builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();
}

namespace {
struct LoadDotOperand {
  LoadDotOperand(tt::LoadOp load,
                 ttg::DotOperandEncodingAttr dotOperandEncoding,
                 bool needTrans = false)
      : load(load), dotOperandEncoding(dotOperandEncoding),
        needTrans(needTrans) {}
  tt::LoadOp load;
  ttg::DotOperandEncodingAttr dotOperandEncoding;
  bool needTrans;
};
} // namespace

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return the encoding. Otherwise return nullptr.
// Negate `needTrans` when a TransOp is seen on the transitive use chain.
static ttg::DotOperandEncodingAttr
allTransitiveUsesHaveDotEncoding(Value val, bool &needTrans) {
  ttg::DotOperandEncodingAttr attr{nullptr};
  for (Operation *user : val.getUsers()) {
    if (user->getNumResults() != 1)
      return nullptr;
    auto tensorType = user->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!tensorType)
      return nullptr;
    if (isa<triton::TransOp>(user))
      needTrans = !needTrans;
    ttg::DotOperandEncodingAttr tempAttr;
    if (tensorType.getEncoding().isa<ttg::SharedEncodingAttr>()) {
      tempAttr =
          allTransitiveUsesHaveDotEncoding(user->getResult(0), needTrans);
    } else if (auto convertLayout =
                   llvm::dyn_cast<ttg::ConvertLayoutOp>(user)) {
      auto tensorType =
          convertLayout.getResult().getType().dyn_cast<RankedTensorType>();
      if (!tensorType)
        return nullptr;
      tempAttr =
          tensorType.getEncoding().dyn_cast<ttg::DotOperandEncodingAttr>();
    } else if (auto dotOp = llvm::dyn_cast<tt::DotOp>(user)) {
      auto tensorType = val.getType().dyn_cast<RankedTensorType>();
      if (!tensorType)
        return nullptr;
      tempAttr =
          tensorType.getEncoding().dyn_cast<ttg::DotOperandEncodingAttr>();
    } else {
      return nullptr;
    }
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return nullptr;
    attr = tempAttr;
  }
  return attr;
}

static void createPrefetchOp(scf::ForOp &forOp, tt::LoadOp loadOp, Value ptr) {
  OpBuilder builder(forOp);
  // Replace the load with load/prefetch in different stage.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp->getLoc();
  auto prefetchOp =
      builder.create<triton::gpu::intel::PrefetchCacheOp>(loc, ptr);
}

/// Create an async load equivalent to the given load.
static void createPrefetchLoad(scf::ForOp &forOp, tt::LoadOp loadOp,
                               Value ptr) {
  createPrefetchOp(forOp, loadOp, ptr);
}

// Return the transitive use of the load which is a dot operand.
static std::optional<LoadDotOperand> loadDotOperand(tt::LoadOp loadOp) {
  bool isCandidate = false;
  if (loadOp.getResult().hasOneUse()) {
    Operation *use = *loadOp.getResult().getUsers().begin();
    if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use)) {
      auto tensorType =
          convertLayout.getResult().getType().cast<RankedTensorType>();
      if (auto sharedEnc =
              tensorType.getEncoding().dyn_cast<ttg::SharedEncodingAttr>()) {
        if (sharedEnc.getHasLeadingOffset()) {
          // MMA V3 case.
          auto newOrder = sharedEnc.getOrder();
          auto ty = loadOp.getType().cast<RankedTensorType>();
          auto oldOrder = ttg::getOrder(ty.getEncoding());
          if (newOrder[0] == oldOrder[0] || newOrder[1] == oldOrder[1]) {
            // The operand of MMAv3 is in SharedEncoding and it's order should
            // not be changed after FuseTranspositions Pass. So we only pipeline
            // the load if the order of the loaded BlockedEncoding is the same
            // as the order of the SharedEncoding it is converted to.
            // TODO: remove this constraint once the LoadOp supports transpose
            // fusion
            //            hasMMAV3 = true;
            return LoadDotOperand(loadOp, nullptr);
          }
        }
      }
    } else if (auto dotOp = llvm::dyn_cast<tt::DotOp>(use)) {
    }
  }
  bool needTrans = false;
  ttg::DotOperandEncodingAttr attr =
      allTransitiveUsesHaveDotEncoding(loadOp.getResult(), needTrans);
  if (!attr)
    return std::nullopt;
  return LoadDotOperand(loadOp, attr, needTrans);
}

/// Collect loads to pipeline. Return success if we can pipeline this loop
static void collectOpsToPipeline(scf::ForOp forOp,
                                 SmallVectorImpl<LoadDotOperand> &ops) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(&op)) {
      bool candidate = false;
      if (isLoadFromTensorPtr(loadOp)) {
        // 2D load/store.
        candidate = true;
      } else {
        auto ptr = loadOp.getPtr();
        unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
        if (auto mask = loadOp.getMask())
          vec =
              std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

        auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
        if (!tensorTy || tensorTy.getRank() < 2)
          continue;
        auto ty =
            tensorTy.getElementType().cast<tt::PointerType>().getPointeeType();
        unsigned width = vec * ty.getIntOrFloatBitWidth();
        // We do not pipeline all loads for the following reasons:
        // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8 and 16.
        // 2. It's likely that pipling small loads won't offer much performance
        //    improvement and may even hurt performance by increasing register
        //    pressure.
        if (width >= 32)
          candidate = true;
      }
      if (!candidate)
        continue;
      std::optional<LoadDotOperand> loadWithDotOperand = loadDotOperand(loadOp);
      if (!loadWithDotOperand.has_value())
        continue;
      ops.push_back(loadWithDotOperand.value());
    }
  }
}

static Value createPrefetch(scf::ForOp &forOp, tt::LoadOp loadOp) {
  OpBuilder builder(forOp);
  auto ty = loadOp.getType().cast<RankedTensorType>();
  if (!loadOp.getResult().hasOneUse())
    return Value();

  //  auto args = forOp.getRegion().getArguments();
  BlockArgument loadPtr = loadOp.getPtr().cast<BlockArgument>();
  //  int argNum = loadPtr.getArgNumber();
  //  Value ptr = forOp.getOpOperandForRegionIterArg(loadPtr).get();
  Value ptr = loadPtr;

  return ptr;
}

static void createPrefetchOps(scf::ForOp &forOp, ArrayRef<LoadDotOperand> loads,
                              int numStages) {
  struct prefetchLoad {
    prefetchLoad(tt::LoadOp load, Value ptr) : load(load), ptr(ptr) {}
    tt::LoadOp load;
    Value ptr;
  };
  int numBuffers = numStages - 1;
  SmallVector<prefetchLoad> prefetchLoads;

  for (const LoadDotOperand &loadOperand : loads) {
    tt::LoadOp loadOp = loadOperand.load;

    //    auto args = forOp.getRegion().getArguments();
    //    BlockArgument loadPtr = loadOp.getPtr().cast<BlockArgument>();
    //    int argNum = loadPtr.getArgNumber();
    //    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    //
    //    auto ops = yield->getOpOperands();
    //    prefetchLoads.emplace_back(loadOp, ops[argNum-
    //    forOp.getNumInductionVars()].get());
    prefetchLoads.emplace_back(loadOp, loadOp.getPtr());
  }

  for (prefetchLoad &prefetchLoad : prefetchLoads) {
    createPrefetchLoad(forOp, prefetchLoad.load, prefetchLoad.ptr);
  }
}

// Combine the current mask with the given predicate.
static Value getPredMask(RewriterBase &rewriter, Type typeLike,
                         Value currentMask, Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
  if (maskType.isa<RankedTensorType>()) {
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
  }
  if (currentMask) {
    mask = rewriter.create<arith::AndIOp>(loc, mask, currentMask);
  }
  return mask;
}

// Function to mask operations during scheduling.
static Operation *predicateOp(RewriterBase &rewriter, Operation *op,
                              Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;
  if (isa<ttg::AsyncCommitGroupOp>(op))
    return op;
  if (isa<ttg::AsyncWaitOp>(op))
    return op;
  if (isa<triton::gpu::intel::PrefetchCacheOp>(op))
    return op;
  if (auto insertOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(insertOp);
    Value mask = getPredMask(rewriter, insertOp.getSrc().getType(),
                             insertOp.getMask(), pred);
    insertOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    rewriter.setInsertionPoint(loadOp);
    Value mask = getPredMask(rewriter, loadOp.getPtr().getType(),
                             loadOp.getMask(), pred);
    loadOp.getMaskMutable().assign(mask);
    return op;
  }

  assert("don't know how to predicate this op" && false);
  return op;
}

static void setWaitNum(Operation *op,
                       mlir::scf::PipeliningOption::PipelinerPart part,
                       unsigned iteration, unsigned numLoadsInStage) {
  if (auto waitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
    waitOp.setNum(numLoadsInStage);
  }
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
    Value v = operand;
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
    Operation *defOp = v.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock()) {
      addDep(defOp, deps, includeArg, filter);
    }
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

// create the schedule for a matmul loop. This is ad hoc based on how we know
// matmul loops should be pipelined and is not a generic scheduler.
static std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp forOp, int numStages) {
  SmallVector<Operation *> prefetchOps;
  SmallVector<Operation *> loadOps;
  //  SmallVector<Operation *> dotOps;
  // Find the prefetch/load ops that will go respectively in stage 0 and stage
  // `numStages - 1`. All the other operations will go in stage `numStages - 1`.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<triton::gpu::intel::PrefetchCacheOp>(op))
      prefetchOps.emplace_back(&op);
    if (isa<tt::LoadOp>(op))
      loadOps.emplace_back(&op);
    //    if (isa<tt::DotOp>(op))
    //      dotOps.emplace_back(&op);
  }
  DenseSet<Operation *> prefetchAndDeps;
  for (Operation *op : prefetchOps) {
    addDep(op, prefetchAndDeps, false);
  }

  // Find depenencies with distance of 1.
  SmallVector<Operation *> distanceOneUsers;
  for (Operation *op : prefetchAndDeps) {
    for (Value operand : op->getOperands()) {
      if (auto arg = operand.dyn_cast<BlockArgument>()) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
          auto yieldOp = op->getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp) {
            distanceOneUsers.push_back(defOp);
          }
        }
      }
    }
  }
  //  // Schedule loads with a distance of 1 in stage 0
  //  for (Operation *op : distanceOneUsers) {
  //    if (isa<tt::LoadOp>(op)) {
  //      addDep(op, prefetchAndDeps, true);
  //    }
  //  }
  // For the rest of the ops we can move then into stage 1 so that they can be
  // closer to their uses.
  DenseSet<Operation *> stage1deps;
  for (Operation *op : distanceOneUsers) {
    //    if (!isa<tt::LoadOp>(op)) {
    addDep(op, stage1deps, true, &prefetchAndDeps);
    //    }
  }

  for (auto &opPair : stage1deps) {
    llvm::outs() << "johnlu stage1deps:" << *opPair << "\n";
    llvm::outs().flush();
  }

  DenseSet<Operation *> loadAndDeps;
  for (Operation *op : loadOps) {
    addDep(op, loadAndDeps, false, &prefetchAndDeps);
  }
  for (auto &opPair : loadAndDeps) {
    llvm::outs() << "johnlu loadAndDeps:" << *opPair << "\n";
    llvm::outs().flush();
  }
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

  for (auto &opPair : schedule) {
    llvm::outs() << "johnlu stage:" << opPair.second << " def:" << *opPair.first
                 << "\n";
    llvm::outs().flush();
  }

  return schedule;
}

bool mlir::triton::gpu::intel::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages, mlir::scf::PipeliningOption &options) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  SmallVector<LoadDotOperand> loads;
  collectOpsToPipeline(forOp, loads);
  if (loads.empty())
    return false;
  // 2. Convert the loads into async loads and create the allocs.
  llvm::outs() << "johnlu here I want to change it to prefetch"
               << "\n";
  llvm::outs().flush();
  createPrefetchOps(forOp, loads, numStages);

  llvm::outs() << "johnlu loop with prefetch:\n" << forOp << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu here to create the schedule for loop"
               << "\n";
  llvm::outs().flush();
  // 3. Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages);

  llvm::outs() << "johnlu here schedule:" << schedule.size() << "\n";
  llvm::outs().flush();
  // 4. Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = predicateOp;
  options.supportDynamicLoops = true;
  unsigned numLoadsInStage = (numStages - 2) * loads.size();
  options.annotateFn =
      [numLoadsInStage](Operation *op,
                        mlir::scf::PipeliningOption::PipelinerPart part,
                        unsigned iteration) {
        return setWaitNum(op, part, iteration, numLoadsInStage);
      };

  return true;
}
