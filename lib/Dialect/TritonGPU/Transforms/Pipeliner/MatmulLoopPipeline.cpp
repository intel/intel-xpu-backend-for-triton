#include "PipelineExpander.h"
#include "PipeliningUtility.h"
#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-matmul-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// TODO: We can extra some helpers into common utilities once we add more
// schedules.

namespace {

struct PipelinedOpInfo {
  int stage = -1;
  Operation *use = nullptr;

  // Layout of the data in the shared memory.
  ttg::SharedEncodingAttr sharedEncoding = nullptr;
  // Blocked encoding is used for loads feeding into other loads.
  ttg::BlockedEncodingAttr blockedEncoding = nullptr;
  bool loadIsMMAV3 = false;
};

} // namespace

// Replace the ForOp's yield with a new one with the given operands appended.
static void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  // Fix up the yield op.
  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands());
  operands.append(newOperands.begin(), newOperands.end());

  OpBuilder builder(yieldOp);
  builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();
}

static void
createAsyncCopy(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                Value insertIdx, Value extractIdx,
                llvm::MapVector<Operation *, PipelinedOpInfo> &opToInfo) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  if (!isExpensiveLoadOrStore(loadOp) && opToInfo[loadOp].blockedEncoding) {
    // For inexpensive loads that do not directly feed into dot ops
    // we want to use optimal layout for the data.
    ttg::BlockedEncodingAttr encoding = opToInfo[loadOp].blockedEncoding;
    auto convertBlockLayout = [&](Value src) {
      auto ty = src.getType().cast<RankedTensorType>();
      auto newTy =
          RankedTensorType::get(ty.getShape(), ty.getElementType(), encoding);
      auto cvt =
          builder.create<ttg::ConvertLayoutOp>(loadOp->getLoc(), newTy, src);
      return cvt.getResult();
    };
    src = convertBlockLayout(src);
    mask = convertBlockLayout(src);
  }

  SmallVector<Value> copyOffsets = {insertIdx, zero, zero};
  tt::MemDescType allocTy = alloc.getType().cast<tt::MemDescType>();
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), /*mutableMemory=*/true);
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, src, view, loadOp.getMask(), loadOp.getOther(), loadOp.getCache(),
      loadOp.getEvict(), loadOp.getIsVolatile());
  Operation *commmit =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));
  builder.create<ttg::AsyncWaitOp>(loc, commmit->getResult(0), 0);

  int stage = opToInfo[loadOp].stage;
  bool isMMV3Load = opToInfo[loadOp].loadIsMMAV3;
  opToInfo.insert({copy, {.stage = stage}});
  opToInfo.insert({commmit, {.stage = stage}});
  opToInfo.erase(loadOp);

  // Extract part.
  SmallVector<Value> loadOffsets = {extractIdx, zero, zero};
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  if (isMMV3Load) {
    auto alloc = cast<ttg::LocalAllocOp>((*loadOp->getUsers().begin()));
    alloc.replaceAllUsesWith(viewLoad.getResult());
    alloc.erase();
  } else {
    for (Operation *user : loadOp->getUsers()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        alloc.replaceAllUsesWith(viewLoad.getResult());
        alloc.erase();
      }
    }
    auto sharedLoad =
        builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad);
    loadOp->replaceAllUsesWith(sharedLoad->getResults());
  }
  loadOp.erase();
}

/// Create an async load equivalent to the given load.
static void
createAsyncLoad(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                Value insertIdx, Value extractIdx, Value phase,
                llvm::MapVector<Operation *, PipelinedOpInfo> &opToInfo) {
  createAsyncCopy(forOp, loadOp, alloc, insertIdx, extractIdx, opToInfo);
}

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and set the shared encoding that
// needs to be used to be compatible with users' layouts.
//
// TODO: Rename, because the name only tells us half the story: We check for all
// users having a dot encoding, but then we return a shared encoding, which is
// surprising given the name.
static std::optional<ttg::SharedEncodingAttr>
allTransitiveUsesHaveDotEncoding(Value val) {
  ttg::SharedEncodingAttr attr;
  for (Operation *user : val.getUsers()) {
    ttg::SharedEncodingAttr tempAttr;
    if (user->getNumResults() != 1)
      return std::nullopt;
    if (auto memDesc =
            user->getResult(0).getType().dyn_cast<triton::MemDescType>()) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      if (!tempAttr)
        tempAttr = memDesc.getEncoding().cast<ttg::SharedEncodingAttr>();
      if (!allTransitiveUsesHaveDotEncoding(user->getResult(0)).has_value())
        return std::nullopt;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return std::nullopt;
      auto dotOpEnc = user->getResult(0)
                          .getType()
                          .cast<TensorOrMemDesc>()
                          .getEncoding()
                          .dyn_cast<ttg::DotOperandEncodingAttr>();
      if (!dotOpEnc)
        return std::nullopt;
      auto srcTy = val.getType().cast<TensorOrMemDesc>();
      auto CTALayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = ttg::getOrder(srcTy.getEncoding());
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
      tempAttr = ttg::SharedEncodingAttr::get(
          val.getContext(), dotOpEnc, srcTy.getShape(),
          ttg::getOrder(srcTy.getEncoding()),
          ttg::getCTALayout(srcTy.getEncoding()),
          srcTy.getElementType().getIntOrFloatBitWidth(), /*needTrans=*/false);
    }
    // Check that the shared encodings needed by the users are compatible.
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return std::nullopt;
    attr = tempAttr;
  }
  return attr;
}

// TODO: This returns true and *sometimes* sets enc?
bool loadDotOperand(tt::LoadOp loadOp, bool &hasMMAV3,
                    ttg::SharedEncodingAttr &enc) {
  if (loadOp.getResult().hasOneUse()) {
    Operation *use = *loadOp.getResult().getUsers().begin();
    if (auto alloc = llvm::dyn_cast<ttg::LocalAllocOp>(use)) {
      auto sharedEnc =
          alloc.getType().getEncoding().cast<ttg::SharedEncodingAttr>();
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
          hasMMAV3 = true;
          return true;
        }
      }
    }
  }

  std::optional<ttg::SharedEncodingAttr> sharedEnc =
      allTransitiveUsesHaveDotEncoding(loadOp.getResult());
  if (!sharedEnc.has_value()) {
    return false;
  }
  enc = *sharedEnc;
  return true;
}

static ttg::BlockedEncodingAttr
getBlockedEncoding(tt::LoadOp loadOp, tt::ModuleAxisInfoAnalysis &axisInfo) {
  Value src = loadOp.getPtr();
  auto ty = src.getType().cast<RankedTensorType>();
  auto mod = loadOp->getParentOfType<ModuleOp>();
  int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  tt::AxisInfo::DimVectorT contiguity =
      axisInfo.getAxisInfo(src)->getContiguity();
  SmallVector<unsigned> order = argSort(contiguity);
  unsigned currPerThread = getNumElementsPerThread(loadOp, order, axisInfo);
  SmallVector<unsigned> sizePerThread(order.size(), 1);
  sizePerThread[order[0]] = currPerThread;
  ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(ty.getEncoding());
  return ttg::BlockedEncodingAttr::get(loadOp->getContext(), ty.getShape(),
                                       sizePerThread, order, numWarps,
                                       threadsPerWarp, ctaLayout);
}

static ttg::SharedEncodingAttr getSharedEncoding(tt::LoadOp loadOp,
                                                 Operation *use, bool isMMAV3) {

  auto ty = loadOp.getType().cast<RankedTensorType>();
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto blockedOrder = ttg::getOrder(ty.getEncoding());
  SmallVector<unsigned> order;
  if (blockedOrder.size() == 3) {
    for (unsigned i = 0; i < blockedOrder.size(); ++i) {
      if (blockedOrder[i] == 0)
        continue;
      order.push_back(blockedOrder[i]);
    }
    order.push_back(0);
  } else {
    order = blockedOrder;
  }
  if (isa<tt::DotOp>(use)) {
    assert(isMMAV3 &&
           "Load used by dot op should be either MMAv3 or have a "
           "shared encoding already picked based on users' layouts.");
    return ttg::SharedEncodingAttr::get(ty.getContext(), ty.getShape(), order,
                                        ctaLayout, ty.getElementType());
  } else {
    assert(!isMMAV3 && "Load used by non-dot op should not be MMAv3.");
    // Use non-swizzled layout for loads that do not feed into dot ops.
    // TODO: This won't be optimal for 2D tensors.
    return ttg::SharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                        ctaLayout);
  }
}

// Create a map from load ops to their distance to the nearest dot op and the
// final use of the load op (another load op, or a dot op).
static llvm::MapVector<tt::LoadOp, std::pair<int, Operation *>>
loadOpsToDistanceAndUse(scf::ForOp forOp) {
  llvm::MapVector<tt::LoadOp, std::pair<int, Operation *>> loadOpToDistAndUse;
  DenseSet<Operation *> seen;

  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  auto isCandidate = [&](tt::LoadOp loadOp) -> bool {
    assert(!isLoadFromTensorPtr(loadOp) &&
           "Block ptr should have been lowered before this pass.");
    auto ptr = loadOp.getPtr();
    unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
    if (auto mask = loadOp.getMask())
      vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

    auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return false;
    auto ty =
        tensorTy.getElementType().cast<tt::PointerType>().getPointeeType();
    unsigned width = vec * ty.getIntOrFloatBitWidth();

    // We do not pipeline all loads for the following reasons:
    // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8, or 16.
    // 2. It's likely that pipling small loads won't offer much performance
    //    improvement and may even hurt performance by increasing register
    //    pressure.
    LDBG("Load " << *loadOp << " has width " << width);
    return width >= 32;
  };

  std::function<void(Operation * op, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        if (!seen.insert(op).second)
          return;
        if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
          if (!isCandidate(loadOp))
            return;
          // TODO: What if there are multiple uses at different distances?
          loadOpToDistAndUse[loadOp] = std::make_pair(distance, use);
          use = op;
          distance++;
        }
        for (Value operand : op->getOperands()) {
          Value v = operand;
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, distance, use);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isa<tt::DotOp>(op))
      continue;
    dfs(&op, 0, &op);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (forOp->hasAttr(tt::kNumStagesAttrName)) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp>(op))
        dfs(&op, 0, &op);
    }
  }

  return loadOpToDistAndUse;
}

/// Collect loads to pipeline. Returns true if loads are found to pipeline.
static bool
collectOpsToPipeline(scf::ForOp forOp,
                     llvm::MapVector<Operation *, PipelinedOpInfo> &opInfo,
                     int numStages, bool &hasMMAV3) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // Loads ordered by their dependency distance to the nearest dot op.
  llvm::MapVector<tt::LoadOp, std::pair<int, Operation *>> loadOpToDistAndUse =
      loadOpsToDistanceAndUse(forOp);
  LLVM_DEBUG({
    DBGS() << "Found " << loadOpToDistAndUse.size() << " loads to pipeline:\n";
    for (const auto &[k, v] : loadOpToDistAndUse) {
      DBGS() << "  " << *k << " distance=" << v.first << " use=" << *v.second
             << "\n";
    }
  });
  if (loadOpToDistAndUse.empty())
    return false;

  int maxDistance = -1;
  for (auto &[op, distAndUse] : loadOpToDistAndUse) {
    if (distAndUse.first > maxDistance) {
      maxDistance = distAndUse.first;
    }
  }
  assert(maxDistance >= 0);

  // Start by initializing PipelinedOpInfo for users of the loads.
  for (auto &[loadOp, distAndUse] : loadOpToDistAndUse)
    opInfo[distAndUse.second] = PipelinedOpInfo();

  unsigned stagesBetweenLoads = ceil<unsigned>(numStages - 2, maxDistance + 1);

  // Then consider the load ops that feed into the dot ops or are used by other
  // loads.
  for (auto &[loadOp, distAndUse] : loadOpToDistAndUse) {
    PipelinedOpInfo loadInfo;
    bool loadIsMMAV3 = false;
    if (isa<tt::DotOp>(distAndUse.second)) {
      ttg::SharedEncodingAttr sharedEnc;
      bool isLoadDotOperand = loadDotOperand(loadOp, loadIsMMAV3, sharedEnc);
      hasMMAV3 |= loadIsMMAV3;
      if (!isLoadDotOperand)
        continue;
      loadInfo.sharedEncoding = sharedEnc;
    } else {
      loadInfo.blockedEncoding = getBlockedEncoding(loadOp, axisInfoAnalysis);
    }
    // If we haven't already assigned a layout do it now.
    if (!loadInfo.sharedEncoding)
      loadInfo.sharedEncoding =
          getSharedEncoding(loadOp, distAndUse.second, loadIsMMAV3);
    loadInfo.loadIsMMAV3 = loadIsMMAV3;
    int stage = (maxDistance - distAndUse.first) * stagesBetweenLoads;
    loadInfo.stage = stage;
    loadInfo.use = distAndUse.second;
    opInfo[loadOp] = loadInfo;
  }

  // Last, find root load users (i.e. users that aren't used by another stage of
  // the pipeline) and assign them to the last stage.
  //
  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp) {
    auto iter = opInfo.find(&op);
    if (iter != opInfo.end() && iter->second.stage == -1) {
      assert(!isa<tt::LoadOp>(iter->first));
      iter->second.stage = numStages - 1;
    }
  }

  return true;
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, tt::LoadOp loadOp,
                         ttg::SharedEncodingAttr sharedEnc, unsigned distance) {
  OpBuilder builder(forOp);
  auto ty = loadOp.getType().cast<RankedTensorType>();
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type memdescType = mlir::triton::MemDescType::get(
      bufferShape, ty.getElementType(), sharedEnc, /*mutableMemory*/ true);
  Value alloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loadOp.getLoc(), memdescType, Value());
  return alloc;
}

// Convert load ops into their asyn version and apply multi-buffering based on
// the required number of buffers.
static SmallVector<Value>
createAsynOps(scf::ForOp &forOp,
              llvm::MapVector<Operation *, PipelinedOpInfo> &opToInfo,
              int numBuffers, bool hasMMAV3) {
  struct AsyncLoad {
    AsyncLoad(tt::LoadOp loadOp, Value alloc) : loadOp(loadOp), alloc(alloc) {}
    tt::LoadOp loadOp;
    Value alloc;
  };
  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> allocs;
  for (auto &[op, info] : opToInfo) {
    if (tt::LoadOp loadOp = dyn_cast<tt::LoadOp>(op)) {
      assert(info.sharedEncoding && "LoadOp shared encoding not defined.");
      Value alloc = createAlloc(forOp, loadOp, info.sharedEncoding, numBuffers);
      assert(alloc && "Failed to create alloc for the async load.");
      allocs.push_back(alloc);
      asyncLoads.emplace_back(loadOp, alloc);
    }
  }

  IRRewriter builder(forOp.getContext());
  builder.setInsertionPoint(forOp);

  Location loc = forOp.getLoc();
  // Create two new counters to index into the allocs.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value insertIdx = minusOne;
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
  SmallVector<Value> newOperands;
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  Value phase;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;
  insertIdx = newForOp.getBody()->getArgument(newOperandIndex);
  extractIdx = newForOp.getBody()->getArgument(newOperandIndex + 1);

  // Create two counters for the insert and extract indices to avoid creating
  // long liverange.
  builder.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
  insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
  Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               insertIdx, numBuffersVal);
  insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);

  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  for (AsyncLoad &asyncLoad : asyncLoads) {
    createAsyncLoad(forOp, asyncLoad.loadOp, asyncLoad.alloc, insertIdx,
                    extractIdx, phase, opToInfo);
  }
  SmallVector<Value> newYieldOperands = {insertIdx, extractIdx};
  // Patch the yield with the updated counters.
  appendToYield(forOp, newYieldOperands);

  return allocs;
}

static void
printSchedule(std::vector<std::pair<Operation *, unsigned>> &schedule,
              int numStages) {
  LDBG("Schedule:");
  for (int i = 0; i < numStages; i++) {
    LDBG("- Stage " << i);
    for (auto &pair : schedule) {
      if (pair.second == i) {
        pair.first->dump();
      }
    }
  }
}

static bool
isScheduleValid(scf::ForOp forOp,
                std::vector<std::pair<Operation *, unsigned>> &schedule) {
  DenseSet<Operation *> seen;
  for (auto &pair : schedule) {
    if (!seen.insert(pair.first).second)
      return false;
  }
  auto loopBody = forOp.getBody()->without_terminator();
  auto numOps = std::distance(loopBody.begin(), loopBody.end());
  if (seen.size() != numOps)
    return false;
  return true;
}

// Create the schedule for a matmul loop. This is ad hoc based on how we know
// matmul loops should be pipelined and is not a generic scheduler.
static std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp forOp, int numStages,
               llvm::MapVector<Operation *, PipelinedOpInfo> &opToInfo,
               bool prefetchExtract) {
  LLVM_DEBUG({
    LDBG("For loop:");
    forOp.dump();

    LDBG("Initial schedule:");
    for (int i = 0; i < numStages; i++) {
      LDBG("- Ops in stage " << i);
      for (auto &[op, info] : opToInfo) {
        if (i == info.stage) {
          op->dump();
        }
      }
    }
  });

  SmallVector<Operation *> extractOps;
  // Find the insert/extract ops that will go respectively in stage 0 and stage
  // `numStages - 2`. All the other operations will go in stage `numStages - 1`.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (prefetchExtract) {
      if (isa<ttg::MemDescSubviewOp, ttg::AsyncWaitOp>(op))
        extractOps.push_back(&op);
    }
  }

  auto printDenseSet = [](DenseSet<Operation *> &set) {
    for (auto op : set) {
      op->dump();
    }
  };

  SmallVector<DenseSet<Operation *>> insertOps(numStages);
  for (auto &[op, info] : opToInfo) {
    if (isa<ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp>(op)) {
      insertOps[info.stage].insert(op);
    }
  }

  // Inserts and dependencies grouped by stage.
  SmallVector<DenseSet<Operation *>> insertAndDeps(numStages);
  DenseSet<Operation *> seen;
  for (int stage = 0; stage < numStages; stage++) {
    for (Operation *op : insertOps[stage]) {
      tt::addDep(op, insertAndDeps[stage], false, &seen);
      seen.insert(insertAndDeps[stage].begin(), insertAndDeps[stage].end());
    }
  }

  LLVM_DEBUG({
    for (int stage = 0; stage < numStages; stage++) {
      LDBG("- insertAndDeps " << stage);
      printDenseSet(insertAndDeps[stage]);
    }
  });

  // Find dependencies with distance of 1.
  SmallVector<DenseSet<Operation *>> distanceOneUsers(numStages);
  for (int stage = 0; stage < numStages - 1; stage++) {
    auto &group = insertAndDeps[stage];
    for (Operation *op : group) {
      for (Value operand : op->getOperands()) {
        if (auto arg = operand.dyn_cast<BlockArgument>()) {
          if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
            auto yieldOp = op->getBlock()->getTerminator();
            Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
            Operation *defOp = v.getDefiningOp();
            if (defOp && group.count(defOp) == 0) {
              // Schedule distance 1 users in the next stage.
              distanceOneUsers[stage].insert(defOp);
            }
          }
        }
      }
    }
    LLVM_DEBUG({
      LDBG("- distanceOneUsers " << stage);
      printDenseSet(distanceOneUsers[stage]);
    });
  }

  // Schedule loads with a distance of 1 together with the insert ops.
  for (unsigned i = 0; i < distanceOneUsers.size(); i++) {
    for (auto op : distanceOneUsers[i]) {
      if (isa<tt::LoadOp>(op))
        tt::addDep(op, insertAndDeps[i], true);
    }
  }

  DenseSet<Operation *> allInsertAndDeps;
  for (auto &set : insertAndDeps) {
    allInsertAndDeps.insert(set.begin(), set.end());
  }

  // Rest of the distance 1 dependencies will be scheduled one
  // stage after the insert ops.
  SmallVector<DenseSet<Operation *>> stage1deps(numStages);
  for (unsigned i = 0; i < distanceOneUsers.size() - 1; i++) {
    auto &group = distanceOneUsers[i];
    for (auto op : group) {
      if (!isa<tt::LoadOp>(op))
        tt::addDep(op, stage1deps[i + 1], true, &allInsertAndDeps);
    }
    LLVM_DEBUG({
      LDBG("- stage1deps " << i);
      printDenseSet(stage1deps[i]);
    });
  }

  DenseSet<Operation *> allStage1Deps;
  for (auto &set : stage1deps) {
    allStage1Deps.insert(set.begin(), set.end());
  }

  DenseSet<Operation *> extractAndDeps;
  for (Operation *op : extractOps) {
    tt::addDep(op, extractAndDeps, true, &allInsertAndDeps);
  }
  LLVM_DEBUG({
    LDBG("- extractAndDeps:");
    printDenseSet(extractAndDeps);
  });

  std::vector<std::pair<Operation *, unsigned>> schedule;
  // Schedule stage `numStage - 1` first.
  tt::addOps(forOp, numStages - 1, schedule, [&](Operation *op) {
    return allInsertAndDeps.count(op) == 0 && allStage1Deps.count(op) == 0 &&
           extractAndDeps.count(op) == 0;
  });

  // Schedule some dependencies with distance of 1 into stage 1 to reduce
  // pressure.
  // Insert the ops in the reverse order of the stages. This helps with saving
  // the number of required buffers.
  for (int i = numStages - 1; i >= 0; i--) {
    auto &group = stage1deps[i];
    tt::addOps(forOp, i, schedule,
               [&](Operation *op) { return group.count(op); });
  }

  for (int i = numStages - 1; i >= 0; i--) {
    auto &group = insertAndDeps[i];
    tt::addOps(forOp, i, schedule,
               [&](Operation *op) { return group.count(op); });
  }

  // Finally schedule the extract ops in stage `numStage - 2` so that they get
  // pre-fetched and play well with pretech pass.
  tt::addOps(forOp, numStages - 2, schedule,
             [&](Operation *op) { return extractAndDeps.count(op); });

  LLVM_DEBUG(printSchedule(schedule, numStages));
  assert(isScheduleValid(forOp, schedule) && "Invalid schedule.");

  return schedule;
}

constexpr static char kNeedWaitAttrName[] = "triton.pipeline.needs_wait";

bool mlir::triton::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  llvm::MapVector<Operation *, PipelinedOpInfo> opToInfo;
  bool hasMMAV3 = false;
  if (!collectOpsToPipeline(forOp, opToInfo, numStages, hasMMAV3))
    return false;

  // Calculate the number of buffers needed for each load.
  // TODO pawel: we could do more fine-grained allocation here and
  // allocate only the number of buffers that specific loads need.
  // Instead, we allocate the maximum number of buffers needed by any load.
  int maxNumBuffers = -1;
  for (auto &[op, info] : opToInfo) {
    if (!isa<tt::LoadOp>(op))
      continue;
    assert(info.stage != -1 && "LoadOp stage not defined");
    assert(info.use && "LoadOp use not defined");
    assert(opToInfo.count(info.use) && "Use not in opToInfo");

    int defStage = info.stage;
    int useStage = opToInfo[info.use].stage;
    int numBuffers = useStage - defStage;

    if (hasMMAV3 && isa<tt::DotOp>(info.use)) {
      // For MMAv3, we need an extra buffer as this is assumed in the wgmma
      // pipelining post-processing.
      numBuffers++;
    }
    if (numBuffers > maxNumBuffers)
      maxNumBuffers = numBuffers;
  }

  // 2. Convert the loads into async loads and create the allocs.
  SmallVector<Value> allocs =
      createAsynOps(forOp, opToInfo, maxNumBuffers, hasMMAV3);

  // 3. Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages, opToInfo, /*prefetchExtract=*/!hasMMAV3);

  // 4. Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = tt::predicateOp;
  options.supportDynamicLoops = true;
  options.annotateFn = [](Operation *op,
                          mlir::triton::PipeliningOption::PipelinerPart part,
                          unsigned iteration) {};
  // Insert a wait 0 after the loop
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), ValueRange({}), 0);
  // Explicitly deallocate allocated tensors after the wait op
  for (auto alloc : allocs)
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return true;
}

/// Find the minimum number of async_commit_group ops between the wait
/// and the associated async_commit_group. This can be safely used as the wait
/// number.
static int minNumInterleavedCommitOps(Operation *waitOp) {
  auto countCommitsBetween = [](Operation *op1, Operation *op2) {
    int count = 0;
    for (auto op = op1; op != op2; op = op->getNextNode()) {
      if (isa<ttg::AsyncCommitGroupOp>(op))
        count++;
      // Intentionally skip block ops' children. This will give us
      // convervatively low number of insert ops.
    }
    return count;
  };

  int minCommitNumber = INT_MAX;

  // DFS the def chain of the extract op to find the insert op. On each path
  // we calculate the number of async_commit. Then we select the minimum number
  // of async_commit ops among all the paths.
  std::function<int(Value, Operation *, int)> minOverHistories =
      [&](Value val, Operation *sinkOp, int thisHistorySum) -> int {
    if (Operation *defOp = val.getDefiningOp()) {
      thisHistorySum += countCommitsBetween(defOp->getNextNode(), sinkOp);
      minCommitNumber = std::min(minCommitNumber, thisHistorySum);
      return minCommitNumber;
    }
    if (auto arg = val.dyn_cast<BlockArgument>()) {
      Block *block = arg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

      // Failed to track, return 0 conservatively.
      if (!forOp)
        return 0;

      Operation *firstForInst = &*forOp.getBody()->begin();
      int insertsBetween = countCommitsBetween(firstForInst, sinkOp);
      thisHistorySum += insertsBetween;
      if (thisHistorySum >= minCommitNumber)
        return minCommitNumber;

      // get the value value assigned to the argument coming from outside the
      // loop
      Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
      int min1 = minOverHistories(incomingVal, forOp, thisHistorySum);

      // get the value value assigned to the argument coming from the previous
      // iteration
      Operation *yieldOp = block->getTerminator();
      Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
      int min2 = minOverHistories(prevVal, yieldOp, thisHistorySum);
      return std::min(std::min(min1, min2), minCommitNumber);
    }
    // Failed to track, return 0 conservatively.
    return 0;
  };

  if (waitOp->getNumOperands() != 1)
    return 0;
  int minCommits = minOverHistories(waitOp->getOperand(0), waitOp, 0);
  return minCommits;
}

// Look for consecutive wait ops and combine them into a single wait op.
static void
combineRedundantWaitOps(llvm::SmallSetVector<ttg::AsyncWaitOp, 8> &waitOps) {
  llvm::SmallSetVector<ttg::AsyncWaitOp, 8> toDelete;
  for (auto waitOp : waitOps) {
    if (toDelete.count(waitOp))
      continue;
    SmallVector<ttg::AsyncWaitOp> waitGroup = {waitOp};
    SmallVector<Value> depTokens;
    unsigned minWaitNumber = waitOp.getNum();
    Operation *next = waitOp->getNextNode();
    while (next && isa<ttg::MemDescSubviewOp, ttg::AsyncWaitOp>(next)) {
      if (auto nextWait = dyn_cast<ttg::AsyncWaitOp>(next)) {
        waitGroup.push_back(nextWait);
        minWaitNumber = std::min(minWaitNumber, nextWait.getNum());
        depTokens.append(nextWait.getOperands().begin(),
                         nextWait.getOperands().end());
      }
      next = next->getNextNode();
    }
    if (waitGroup.size() == 1)
      continue;
    OpBuilder builder(waitGroup.back());
    auto newWaitOp = builder.create<ttg::AsyncWaitOp>(waitOp.getLoc(),
                                                      depTokens, minWaitNumber);
    toDelete.insert(waitGroup.begin(), waitGroup.end());
  }
  for (auto waitOp : toDelete) {
    waitOp->erase();
  }
}

/// Update wait op number by analyzing the number of async_commit_group ops
/// along all paths.
void mlir::triton::updateWaits(ModuleOp module) {
  llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
  module.walk([&](ttg::AsyncWaitOp waitOp) {
    int minNumCommits = minNumInterleavedCommitOps(waitOp);
    waitOp.setNum(minNumCommits);
    waitOps.insert(waitOp);
  });
  combineRedundantWaitOps(waitOps);
}

// Tries to cast op to a tt::DotOp, if it's an MMAv3 (i.e. wgmma / hopper) dot.
//
// Note that this returns nullptr for ttng::DotAsyncOps, but we can't call this
// "dynCastMMAv3SyncDot", because this is sometimes used before we've converted
// eligible tt::DotOps to ttng::DotAsyncOp.
tt::DotOp dynCastMMAv3Dot(Operation *op) {
  if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
    auto resEnc =
        dotOp.getType().getEncoding().dyn_cast<ttg::NvidiaMmaEncodingAttr>();
    if (resEnc && resEnc.isHopper()) {
      return dotOp;
    }
  }
  return nullptr;
}

// Determines whether a tt.dot op can be transformed into ttng.dot.async.  The
// dot can be made async if all of the following are true.
//
//  0. The operation is an MMAv3 dot.
//
//  1. All operands that touch shared memory are multi-buffered, i.e. can't read
//     an incomplete value while it's being written asynchronously by a load.
//
//  2. During iteration i, nothing other than the loop's `yield` reads the
//     result of the dot.
//
//  3. During iteration i, the result of the dot from iteration i-1 is consumed
//     only by other MMAv3 dots (either sync or async) as the `c` operand.
//
//     This is safe because the following pseudo-PTX is valid:
//
//        %accum = dot_async %a1, %b1, %c1
//        %accum = dot_async %a2, %b2, %accum
//
//     That is, the second async dot can use the result of the first one without
//     an intervening wait.  However, the only operation that can legally read
//     %accum before the wait is another dot.async, and this only works for the
//     `c` operand, not `a` or `b`.  See
//     https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
//     (ttng::DotAsyncOp corresponds to wgmma.fence followed by one or more
//     wgmma.async ops, so our reading is that the two ttng::DotAsyncOps don't
//     have to correspond to wgmma.async ops with the same shapes as specified
//     in the docs, because there's an intervening fence.)
//
//     A synchronous MMAv3 dot will be codegen'ed as `dot.async; wait 0`, so it
//     works just as well as an async dot for these purposes.
//
// On success, this function returns the index of the dot in the loop's
// iter_args.  (Rule (2) above ensures this is well-defined.)
static std::optional<int> dotCanBeMadeAsync(Operation *op, scf::ForOp forOp) {
  // Rule 0: The operation is an MMAv3 dot.
  tt::DotOp dotOp = dynCastMMAv3Dot(op);
  if (!dotOp)
    return std::nullopt;

  // Rule 1: All shmem operands are multi-buffered.
  auto checkOperand = [&](Value operand) {
    if (!isa<ttg::SharedEncodingAttr>(
            operand.getType().cast<TensorOrMemDesc>().getEncoding())) {
      return true;
    }

    // If it's a shmem operand, it must either be defined outside the loop, or
    // come from an MemDescSubview op.  Only ConvertLayout and Trans ops are
    // allowed in between.
    Value transitiveOperand = operand;
    while (isa_and_nonnull<ttg::ConvertLayoutOp, tt::TransOp>(
        transitiveOperand.getDefiningOp())) {
      transitiveOperand = transitiveOperand.getDefiningOp()->getOperand(0);
    }
    return forOp.isDefinedOutsideOfLoop(transitiveOperand) ||
           isa<ttg::MemDescSubviewOp>(transitiveOperand.getDefiningOp());
  };

  // We don't have to call checkOperand on getC() because it's always in
  // registers, never in shmem.
  assert(isa<ttg::NvidiaMmaEncodingAttr>(dotOp.getC().getType().getEncoding()));
  if (!checkOperand(dotOp.getA()) || !checkOperand(dotOp.getB()))
    return std::nullopt;

  // Rule 2: The dot should only be used by the for loop's `yield`.
  if (!dotOp->hasOneUse() ||
      *dotOp->getUsers().begin() != forOp.getBody()->getTerminator())
    return std::nullopt;

  // The result of the dot becomes this loop carry value.
  auto iterArgIdx = dotOp->getUses().begin()->getOperandNumber();

  // Rule 3: The only users of the dot's result from iteration i-1 are other
  // MMAv3 dots (synchronous or async) as the `c` operand.
  if (!llvm::all_of(forOp.getRegionIterArg(iterArgIdx).getUses(),
                    [&](OpOperand &use) {
                      return dynCastMMAv3Dot(use.getOwner()) &&
                             use.getOperandNumber() == 2;
                    })) {
    return std::nullopt;
  }

  return iterArgIdx;
}

// If necessary, insert a dot-wait inside the loop, waiting for the results of
// the async dots from iteration i-1 to complete.  (We pipeline to depth 2, so
// there are at most 2 dots in flight at any point in time.)
//
// We can skip inserting the wait if we have a synchronous MMAv3 dot that
// appears either (a) before any uses of async dots' values from iteration i-1,
// or (b) after all uses of the async dots in iteration this iteration.
//
// By contract, the only users of the async dots' results from iteration i-1 are
// other MMAv3 dots, and the only user of the async dots in the current
// iteration is the loop's `yield` op.  So it's sufficient to check whether a
// sync dot appears first or last in the list of all sync+async dots.
//
// If such a sync dot exists, we convert it to async+wait so that we can thread
// the dependencies through the wait op.  This keeps other passes from
// reordering the dots.
//
// TODO(jlebar): It's possible that the sync dot might appear in the wrong place
// but could be hoisted/sunk to the right place.  Right now we don't handle
// this.
static void insertAsyncDotWaitInLoop(
    scf::ForOp forOp,
    MutableArrayRef<std::pair<Operation *, int /*iterArgIdx*/>> asyncDots) {
  // If a synchronous dot appears before an async dot in this range, return it.
  auto findSynchronizingDot = [](auto &&range) -> tt::DotOp {
    for (Operation &op : range) {
      if (isa<ttng::DotAsyncOp>(op)) {
        break;
      }
      if (tt::DotOp dot = dynCastMMAv3Dot(&op)) {
        return dot;
      }
    }
    return nullptr;
  };

  tt::DotOp synchronizingDot;
  bool syncAtBeginning = false;
  if (auto dot = findSynchronizingDot(*forOp.getBody())) {
    synchronizingDot = dot;
    syncAtBeginning = true;
  } else if (auto dot = findSynchronizingDot(llvm::reverse(*forOp.getBody()))) {
    synchronizingDot = dot;
  }

  // If we have a synchyronizing dot, convert it to async and insert a wait
  // right after it.  Otherwise, we'll insert the wait right after the last
  // async dot.  (You might want to put the wait at the end of the loop, but
  // there might be a load into shmem between the last async dot and the end of
  // the loop, and that could clobber memory being used by a dot.)
  IRRewriter builder(forOp.getContext());
  if (synchronizingDot) {
    builder.setInsertionPointAfter(synchronizingDot);
    builder.replaceOpWithNewOp<ttng::DotAsyncOp>(
        synchronizingDot.getOperation(), synchronizingDot.getA(),
        synchronizingDot.getB(), synchronizingDot.getC(),
        synchronizingDot.getAllowTF32(),
        synchronizingDot.getMaxNumImpreciseAcc());
  } else {
    builder.setInsertionPointAfter(asyncDots.back().first);
  }

  SmallVector<Value> waitOperands;
  for (auto [asyncDot, iterArgIdx] : asyncDots) {
    waitOperands.push_back(syncAtBeginning
                               ? cast<Value>(forOp.getRegionIterArg(iterArgIdx))
                               : cast<Value>(asyncDot->getResult(0)));
  }

  // If this wait belongs to a synchronous dot, we need to wait for *all* async
  // dots to complete, so we pass 0 for `pendings`.  OTOH if there are no sync
  // dots, we wait for all async dots from the i-1'th iteration to complete, IOW
  // we wait until there are at most `asyncDots.size()` dots in flight.
  auto dotWait = builder.create<ttng::DotWaitOp>(
      asyncDots.back().first->getLoc(), waitOperands,
      /*pendings=*/synchronizingDot ? 0 : asyncDots.size());
  for (int i = 0; i < waitOperands.size(); ++i) {
    waitOperands[i].replaceAllUsesExcept(dotWait.getResult(i), dotWait);
  }
}

// Convert tt::DotOps into ttng::DotAsyncOps and insert waits as necessary.
//
// We assume we have space for each dot to be pipelined to depth 2, i.e. each
// dot op in the loop can have at most 2 wgmma ops in flight at once.
void triton::asyncLaunchDots(scf::ForOp forOp) {
  LDBG("Original loop:\n" << *forOp);

  // Dots that can be made async.  Each dot's only use is in the loop's `yield`
  // statement; the `int` is its index in that op.
  SmallVector<std::pair<Operation *, int /*iterArgIdx*/>> asyncDots;
  for (Operation &op : *forOp.getBody()) {
    if (auto iterArgsIdx = dotCanBeMadeAsync(&op, forOp)) {
      LDBG("Will convert into an async dot: " << op);
      asyncDots.push_back({&op, iterArgsIdx.value()});
    }
  }

  if (asyncDots.empty()) {
    LDBG("No dots to make async.");
    return;
  }

  IRRewriter builder(forOp.getContext());

  // First, make each eligible dot asynchronous.
  for (auto &[op, iterArgIdx] : asyncDots) {
    auto dotOp = cast<tt::DotOp>(op);
    builder.setInsertionPoint(op);
    op = builder.replaceOpWithNewOp<ttng::DotAsyncOp>(
        dotOp, dotOp.getA(), dotOp.getB(), dotOp.getC(), dotOp.getAllowTF32(),
        dotOp.getMaxNumImpreciseAcc());
  }

  // Next, insert a wait inside the loop.  We pipeline to depth 2, so the third
  // iteration's set of asynchronous dots (and their corresponding async copies
  // from global to shmem) can't start until the first iteration's set has
  // completed.
  insertAsyncDotWaitInLoop(forOp, asyncDots);

  // Finally, insert a wait after the loop, waiting for dots from the final
  // iteration of the loop.
  SmallVector<Value> waitOperands;
  for (auto [asyncDot, iterArgIdx] : asyncDots) {
    waitOperands.push_back(forOp.getResult(iterArgIdx));
  }
  // Wait until there are 0 outstanding async dot ops.
  builder.setInsertionPointAfter(forOp);
  auto dotWaitAfterLoop =
      builder.create<ttng::DotWaitOp>(forOp.getLoc(), waitOperands, 0);
  for (int i = 0; i < waitOperands.size(); ++i) {
    waitOperands[i].replaceAllUsesExcept(dotWaitAfterLoop.getResult(i),
                                         dotWaitAfterLoop);
  }
}
