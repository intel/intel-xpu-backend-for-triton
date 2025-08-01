#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Utils/DefUseChain.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LayoutUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

#define DEBUG_TYPE "tritonintelgpu-optimize-dot-operands"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEDOTOPERANDS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// Transform:
//   %ptr = make_block_ptr [shapeN, shapeK], [strideN, strideK], [offN, offK]
//        : tt.ptr<tensor<NxK, enc>
//   %load = tt.load %ptr, {blockIO=<row_major|column_major>}
//         : tt.ptr<tensor<NxK, enc>
//   %trans = tt.trans %load : tt.ptr<tensor<KxN, dotEnc>>
//   tt.dot(%a, %trans)
// into:
//   %ptr = make_block_ptr [shapeK, shapeN], [strideK, strideN], [offK, offN]
//        : tt.ptr<tensor<KxN, dotEnc>
//   %load = tt.load %ptr, {blockIO=<column_major|row_major>}
//         : tt.ptr<tensor<KxN, dotEnc>
//   tt.dot(%a, %load)
class FuseTransWithLoad {
private:
  SmallPtrSet<Operation *, 8> cleanUp;

public:
  using DefUseChain = tt::intel::DefUseChain;
  using DefUseChainManager = tt::intel::DefUseChainManager;
  using DefUseChains = DefUseChainManager::DefUseChains;

  void run(ModuleOp moduleOp) {
    // Collect def-use chains originating at a `MakeTensorPtrOp` operation
    // and terminating at a candidate `tt::TransOp` operation.
    // Note: A candidate `TransOp` must use the result of a `LoadOp` using a ptr
    // created the `MakeTensorPtrOp` rooting the def-use chain.
    DefUseChainManager manager;
    moduleOp.walk([&](tt::TransOp transOp) {
      if (isCandidate(transOp)) {
        auto loadOp = cast<tt::LoadOp>(transOp.getSrc().getDefiningOp());
        tt::MakeTensorPtrOp makeTensorPtrOp =
            *triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
        manager.createChains(makeTensorPtrOp, transOp);
      }
    });

    if (manager.getChains().empty())
      return;

    LLVM_DEBUG(llvm::dbgs() << "[Initial set of chains]:\n" << manager << "\n");

    // Prune chains that overlap with other chains (except at the root).
    unsigned numChainsCollected = manager.getChains().size();
    bool includeStart = false;
    manager.pruneOverlappingChains(includeStart);
    if (manager.getChains().empty())
      return;

    LLVM_DEBUG({
      if (manager.getChains().size() != numChainsCollected)
        llvm::dbgs() << "[After pruning]:\n" << manager << "\n";
    });

    // Prune chains that cannot be fused.
    pruneInvalid(manager.getChainsMutable());
    if (manager.getChains().empty())
      return;

    LLVM_DEBUG(llvm::dbgs() << "[Before fusion]:\n" << manager << "\n");

    // Fuse tt.LoadOp->tt.TransOp operations.
    fuseTransOp(manager.getChains());

    // Remove operations that are no longer used.
    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);

    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

private:
  // Duplicate the root operation of the given chains.
  void duplicateRoot(DefUseChains &chains) const {
    std::map<Operation *, DefUseChains> rootToChains;
    for (const DefUseChain &chain : chains) {
      Operation *start = chain.getStart();
      if (!rootToChains[start].empty())
        continue;

      DefUseChains sameRootChains{chain};
      rootToChains[start] = sameRootChains;
      for (const DefUseChain &otherChain : chains) {
        if (otherChain == chain || otherChain.getStart() != start)
          continue;

        rootToChains[start].insert(otherChain);
      }
    }

    for (auto &entry : rootToChains) {
      DefUseChains &sameRootChains = entry.second;
      duplicateRoot(sameRootChains, chains);
    }
  }

  // Duplicate the root operation of \p sameRootChains and update \p chains.
  void duplicateRoot(DefUseChains &sameRootChains, DefUseChains &chains) const {
    assert(llvm::all_of(sameRootChains, [&](const DefUseChain &chain) {
      const DefUseChain &firstChain = *sameRootChains.begin();
      return firstChain.getStart() == chain.getStart();
    }));

    for (auto it = sameRootChains.begin(); it != sameRootChains.end(); ++it) {
      const DefUseChain &chain = *it;
      Operation *start = chain.getStart();
      auto users = start->getUsers();
      if (llvm::count_if(users, [](auto) { return true; }) == 1)
        continue;

      OpBuilder builder(start);
      Operation *duplicate = builder.insert(start->clone());
      assert(start->getNumResults() == 1);

      Value res = start->getResult(0);
      Value dupRes = duplicate->getResult(0);
      res.replaceUsesWithIf(dupRes, [&](OpOperand &operand) {
        Operation *op = operand.getOwner();
        return chain.contains(op);
      });

      DefUseChainManager manager;
      manager.createChains(duplicate, chain.getEnd());
      for (DefUseChain newChain : manager.getChains())
        chains.insert(newChain);
      chains.erase(chain);
    }
  }

  void fuseTransOp(const DefUseChains &chains) {
    for (const DefUseChain &chain : chains)
      fuseTransOp(chain);
  }

  void fuseTransOp(const DefUseChain &chain) {
    assert(
        isa<tt::MakeTensorPtrOp>(chain.getStart()) &&
        "Expecting 'chain' to be rooted by a 'tt.make_tensor_ptr' operation");
    assert(isa<tt::TransOp>(chain.getEnd()) &&
           "Expecting 'chain' to be terminated by a 'tt.trans' operation");

    auto makeTensorPtrOp = cast<tt::MakeTensorPtrOp>(chain.getStart());
    auto transOp = cast<tt::TransOp>(chain.getEnd());
    auto loadOp = cast<tt::LoadOp>(transOp.getSrc().getDefiningOp());
    LLVM_DEBUG(llvm::dbgs()
               << "Fusing:\n  " << transOp << "\nwith:\n  " << loadOp << "\n");

    // Create a MakeTensorPtrOp yielding a block pointer to the transposed
    // tensor...
    auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
    auto tensorType = cast<RankedTensorType>(transOp.getType());
    auto newPtrType =
        tt::PointerType::get(tensorType, ptrType.getAddressSpace());
    SmallVector<Value> newShape(llvm::reverse(makeTensorPtrOp.getShape()));
    SmallVector<Value> newStrides(llvm::reverse(makeTensorPtrOp.getStrides()));
    SmallVector<Value> newOffsets(llvm::reverse(makeTensorPtrOp.getOffsets()));

    OpBuilder builder(makeTensorPtrOp);
    Value ptr = builder.create<tt::MakeTensorPtrOp>(
        makeTensorPtrOp.getLoc(), newPtrType, makeTensorPtrOp.getBase(),
        newShape, newStrides, newOffsets, makeTensorPtrOp.getOrderAttr());
    LLVM_DEBUG(llvm::dbgs() << "newMakeTensorPtrOp:\n  " << ptr << "\n");

    // ... and propagate it through the def-use chain.
    propagateToUsers(ptr, chain);
    cleanUp.insert(makeTensorPtrOp);
  }

  // Candidate is of the form:
  //   tt.dot(tt.trans(tt.load(..., {blockIO=...})))
  // Where:
  //  - the transpose result is used by the dot operation, and
  //  - the transpose operation uses the result of a 2-dim load operation on a
  //    block pointer (transitively) defined by a `make_tensor_ptr` operation.
  bool isCandidate(tt::TransOp transOp) const {
    assert(transOp && "Expecting a valid transpose operation");

    // Check whether \p transOp is used by a `dotOp` (directly or indirectly).
    auto usedByDotOp = [](tt::TransOp transOp) {
      if (!transOp->hasOneUse())
        return false;

      Operation *user = *transOp->getUsers().begin();
      while (user) {
        if (isa<tt::DotOp>(user))
          return true;
        if (!user->hasOneUse())
          break;
        user = *user->getUsers().begin();
      }

      return false;
    };

    Attribute transOpEncoding = transOp.getType().getEncoding();
    if (!usedByDotOp(transOp) || !transOpEncoding ||
        !isa<ttg::DotOperandEncodingAttr>(transOpEncoding))
      return false;

    Operation *defOp = transOp.getSrc().getDefiningOp();
    if (!defOp || !isa<tt::LoadOp>(defOp))
      return false;

    auto loadOp = cast<tt::LoadOp>(defOp);
    bool loadOpHasBlockIOAttr = loadOp->hasAttrOfType<StringAttr>(
        ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
    if (!loadOp->hasOneUse() || !loadOpHasBlockIOAttr)
      return false;

    Type ptrType = loadOp.getPtr().getType();
    if (!tt::isTensorPointerType(ptrType) ||
        cast<RankedTensorType>(cast<tt::PointerType>(ptrType).getPointeeType())
                .getRank() != 2)
      return false;

    std::optional<tt::MakeTensorPtrOp> makeTensorPtrOp =
        triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());

    return makeTensorPtrOp.has_value();
  }

  // Prune chains that cannot be handled during fusion. For example, operations
  // in the def-use chain should have a single user, except in special
  // circumstances (e.g. the root operation of a chain might have more than one
  // user).
  void pruneInvalid(DefUseChains &chains) const {
    assert(!chains.empty() && "Expecting at least one candidate chain");

    // Duplicate the root operation if necessary.
    // Note: at this point overlap, if present, can only happen at the root.
    duplicateRoot(chains);

    for (auto it = chains.begin(); it != chains.end();) {
      if (!validateChain(*it))
        it = chains.erase(it);
      else
        ++it;
    }
  }

  // Determine whether all operations in the given def-use chain have a single
  // user.
  // Note: we allow an operation in the def-use chain to have an additional user
  // if the operation is in a for loop, and the additional user is the loop
  // yield operation, provided that the result yielded is not used after the
  // loop.
  // Example:
  //   make_tensor_ptr -> advance -> load (OK)
  //   make_tensor_ptr -> for init_arg -> advance -> load (OK)
  //                                   -> yield (OK)
  //   make_tensor_ptr -> for init_arg -> advance -> load (OK)
  //                                              -> yield -> load (NOT OK)
  //
  bool validateChain(const DefUseChain &chain) const {
    auto validateOperation = [](Operation *op, Operation *&nextOp) {
      assert(nextOp == nullptr);
      if (op->hasOneUse())
        return true;
      if (!op->getParentOfType<LoopLikeOpInterface>())
        return false;

      auto loopOp = op->getParentOfType<LoopLikeOpInterface>();
      auto yieldOp = cast<scf::YieldOp>(
          loopOp.getYieldedValues()[0].getParentBlock()->getTerminator());

      SmallVector<Operation *> users(op->getUsers());
      if (users.size() > 2 || llvm::none_of(users, [&](Operation *user) {
            return user == yieldOp;
          }))
        return false;

      auto yieldedValUsedAfterLoop = [&op, &yieldOp]() {
        auto it =
            llvm::find_if(yieldOp->getOpOperands(), [&op](OpOperand &operand) {
              return operand.get() == op->getResult(0);
            });
        assert(it != yieldOp->getOpOperands().end());
        OpOperand &operand = *it;
        auto loopOp = cast<LoopLikeOpInterface>(yieldOp->getParentOp());
        OpResult res = loopOp->getResult(operand.getOperandNumber());
        return !res.getUsers().empty();
      };

      if (yieldedValUsedAfterLoop())
        return false;

      nextOp = *llvm::find_if(
          users, [](Operation *user) { return !isa<scf::YieldOp>(user); });
      return true;
    };

    Operation *currentOp = chain.getStart();
    while (currentOp != chain.getEnd()) {
      Operation *user = nullptr;
      if (!validateOperation(currentOp, user)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Fails safety checks: " << *currentOp << "\n");
        return false;
      }

      user = (!user) ? user = *currentOp->getUsers().begin() : user;
      if (user->getNumRegions() == 0) {
        currentOp = user;
        continue;
      }

      // Current limitation: give up if the use is a branch.
      if (isa<scf::IfOp>(user))
        return false;

      [[maybe_unused]] Operation *oldCurrentOp = currentOp;

      // Find the next operation in the def-use chain inside the loop body.
      if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
        for (auto [arg, init] :
             llvm::zip(loopOp.getRegionIterArgs(), loopOp.getInits())) {
          if (init == currentOp->getResult(0)) {
            if (!arg.hasOneUse())
              return false;

            currentOp = *arg.getUsers().begin();
            break;
          }
        }
      }

      assert(currentOp != oldCurrentOp && "Infinite loop detected!");
    }

    return true;
  }

  // Propagate \p newVal to operations in the given def-use chain.
  void propagateToUsers(Value newVal, const DefUseChain &chain) {
    auto start = cast<tt::MakeTensorPtrOp>(chain.getStart());
    Operation *end = chain.getEnd();
    auto it = llvm::find_if(start->getUsers(), [&](Operation *user) {
      return chain.contains(user);
    });
    assert(it != start->getUsers().end() && "Expecting valid iterator");

    Operation *nextOp = *it;
    propagateToUser(newVal, start.getResult(), nextOp, end);
  }

  // Propagate \p newVal to users of \p origOp.
  void propagateToUsers(Value newVal, Value origVal, Operation *origOp,
                        Operation *sentinel) {
    assert(origOp && sentinel && "Expecting valid operations");
    const SmallVector<Operation *> users(origOp->getUsers());
    for (Operation *user : users)
      propagateToUser(newVal, origVal, user, sentinel);
  }

  // If \p user is not \p sentinel, propagate \p newVal to \p user. Otherwise
  // terminate the propagation.
  void propagateToUser(Value newVal, Value origVal, Operation *user,
                       Operation *sentinel) {
    assert(user && sentinel && "Expecting valid operations");
    assert(llvm::is_contained(origVal.getUsers(), user) && "Invalid usage");

    LLVM_DEBUG({
      llvm::dbgs() << "In " << __func__ << "\n";
      llvm::dbgs() << "user of:";
      if (origVal.getDefiningOp()) {
        llvm::dbgs() << "\n  " << *origVal.getDefiningOp() << "\n";
      } else {
        origVal.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << " ";
      }
      llvm::dbgs() << "is:\n  ";
      user->dumpPretty();
    });

    if (user == sentinel) {
      LLVM_DEBUG(llvm::dbgs() << "Reached sentinel\n");
      sentinel->replaceAllUsesWith(newVal.getDefiningOp());
      cleanUp.insert(sentinel);
      return;
    }

    Location loc = user->getLoc();
    if (auto advanceOp = dyn_cast<tt::AdvanceOp>(user)) {
      OpBuilder rewriter(advanceOp);
      SmallVector<Value> newOffsets(llvm::reverse(advanceOp.getOffsets()));
      auto newAdvanceOp = rewriter.create<tt::AdvanceOp>(loc, newVal.getType(),
                                                         newVal, newOffsets);
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "newAdvanceOp: " << newAdvanceOp << "\n");
      cleanUp.insert(advanceOp);
      return propagateToUsers(newAdvanceOp, advanceOp.getResult(), advanceOp,
                              sentinel);
    }

    if (auto loadOp = dyn_cast<tt::LoadOp>(user)) {
      OpBuilder rewriter(loadOp);
      auto newLoadOp = rewriter.create<tt::LoadOp>(
          loadOp.getLoc(), triton::getPointeeType(newVal.getType()), newVal,
          loadOp.getMask(), loadOp.getOther(), loadOp.getBoundaryCheckAttr(),
          loadOp.getPaddingAttr(), loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());

      StringRef blockIOAttrName =
          ttgi::TritonIntelGPUDialect::getBlockIOAttrName();
      StringAttr attr = loadOp->getAttrOfType<StringAttr>(blockIOAttrName);
      StringAttr newAttr =
          (attr == "row_major")
              ? StringAttr::get(loadOp->getContext(), "column_major")
          : (attr == "column_major")
              ? StringAttr::get(loadOp->getContext(), "row_major")
              : nullptr;
      assert(newAttr && "Expecting a valid blockIO attribute");

      newLoadOp->setAttr(blockIOAttrName, newAttr);
      LLVM_DEBUG(llvm::dbgs().indent(2) << "newLoadOp: " << newLoadOp << "\n");
      cleanUp.insert(loadOp);
      return propagateToUsers(newLoadOp, loadOp.getResult(), loadOp, sentinel);
    }

    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      int opNum = -1;
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        if (operand.get() == origVal) {
          opNum = operand.getOperandNumber();
          yieldOp->setOperand(operand.getOperandNumber(), newVal);
          break;
        }
      }

      // Update the yield's parent operation result type.
      Operation *parentOp = yieldOp->getParentOp();
      OpResult res = parentOp->getOpResult(opNum);
      res.setType(newVal.getType());
      return;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user))
      return propagateToLoop(newVal, origVal, forOp, sentinel);

    llvm_unreachable("Unexpected kind of user");
  }

  void propagateToLoop(Value newVal, Value origVal, LoopLikeOpInterface loopOp,
                       Operation *sentinel) {
    assert(sentinel && sentinel != loopOp && "Unexpected sentinel kind");
    LLVM_DEBUG({
      llvm::dbgs() << "In " << __func__ << "\n";
      llvm::dbgs() << "newVal: " << newVal << "\n";
    });

    for (auto [initArg, rgnInitArg] :
         llvm::zip(loopOp.getInitsMutable(), loopOp.getRegionIterArgs())) {
      if (initArg.get() == origVal) {
        initArg.set(newVal);
        rgnInitArg.setType(initArg.get().getType());
        const SmallVector<Operation *> users(rgnInitArg.getUsers());
        for (Operation *user : users)
          propagateToUser(rgnInitArg, rgnInitArg, user, sentinel);
      }
    }
  }
};

} // namespace

class TritonIntelGPUOptimizeDotOperandsPass
    : public ttgi::impl::TritonIntelGPUOptimizeDotOperandsBase<
          TritonIntelGPUOptimizeDotOperandsPass> {

public:
  using ttgi::impl::TritonIntelGPUOptimizeDotOperandsBase<
      TritonIntelGPUOptimizeDotOperandsPass>::
      TritonIntelGPUOptimizeDotOperandsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    FuseTransWithLoad fuser;
    fuser.run(moduleOp);
  }
};
