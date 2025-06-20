#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
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

// Represent a def-use chain rooted at 'start' and terminating at tt.trans
// operation 'end'.
class Chain {
  friend raw_ostream &operator<<(raw_ostream &os, const Chain &chain);
  using Operations = llvm::SmallSetVector<Operation *, 4>;

public:
  Chain(Operation *start, tt::TransOp end) : start(start), end(end) {
    assert(start && end && "Expecting valid operations");
    assert(start != end && "Expecting distinct operations");
    assert(
        isTransitivelyUsedBy(start, end) &&
        "'end' operation should (transitively) use the result of the 'start' "
        "operation");
  }
  bool operator<(const Chain &other) const {
    return start < other.start || end < other.end;
  }
  bool operator==(const Chain &other) const {
    return start == other.start && end == other.end;
  }

  Operation *getStart() const { return start; }
  tt::TransOp getEnd() const { return end; }

  // Returns true if \p producer yields a result that is used (directly or
  // indirectly) by \p consumer.
  static bool isTransitivelyUsedBy(Operation *producer, Operation *consumer) {
    assert(producer && consumer && "Expecting valid operations");

    auto addUsers = [](Operation *op, Operations &users) {
      auto addUsers = [&](Operation *op) {
        // Add users of the block arguments in the 'after' region of a while
        // loop.
        if (auto condOp = dyn_cast<scf::ConditionOp>(op)) {
          if (auto whileOp = condOp->getParentOfType<scf::WhileOp>()) {
            for (BlockArgument arg : whileOp.getAfterArguments())
              for (Operation *user : arg.getUsers())
                users.insert(user);
          }
        }

        for (Operation *user : op->getUsers())
          users.insert(user);
      };

      auto addInitArgsUsers = [&](LoopLikeOpInterface loopOp) {
        for (Value val : loopOp.getRegionIterArgs())
          for (Operation *user : val.getUsers())
            addUsers(user);
      };

      if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op))
        addInitArgsUsers(loopOp);
      else
        addUsers(op);
    };

    Operations users;
    addUsers(producer, users);

    llvm::errs() << "producer: " << *producer << "\n";
    llvm::errs() << "users\n";
    for (Operation *user : users) {
      llvm::errs() << *user << "\n";
    }

    while (!users.contains(consumer)) {
      unsigned currentSize = users.size();
      for (Operation *user : users)
        addUsers(user, users);

      if (users.size() == currentSize) {
        llvm::errs() << "at line: " << __LINE__ << "\n";
        break;
      }
      llvm::errs() << "users\n";
      for (Operation *user : users) {
        llvm::errs() << *user << "\n";
      }
    }

    llvm::errs() << "consumer: " << *consumer << "\n";
    return users.contains(consumer);
  }

private:
  Operation *start = nullptr;
  tt::TransOp end = nullptr;
};

raw_ostream &operator<<(raw_ostream &os, const Chain &chain) {
  os << "[" << chain.start << ", " << &chain.end << "]\n";
  os.indent(2) << "start: " << *chain.start << "\n";
  os.indent(2) << "end: " << chain.end << "\n";
  return os;
}

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
  using Chains = std::set<Chain>;

  void run(ModuleOp moduleOp) {
    Chains chains;

    // Collect def-use chains originating at a `tt::MakeTensorPtrOp` operation
    // and terminating at a candidate `tt::TransOp` operation.
    // Note: A candidate `tt::TransOp` must use the result of a `tt.LoadOp`
    // using a ptr created the `tt::MakeTensorPtrOp` rooting the def-use chain.
    moduleOp.walk([&](tt::TransOp transOp) {
      if (isCandidate(transOp)) {
        auto loadOp = cast<tt::LoadOp>(transOp.getSrc().getDefiningOp());
        tt::MakeTensorPtrOp makeTensorPtrOp =
            *triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
        Chain chain(makeTensorPtrOp, transOp);
        chains.insert(chain);
      }
    });

    if (chains.empty())
      return;

    LLVM_DEBUG({
      llvm::dbgs() << "[Initial set of chains]:\n";
      for (const Chain &chain : chains)
        llvm::dbgs() << chain << "\n";
    });

    // If the same operation is the root of multiple chains, duplicate it to
    // make each chain disjoint from the others.
    if (chains.size() >= 2) {
      std::map<Operation *, Chains> rootToChains;
      for (const Chain &chain : chains) {
        Operation *start = chain.getStart();
        if (!rootToChains[start].empty())
          continue;

        Chains sameRootChains{chain};
        rootToChains[start] = sameRootChains;
        for (const Chain &otherChain : chains) {
          if (otherChain == chain || otherChain.getStart() != start)
            continue;

          Chains &sameRootChains = rootToChains[start];
          sameRootChains.insert(otherChain);
          rootToChains[start] = sameRootChains;
        }
      }

      for (auto &entry : rootToChains) {
        if (entry.second.size() == 1)
          continue;

        // duplicate the root operation.
        for (auto it = ++entry.second.begin(); it != entry.second.end(); ++it) {
          const Chain &chain = *it;
          Operation *start = chain.getStart();
          OpBuilder builder(start);
          Operation *duplicate = builder.insert(start->clone());
          assert(start->getNumResults() == 1);

          Value res = start->getResult(0);
          Value dupRes = duplicate->getResult(0);
          res.replaceUsesWithIf(dupRes, [&](OpOperand &operand) {
            return Chain::isTransitivelyUsedBy(operand.getOwner(),
                                               chain.getEnd());
          });

          // remove the chain and insert a new one, rooted by the duplicated
          // operation.
          Chain newChain(duplicate, chain.getEnd());
          chains.insert(newChain);
          chains.erase(chain);
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[Before Pruning]:\n";
      for (const Chain &chain : chains)
        llvm::dbgs() << chain << "\n";
    });

    // Prune candidate chains containing load/trans operations that cannot be
    // safely fused.
    prune(chains);

    LLVM_DEBUG({
      llvm::dbgs() << "[After Pruning]:\n";
      for (const Chain &chain : chains)
        llvm::dbgs() << chain << "\n";
    });

    // Fuse operations.
    fuse(chains);

    // Remove operations that are no longer used.
    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);

    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

  void fuse(const Chains &chains) {
    for (const Chain &chain : chains)
      fuseTransOpInChain(chain);
  }

  void fuseTransOpInChain(const Chain &chain) {
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

private:
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

    auto ptrType = cast<tt::PointerType>(loadOp.getPtr().getType());
    if (!isTensorPointerType(ptrType) ||
        cast<RankedTensorType>(ptrType.getPointeeType()).getRank() != 2)
      return false;

    std::optional<tt::MakeTensorPtrOp> makeTensorPtrOp =
        triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());

    return makeTensorPtrOp.has_value();
  }

  // Each operation in the def-use chain must have a single user, except in
  // special circumstances. Prune chains that do not satisfy this condition.
  void prune(Chains &chains) const {
    assert(!chains.empty() && "Expecting at least one candidate chain");
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
  bool validateChain(const Chain &chain) const {
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
        // If the operation has multiple users, attempt to determine whether all
        // users are in a def-use chain terminated by the equivalent candidate
        // transpose operations.
        LLVM_DEBUG(llvm::dbgs()
                   << "Fails safety checks: " << *currentOp << "\n");
        return false;
      }

      user = (!user) ? user = *currentOp->getUsers().begin() : user;
      if (user->getNumRegions() == 0) {
        currentOp = user;
        continue;
      }

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
  void propagateToUsers(Value newVal, const Chain &chain) {
    auto start = cast<tt::MakeTensorPtrOp>(chain.getStart());
    tt::TransOp end = chain.getEnd();
    auto it = llvm::find_if(start->getUsers(), [&](Operation *user) {
      return Chain::isTransitivelyUsedBy(user, end);
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
          loadOp.getLoc(), newVal, loadOp.getMask(), loadOp.getOther(),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());

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
