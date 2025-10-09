#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/DefUseChain.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-intel-fuse-reshape"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELFUSERESHAPE
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

// Transform:
//   %one = arith.constant 1 : i64
//   %ptr = make_tensor_ptr %q_view, [%q, %q_23, %q_24],
//            [%q_25, %q_26, %one], [%offset_5, %offset_1_13, %q_28]
//            {order = array<i32: 2, 1, 0>} : <tensor<1x512x64xf16>>
//   %load = tt.load %ptr {boundaryCheck = array<i32: 1, 2>}
//         : !tt.ptr<tensor<1x512x64xf16>>
//   %a = tt.reshape %load : tensor<1x512x64xf16> -> tensor<512x64xf16>
//   tt.dot(%a, ...)
// into:
//   %one = arith.constant 1 : i64
//   %ptr = make_tensor_ptr %q_view, [%q_23, %q_24], [%q_26, %one],
//            [%offset_1_13, %offset_5*%q_25+%q_28]
//            {order = array<i32: 1, 0>} : <tensor<512x64xf16>>
//   %a = tt.load %ptr {boundaryCheck = array<i32: 0, 1>}
//      : !tt.ptr<tensor<512x64xf16>>
//   tt.dot(%a, ...)
class FuseReshape {
private:
  SmallPtrSet<Operation *, 8> cleanUp;

public:
  using DefUseChain = tt::intel::DefUseChain;
  using DefUseChainManager = tt::intel::DefUseChainManager;
  using DefUseChains = DefUseChainManager::DefUseChains;

  void run(ModuleOp moduleOp) {
    // Collect def-use chains originating at a `MakeTensorPtrOp` operation
    // and terminating at a candidate `tt::ReshapeOp` operation.
    // Note: A candidate `reshapeOp` must use the result of a `loadOp` using a
    // ptr created the `MakeTensorPtrOp` rooting the def-use chain.
    DefUseChainManager manager;
    moduleOp.walk([&](tt::ReshapeOp reshapeOp) {
      if (isCandidate(reshapeOp)) {
        auto loadOp = cast<tt::LoadOp>(reshapeOp.getSrc().getDefiningOp());
        tt::MakeTensorPtrOp makeTensorPtrOp =
            *triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
        manager.createChains(makeTensorPtrOp, reshapeOp);
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

    // Fuse tt.LoadOp->tt.reshapeOp operations.
    fuse(manager.getChains());

    // Remove operations that are no longer used.
    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);
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

  void fuse(const DefUseChains &chains) {
    for (const DefUseChain &chain : chains)
      fuse(chain);
  }

  void fuse(const DefUseChain &chain) {
    assert(
        isa<tt::MakeTensorPtrOp>(chain.getStart()) &&
        "Expecting 'chain' to be rooted by a 'tt.make_tensor_ptr' operation");
    assert(isa<tt::ReshapeOp>(chain.getEnd()) &&
           "Expecting 'chain' to be terminated by a 'tt.reshape' operation");

    auto makeTensorPtrOp = cast<tt::MakeTensorPtrOp>(chain.getStart());
    auto reshapeOp = cast<tt::ReshapeOp>(chain.getEnd());
    auto loadOp = cast<tt::LoadOp>(reshapeOp.getSrc().getDefiningOp());
    LLVM_DEBUG(llvm::dbgs() << "Fusing:\n  " << reshapeOp << "\nwith:\n  "
                            << loadOp << "\n");

    // Create a MakeTensorPtrOp yielding a 2-dim block pointer.
    auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
    [[maybe_unused]] ArrayRef<int64_t> resShape =
        cast<RankedTensorType>(ptrType.getPointeeType()).getShape();
    assert(resShape[0] == 1 && "Result shape should have extent equal to 1 in "
                               "the outermost dimension");

    auto tensorType = cast<RankedTensorType>(reshapeOp.getType());
    auto newPtrType =
        tt::PointerType::get(tensorType, ptrType.getAddressSpace());

    // Compute the index of the innermost dimension.
    ArrayRef<int> order = makeTensorPtrOp.getOrder();
    assert(order.size() == 3 && order[0] == 2 && "Invalid order");

    unsigned innermostDimIdx = 0;
    for (int elem : makeTensorPtrOp.getOrder()) {
      if (elem == 0)
        break;
      ++innermostDimIdx;
    }

    OpBuilder builder(makeTensorPtrOp);
    Location loc = makeTensorPtrOp.getLoc();
    OperandRange shapes = makeTensorPtrOp.getShape();
    OperandRange strides = makeTensorPtrOp.getStrides();
    OperandRange offsets = makeTensorPtrOp.getOffsets();

    // Collapse the 3-dim tensor into a 2-dim tensor.
    // Given a make_tensor_ptr with:
    //   shape  [s0, s1, s2]
    //   stride [a, b, c]
    //   offset [x, y, z]
    //   order  [2, 1, 0]
    // We create a make_tensor_ptr with:
    //   shape  [s0 * a / b + s1, s2]
    //   stride [b, c]
    //   offset [x * a / b + y, z]
    //   order  [1, 0]
    SmallVector<Value> newShape(makeTensorPtrOp.getShape().drop_front());
    SmallVector<Value> newStrides(makeTensorPtrOp.getStrides().drop_front());
    SmallVector<Value> newOffsets(makeTensorPtrOp.getOffsets().drop_front());

    unsigned newInnermostDimIdx = (innermostDimIdx - 1);
    unsigned newOutermostDimIdx = !newInnermostDimIdx;
    auto div = builder.create<arith::DivUIOp>(loc, strides[0],
                                              newStrides[newOutermostDimIdx]);

    newShape[newOutermostDimIdx] = builder.create<arith::AddIOp>(
        loc, builder.create<arith::MulIOp>(loc, shapes[0], div),
        newShape[newOutermostDimIdx]);
    newOffsets[newOutermostDimIdx] = builder.create<arith::AddIOp>(
        loc,
        builder.create<arith::MulIOp>(
            loc, offsets[0],
            builder.create<arith::TruncIOp>(loc, offsets[0].getType(), div)),
        newOffsets[newOutermostDimIdx]);

    Value ptr = builder.create<tt::MakeTensorPtrOp>(
        loc, newPtrType, makeTensorPtrOp.getBase(), newShape, newStrides,
        newOffsets,
        DenseI32ArrayAttr::get(
            builder.getContext(),
            makeTensorPtrOp.getOrderAttr().asArrayRef().drop_front()));

    LLVM_DEBUG(llvm::dbgs() << "newMakeTensorPtrOp:\n  " << ptr << "\n");

    // Adjust the boundary check on the load operation.
    ArrayRef<int> boundaryCheck = loadOp.getBoundaryCheck();
    assert(boundaryCheck.size() == 2 && "Expecting a 2-dim load");
    loadOp.setBoundaryCheck({boundaryCheck[0] - 1, boundaryCheck[1] - 1});

    // Propagate the new ptr through the def-use chain.
    propagateToUsers(ptr, chain);
    cleanUp.insert(makeTensorPtrOp);
  }

  // Candidate is of the form:
  //   tt.dot(tt.reshape(tt.load(..., )))
  // Where:
  //  - the reshape operation drops the outermost dimension of the operand,
  //    which is a 3-dim tensor with outermost dimension extent equal to one
  //  - the reshape result is used by the dot operation
  //  - the reshape operation uses the result of a 3-dim load operation on a
  //    block pointer (transitively) defined by a `make_tensor_ptr` operation
  //  - the block pointer points to a tensor that has extent equal to 1 on the
  //    outermost dimension
  bool isCandidate(tt::ReshapeOp reshapeOp) const {
    assert(reshapeOp && "Expecting a valid reshape operation");

    ArrayRef<int64_t> reshapeOperandShape =
        reshapeOp.getSrc().getType().getShape();
    if (reshapeOperandShape.size() != 3 || reshapeOperandShape.front() != 1)
      return false;

    ArrayRef<int64_t> reshapeResultShape = reshapeOp.getType().getShape();
    if (reshapeResultShape.size() != reshapeOperandShape.size() - 1)
      return false;

    for (auto pair :
         llvm::zip(reshapeOperandShape.drop_front(), reshapeResultShape)) {
      if (std::get<0>(pair) != std::get<1>(pair))
        return false;
    }

    // Check whether \p reshapeOp is used by a `dotOp` (directly or indirectly).
    auto usedByDotOp = [](tt::ReshapeOp reshapeOp) {
      if (!reshapeOp->hasOneUse())
        return false;

      Operation *user = *reshapeOp->getUsers().begin();
      while (user) {
        if (isa<tt::DotOp>(user))
          return true;
        if (!user->hasOneUse())
          break;
        user = *user->getUsers().begin();
      }

      return false;
    };

    if (!usedByDotOp(reshapeOp))
      return false;

    Operation *defOp = reshapeOp.getSrc().getDefiningOp();
    if (!defOp || !isa<tt::LoadOp>(defOp))
      return false;

    auto loadOp = cast<tt::LoadOp>(defOp);
    if (!loadOp->hasOneUse())
      return false;

    Type ptrType = loadOp.getPtr().getType();
    if (!tt::isTensorPointerType(ptrType))
      return false;

    std::optional<tt::MakeTensorPtrOp> makeTensorPtrOp =
        triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
    if (!makeTensorPtrOp)
      return false;

    tt::PointerType ptrTy = makeTensorPtrOp->getResult().getType();
    auto tensorTy = cast<RankedTensorType>(ptrTy.getPointeeType());
    assert((tensorTy.getRank() == 3 && tensorTy.getDimSize(0) == 1) &&
           "Unexpected tensor type");

    // Ensure the outermost dimension is the one with highest order.
    ArrayRef<int> order = makeTensorPtrOp->getOrder();
    if (order.front() != tensorTy.getRank() - 1)
      return false;

    // Ensure that the innermost stride is one.
    unsigned innermostDimIdx = 0;
    for (int i : order) {
      if (i == 0)
        break;
      ++innermostDimIdx;
    }

    auto strides = makeTensorPtrOp->getStrides();
    Value innermostStride = strides[innermostDimIdx];
    if (!innermostStride.getDefiningOp() ||
        !isa<arith::ConstantIntOp>(innermostStride.getDefiningOp()))
      return false;

    auto integerCst =
        cast<arith::ConstantIntOp>(innermostStride.getDefiningOp());
    if (integerCst.value() != 1)
      return false;

    // Ensure the load boundary check doesn't check the outermost dimension.
    return llvm::none_of(loadOp.getBoundaryCheck(),
                         [&](int val) { return val == 0; });
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
            auto argUsers = arg.getUsers();
            for (Operation *user : argUsers) {
              if (chain.contains(user)) {
                currentOp = user;
                break;
              }
            }
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
      llvm::dbgs() << "user of: ";
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
      SmallVector<Value> newOffsets(advanceOp.getOffsets().drop_front());
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
      newLoadOp->setAttrs(loadOp->getAttrs());

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

struct TritonIntelFuseReshape
    : tt::intel::impl::TritonIntelFuseReshapeBase<TritonIntelFuseReshape> {
public:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    FuseReshape fuser;
    fuser.run(moduleOp);
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
