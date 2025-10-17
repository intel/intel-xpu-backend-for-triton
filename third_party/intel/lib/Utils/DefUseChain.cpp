#include "intel/include/Utils/DefUseChain.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "triton-intel-fuser"

namespace mlir::triton::intel {

// *******************************************************************************
// DefUseChain
// *******************************************************************************

DefUseChain::Operations DefUseChain::intersect(const DefUseChain &other) const {
  // Algorithm: I(S1,S2) = U(S1,S2) - U(S1-S2, S2-S1).

  Operations U = ops;
  if (!U.set_union(other.ops))
    return ops;

  Operations D1 = ops;
  Operations D2 = other.ops;
  D1.set_subtract(other.ops);
  D2.set_subtract(ops);

  Operations &U1 = D1;
  U1.set_union(D2);
  U.set_subtract(U1);

  return U;
}

raw_ostream &operator<<(raw_ostream &os, const DefUseChain &chain) {
  os << "[" << chain.start << ", " << chain.end << "]\n";
  os.indent(2) << "start: " << *chain.start << "\n";
  os.indent(2) << "end: " << *chain.end << "\n";
  os.indent(2) << "ops (" << chain.ops.size() << "):\n";
  for (Operation *op : chain.ops)
    os.indent(4) << *op << "\n";
  return os;
}

// *******************************************************************************
// DefUseChainManager
// *******************************************************************************

void DefUseChainManager::createChains(Operation *start, Operation *end) {
  assert(start && end && "Expecting valid operations");
  assert(start != end && "Expecting distinct operations");

  Operations path;
  SmallVector<Operations, 32> allPaths;
  findAllPaths(start, end, path, allPaths);

  for (Operations &path : allPaths) {
    DefUseChain chain(path);
    chains.insert(chain);
  }
}

void DefUseChainManager::pruneOverlappingChains(bool includeStart) {
  DefUseChains overlappingChains = getOverlappingChains();
  for (const DefUseChain &chain : overlappingChains) {
    for (const DefUseChain &other : overlappingChains) {
      if (chain == other || !other.overlap(chain))
        continue;

      Operations intersection = other.intersect(chain);
      assert(!intersection.empty() && "Expecting overlap");

      if (includeStart) {
        chains.erase(other);
        continue;
      }

      if (intersection.size() == 1 && intersection.contains(chain.getStart()))
        continue;

      assert(!includeStart && "Expecting 'includeStart' to be false");
      chains.erase(other);
    }
  }
}

void DefUseChainManager::findAllPaths(Operation *op, Operation *end,
                                      Operations &path,
                                      SmallVectorImpl<Operations> &allPaths) {
  assert(op && end && "Incorrect usage");

  // Add the current operation to the path.
  path.insert(op);

  // Reached the end, add the path to allPaths and end the recursion.
  if (op == end) {
    allPaths.push_back(path);
    path.pop_back();
    return;
  }

  Operations users;
  addUsers(op, path, users);

  // Recur for all users of the current operation.
  for (Operation *user : users) {
    bool pathContainsUser = llvm::find_if(path, [&](Operation *op) {
                              return op == user;
                            }) != path.end();
    if (!pathContainsUser)
      findAllPaths(user, end, path, allPaths);
  }

  path.pop_back();
}

void DefUseChainManager::addUsers(Operation *op, Operations path,
                                  Operations &users) const {
  assert(op && "Expecting a valid operation");
  assert(!path.empty() && "path should not be empty");
  assert(path.back() == op && "path should have 'op' as the last operation");

  auto addInitArgsUsers = [&](LoopLikeOpInterface loopOp,
                              Operation *previousOp) {
    // Add users of the block arguments initialized by `previousOp`.
    assert(previousOp && "Expecting valid operation");
    assert(previousOp->getNumResults() == 1 && "Unexpected operation");
    for (auto [arg, initVal] :
         llvm::zip(loopOp.getRegionIterArgs(), loopOp.getInits())) {
      // Skip arguments that aren't initialized by the previous operation.
      if (initVal != previousOp->getResult(0))
        continue;

      for (Operation *user : arg.getUsers()) {
        // Transfer to the 'after region' arguments of a while loop.
        if (auto condOp = dyn_cast<scf::ConditionOp>(user)) {
          if (auto whileOp = condOp->getParentOfType<scf::WhileOp>()) {
            unsigned argNo = 0;
            for (Value condOpArg : condOp.getArgs()) {
              if (condOpArg == arg)
                break;
              ++argNo;
            }

            BlockArgument afterRgnArg = whileOp.getAfterArguments()[argNo];
            users.insert_range(afterRgnArg.getUsers());
            continue;
          }
        }
        users.insert(user);
      }
    }
  };

  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    path.pop_back();
    Operation *previousOp = path.empty() ? nullptr : path.back();
    addInitArgsUsers(loopOp, previousOp);
  } else {
    users.insert_range(op->getUsers());
  }
}

DefUseChainManager::DefUseChains
DefUseChainManager::getOverlappingChains() const {
  if (chains.size() < 2)
    return {};

  DefUseChains overlappingChains;
  for (auto it1 = chains.begin(); it1 != chains.end(); ++it1) {
    for (auto it2 = it1; it2 != chains.end(); ++it2) {
      if (it2 == it1)
        continue;
      if (it2->overlap(*it1)) {
        overlappingChains.insert(*it1);
        overlappingChains.insert(*it2);
      }
    }
  }

  return overlappingChains;
}

raw_ostream &operator<<(raw_ostream &os, const DefUseChainManager &manager) {
  os << "Chains(" << manager.getChains().size() << "):\n";
  for (const DefUseChain &chain : manager.getChains())
    os.indent(2) << chain << "\n";
  return os;
}

// *******************************************************************************
// Fuser
// *******************************************************************************

void Fuser::fuse(const DefUseChains &chains) {
  for (const DefUseChain &chain : chains)
    fuse(chain);
}

void Fuser::duplicateRoot(DefUseChains &chains) const {
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

void Fuser::duplicateRoot(DefUseChains &sameRootChains,
                          DefUseChains &chains) const {
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

void Fuser::pruneInvalid(DefUseChains &chains) const {
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

bool Fuser::validateChain(const DefUseChain &chain) const {
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
    if (users.size() > 2 ||
        llvm::none_of(users, [&](Operation *user) { return user == yieldOp; }))
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
      LLVM_DEBUG(llvm::dbgs() << "Fails safety checks: " << *currentOp << "\n");
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

void Fuser::propagateToUsers(Value newVal, const DefUseChain &chain,
                             IRMapping &mapping) {
  auto start = cast<MakeTensorPtrOp>(chain.getStart());
  Operation *end = chain.getEnd();
  auto it = llvm::find_if(
      start->getUsers(), [&](Operation *user) { return chain.contains(user); });
  assert(it != start->getUsers().end() && "Expecting valid iterator");

  Operation *nextOp = *it;
  propagateToUser(newVal, start.getResult(), nextOp, end, mapping);
}

void Fuser::propagateToUsers(Value newVal, Value origVal, Operation *origOp,
                             Operation *sentinel, IRMapping &mapping) {
  assert(origOp && sentinel && "Expecting valid operations");
  const SmallVector<Operation *> users(origOp->getUsers());
  for (Operation *user : users)
    propagateToUser(newVal, origVal, user, sentinel, mapping);
}

void Fuser::propagateToLoop(Value newVal, Value origVal,
                            LoopLikeOpInterface loopOp, Operation *sentinel,
                            IRMapping &mapping) {
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
        propagateToUser(rgnInitArg, rgnInitArg, user, sentinel, mapping);
    }
  }
}

} // namespace mlir::triton::intel
