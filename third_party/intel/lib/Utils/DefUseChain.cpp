#include "intel/include/Utils/DefUseChain.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

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

bool DefUseChain::isTransitivelyUsedBy(Operation *op, Operation *consumer,
                                       DefUseChain::Operations &path) const {
  if (!contains(op) || !contains(consumer))
    return false;

  path.insert(op);
  if (op == end)
    return true;

  auto addUsers = [this](Operation *op, Operations &users) {
    if (auto condOp = dyn_cast<scf::ConditionOp>(op)) {
      if (auto whileOp = condOp->getParentOfType<scf::WhileOp>()) {
        for (BlockArgument arg : whileOp.getAfterArguments())
          for (Operation *user : arg.getUsers())
            if (contains(user))
              users.insert(user);
      }
    }

    for (Operation *user : op->getUsers()) {
      if (contains(user))
        users.insert(user);
    }
  };

  auto addInitArgsUsers = [&](LoopLikeOpInterface loopOp, Operations &users) {
    for (Value val : loopOp.getRegionIterArgs())
      for (Operation *user : val.getUsers())
        addUsers(user, users);
  };

  Operations users;
  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op))
    addInitArgsUsers(loopOp, users);
  else
    addUsers(op, users);

  for (Operation *user : users)
    return isTransitivelyUsedBy(user, end, path);

  return false;
}

raw_ostream &operator<<(raw_ostream &os, const DefUseChain &chain) {
  if (!chain.valid) {
    os << "Chains is invalid";
    return os;
  }

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
    if (chain.valid)
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

// Find all def-use paths from \p start to \p end and add them to \p allPaths.
void DefUseChainManager::findAllPaths(Operation *start, Operation *end,
                                      Operations &path,
                                      SmallVectorImpl<Operations> &allPaths) {
  assert(start && end && "Incorrect usage");

  // Add the current node to the path.
  path.insert(start);

  // Reached the end, add the path to allPaths and end the recursion.
  if (start == end) {
    allPaths.push_back(path);
    path.pop_back();
    return;
  }

  Operations users;
  addUsers(start, users);

  // Recur for all users of the current operation.
  for (Operation *user : users) {
    auto it = llvm::find_if(path, [&](Operation *op) { return op == user; });
    if (it == path.end())
      findAllPaths(user, end, path, allPaths);
  }

  path.pop_back();
}

void DefUseChainManager::addUsers(Operation *op, Operations &users) const {
  assert(op && "Expecting valid operation");

  auto addUsers = [&](Operation *op) {
    // Add users of the block arguments in the 'after' region of a while loop.
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
  for (const DefUseChain &chain : manager.getChains())
    os << chain << "\n";
  return os;
}

} // namespace mlir::triton::intel
