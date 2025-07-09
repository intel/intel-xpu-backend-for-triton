#include "intel/include/Utils/DefUseChain.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

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

} // namespace mlir::triton::intel
