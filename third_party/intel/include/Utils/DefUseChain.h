#ifndef TRITON_INTEL_UTILS_DEFUSECHAIN_H
#define TRITON_INTEL_UTILS_DEFUSECHAIN_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include <unordered_set>

namespace mlir::triton::intel {

/// \class DefUseChain
/// Represent a def-use chain rooted at an operation 'start' and terminating at
/// another operation 'end'.
/// Note: a \class DefUseChain can only be constructed by a \class
/// DefUseChainManager.
class DefUseChain {
  friend class DefUseChainManager;
  friend raw_ostream &operator<<(raw_ostream &, const DefUseChain &);

public:
  using Operations = llvm::SmallSetVector<Operation *, 32>;

  DefUseChain() = delete;

  bool operator==(const DefUseChain &other) const { return ops == other.ops; }

  const Operations &getOps() const { return ops; }
  Operation *getStart() const { return start; }
  Operation *getEnd() const { return end; }

  /// Compute the intersection between the ops in this chain and the ops in \p
  /// other.
  Operations intersect(const DefUseChain &other) const;

  /// Return true if this chain and \p other contain one or more common
  /// operations, and false otherwise.
  bool overlap(const DefUseChain &other) const {
    return !intersect(other).empty();
  }

  // Return true if the chain contains the given operation \p op, and false
  // otherwise.
  bool contains(Operation *op) const {
    assert(op && "Expecting a valid operation");
    return ops.contains(op);
  }

private:
  DefUseChain(const Operations &ops)
      : ops(ops), start(ops.front()), end(ops.back()) {
    assert(start && end && "Expecting valid operations");
    assert(start != end && "Expecting distinct operations");
  }

  Operations ops;   //< operations in the chain
  Operation *start; //< first operation in the chain
  Operation *end;   //< last operation in the chain
};

struct DefUseChainHash {
  size_t operator()(const mlir::triton::intel::DefUseChain &c) const noexcept {
    return llvm::hash_combine(c.getStart(), c.getEnd());
  }
};

/// \class DefUseChainManager
/// Manages collection of one or more \class DefUseChain.
class DefUseChainManager {
  friend raw_ostream &operator<<(raw_ostream &, const DefUseChainManager &);

public:
  using DefUseChains = std::unordered_set<DefUseChain, DefUseChainHash>;
  using Operations = DefUseChain::Operations;

  /// Create all def-use chains rooted at \p start and terminated by \p end.
  void createChains(Operation *start, Operation *end);

  DefUseChains &getChainsMutable() { return chains; }
  const DefUseChains &getChains() const { return chains; }

  /// Prune overlapping def-use chains.
  /// Include the start operation unless \p includeStart is false.
  void pruneOverlappingChains(bool includeStart);

private:
  /// Find all def-use paths from \p op to \p end and add them to \p allPaths.
  /// Note: \p path is the current def-use path being constructed.
  void findAllPaths(Operation *op, Operation *end, Operations &path,
                    SmallVectorImpl<Operations> &allPaths);

  /// Add the users of \p op to \p users.
  /// Note: \p path is the def-use path being constructed.
  void addUsers(Operation *op, const Operations path, Operations &users) const;

  /// Return def-use chains that overlap.
  DefUseChains getOverlappingChains() const;

  DefUseChains chains;
};

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_UTILS_DEFUSECHAIN_H
