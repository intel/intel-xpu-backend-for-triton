#include "intel/include/Analysis/AliasReuseAnalysis.h"

#include "intel/include/Utils/Utility.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "intel-alias-reuse"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::intel {

namespace {

//===----------------------------------------------------------------------===//
// AliasInfo lattice element
//===----------------------------------------------------------------------===//

/// Lattice element tracking the set of "root" SSA values that a pointer may
/// resolve to. Three states:
///   - Bottom (⊥): default-constructed, uninitialized. No information yet.
///   - Known: a concrete, possibly multi-element set of root SSA values.
///   - Unknown (⊤): the pointer has an unresolved/opaque origin and must be
///     assumed to MayAlias any other pointer.
///
/// Lattice order: ⊥ ⊑ Known ⊑ ⊤. `join` is set union for two Known states;
/// joining with ⊤ yields ⊤; ⊥ is the identity under join.
class AliasInfo {
public:
  AliasInfo() = default;
  explicit AliasInfo(Value root) { roots.insert(root); }

  /// Returns the top lattice element (unknown/opaque pointer origin).
  static AliasInfo getUnknown() { return AliasInfo(/*unknown=*/true); }

  void insert(Value v) {
    assert(!unknown && "cannot insert into unknown AliasInfo");
    roots.insert(v);
  }

  const DenseSet<Value> &getRoots() const { return roots; }
  bool isUnknown() const { return unknown; }
  bool isUninitialized() const { return !unknown && roots.empty(); }

  bool operator==(const AliasInfo &other) const {
    return unknown == other.unknown && roots == other.roots;
  }

  /// May-alias join: ⊤ absorbs, ⊥ is identity, otherwise set union.
  static AliasInfo join(const AliasInfo &lhs, const AliasInfo &rhs) {
    if (lhs.unknown || rhs.unknown)
      return getUnknown();
    if (lhs == rhs)
      return lhs;
    AliasInfo ret;
    for (Value v : lhs.roots)
      ret.roots.insert(v);
    for (Value v : rhs.roots)
      ret.roots.insert(v);
    return ret;
  }

  void print(raw_ostream &os) const {
    if (unknown) {
      os << "unknown";
      return;
    }
    os << "roots = {";
    llvm::interleaveComma(roots, os, [&](Value v) { v.print(os); });
    os << "}";
  }

private:
  explicit AliasInfo(bool unknown) : unknown(unknown) {}

  bool unknown = false;
  DenseSet<Value> roots;
};

//===----------------------------------------------------------------------===//
// Alias-pointer dataflow analysis
//===----------------------------------------------------------------------===//

/// Sparse forward dataflow analysis that propagates `AliasInfo` through the
/// pointer-producing op set enumerated in the plan. SCF forwarding (iter_args,
/// yields, while-region plumbing) is handled by the framework via
/// `RegionBranchOpInterface`.
class AliasDataflow : public dataflow::SparseForwardDataFlowAnalysis<
                          dataflow::Lattice<AliasInfo>> {
public:
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AliasInfo>>::SparseForwardDataFlowAnalysis;
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AliasInfo>>::getLatticeElement;

  /// Seeds the lattice for an entry state. Pointer-typed entry-block function
  /// arguments seed with themselves (known root). Any other pointer-typed
  /// anchor with no defining op in this analysis is seeded as Unknown (⊤).
  /// Non-pointer anchors get Bottom (⊥); they are irrelevant to the alias
  /// question but kept monotone.
  void setToEntryState(dataflow::Lattice<AliasInfo> *lattice) override {
    Value anchor = lattice->getAnchor();
    if (isPointerLike(anchor.getType())) {
      if (isEntryBlockFuncArg(anchor)) {
        propagateIfChanged(lattice, lattice->join(AliasInfo(anchor)));
        return;
      }
      // Pointer-typed anchor with unresolved origin (e.g., a block argument
      // not handled by the framework, or a value outside this analysis's
      // purview). Must be treated as MayAlias anything.
      propagateIfChanged(lattice, lattice->join(AliasInfo::getUnknown()));
      return;
    }
    propagateIfChanged(lattice, lattice->join(AliasInfo()));
  }

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
                 ArrayRef<dataflow::Lattice<AliasInfo> *> results) override {
    // Pointer-producing passthrough ops: forward operand 0's AliasInfo to
    // every result.
    if (isa<tt::AddPtrOp, tt::SplatOp, tt::BroadcastOp, tt::BitcastOp,
            tt::ExpandDimsOp, tt::ReshapeOp, tt::TransOp, ttg::ConvertLayoutOp>(
            op)) {
      if (operands.empty()) {
        setAllToEntryStates(results);
        return success();
      }
      const AliasInfo &in = operands[0]->getValue();
      for (auto *result : results)
        propagateIfChanged(result, result->join(in));
      return success();
    }

    // Any other op: pointer-typed results come from an opaque producer
    // (e.g., arith.select between two pointers) and must be treated as
    // Unknown (⊤) — MayAlias anything. Non-pointer results stay at Bottom.
    for (auto *result : results) {
      if (isPointerLike(result->getAnchor().getType()))
        propagateIfChanged(result, result->join(AliasInfo::getUnknown()));
      else
        propagateIfChanged(result, result->join(AliasInfo()));
    }
    return success();
  }

private:
  /// True if `type` is a Triton pointer or a tensor of Triton pointers.
  static bool isPointerLike(Type type) {
    if (isa<tt::PointerType>(type))
      return true;
    if (auto tensorTy = dyn_cast<RankedTensorType>(type))
      return isa<tt::PointerType>(tensorTy.getElementType());
    return false;
  }

  /// True if `v` is a block argument of the entry block of its parent
  /// FunctionOpInterface.
  static bool isEntryBlockFuncArg(Value v) {
    auto blockArg = dyn_cast<BlockArgument>(v);
    if (!blockArg)
      return false;
    Block *owner = blockArg.getOwner();
    if (!owner)
      return false;
    auto funcOp = dyn_cast_or_null<FunctionOpInterface>(owner->getParentOp());
    if (!funcOp)
      return false;
    Region &body = funcOp.getFunctionBody();
    return !body.empty() && owner == &body.front();
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the pointer operand for a tracked memory-effect op.
/// For descriptor ops, traces the base pointer through SCF iter_args,
/// yields, `scf.if`, `arith.select`, and unrealized casts via
/// `findDefiningOpOfType<tt::MakeTensorDescOp>`. When that trace fails
/// (e.g., the descriptor comes from a call, a region with mismatched
/// branches, or any unmodeled producer), returns the descriptor Value
/// itself. That value is never seeded by the dataflow, so the snapshot
/// comes back as uninitialized and the op is conservatively treated as
/// Unknown — MayAlias everything — rather than being silently dropped.
/// Returns a null Value for anything else.
Value getMemOpPointer(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case<tt::LoadOp>([](auto op) { return op.getPtr(); })
      .Case<tt::StoreOp>([](auto op) { return op.getPtr(); })
      .Case<tt::AtomicRMWOp>([](auto op) { return op.getPtr(); })
      .Case<tt::AtomicCASOp>([](auto op) { return op.getPtr(); })
      .Case<tt::DescriptorLoadOp, tt::DescriptorStoreOp, tt::DescriptorGatherOp,
            tt::DescriptorScatterOp, tt::DescriptorReduceOp>(
          [](auto op) -> Value {
            Value desc = op.getDesc();
            if (std::optional<tt::MakeTensorDescOp> makeDesc =
                    findDefiningOpOfType<tt::MakeTensorDescOp>(desc))
              return makeDesc->getBase();
            // Couldn't resolve to a MakeTensorDescOp — keep the op alive
            // by returning the descriptor itself as an opaque sentinel.
            return desc;
          })
      .Default([](auto) { return Value(); });
}

/// True iff `op` is one of the memory-effect ops this analysis tracks.
bool isTrackedMemOp(Operation *op) {
  return isa<tt::LoadOp, tt::StoreOp, tt::AtomicRMWOp, tt::AtomicCASOp,
             tt::DescriptorLoadOp, tt::DescriptorStoreOp,
             tt::DescriptorGatherOp, tt::DescriptorScatterOp,
             tt::DescriptorReduceOp>(op);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// AliasReuseAnalysis
//===----------------------------------------------------------------------===//

AliasReuseAnalysis::AliasReuseAnalysis(tt::FuncOp func) {
  // Collect every memory-effect op in program order.
  func.walk([&](Operation *op) {
    if (!isTrackedMemOp(op))
      return;
    Value ptr = getMemOpPointer(op);
    if (!ptr)
      return;
    memOps.push_back(op);
    memOpPtrs.push_back(ptr);
  });

  // Run the dataflow solver.
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  AliasDataflow *analysis = solver->load<AliasDataflow>();
  if (failed(solver->initializeAndRun(func))) {
    LDBG("dataflow solver failed; pessimizing every query");
    pessimizeAll = true;
    return;
  }

  // Snapshot AliasInfo for every tracked memory-op pointer.
  auto snapshotRoots = [&](Value ptr) {
    if (pointerRoots.contains(ptr))
      return;
    const auto *lattice = analysis->getLatticeElement(ptr);
    RootSnapshot snap;
    if (!lattice) {
      // No lattice element computed — conservatively Unknown.
      snap.unknown = true;
    } else {
      const AliasInfo &info = lattice->getValue();
      if (info.isUnknown() || info.isUninitialized()) {
        // Uninitialized (bottom) also treated as Unknown: the solver never
        // resolved this pointer, so we cannot prove NoAlias.
        snap.unknown = true;
      } else {
        for (Value r : info.getRoots())
          snap.roots.insert(r);
      }
    }
    pointerRoots.try_emplace(ptr, std::move(snap));
  };
  for (Value ptr : memOpPtrs)
    snapshotRoots(ptr);
}

ArrayRef<Operation *>
AliasReuseAnalysis::getAliasingMemOps(Operation *queryOp) const {
  auto [it, inserted] = resultCache.try_emplace(queryOp);
  if (!inserted)
    return it->second;

  SmallVector<Operation *> &peers = it->second;

  Value qPtr = getMemOpPointer(queryOp);
  if (!qPtr)
    return peers; // not a tracked op — return empty

  if (pessimizeAll) {
    for (Operation *op : memOps)
      if (op != queryOp)
        peers.push_back(op);
    return peers;
  }

  for (auto [mPtr, mOp] : llvm::zip(memOpPtrs, memOps)) {
    if (mOp == queryOp)
      continue;
    if (mayAlias(qPtr, mPtr))
      peers.push_back(mOp);
  }

  LDBG("getAliasingMemOps(" << *queryOp << "): " << peers.size() << " peer(s)");
  return peers;
}

bool AliasReuseAnalysis::mayAlias(Value a, Value b) const {
  auto itA = pointerRoots.find(a);
  auto itB = pointerRoots.find(b);
  // A pointer with no entry is treated as Unknown (it was never seeded).
  bool aUnknown = itA == pointerRoots.end() || itA->second.unknown;
  bool bUnknown = itB == pointerRoots.end() || itB->second.unknown;
  if (aUnknown || bUnknown)
    return true;
  const DenseSet<Value> &ra = itA->second.roots;
  const DenseSet<Value> &rb = itB->second.roots;
  for (const Value &v : ra)
    if (rb.count(v))
      return true;
  return false;
}

ArrayRef<Operation *>
AliasReuseAnalysis::getAliasingMemOps(tt::LoadOp loadOp) const {
  return getAliasingMemOps(loadOp.getOperation());
}

ArrayRef<Operation *>
AliasReuseAnalysis::getAliasingMemOps(tt::DescriptorLoadOp loadOp) const {
  return getAliasingMemOps(loadOp.getOperation());
}

} // namespace mlir::triton::intel
