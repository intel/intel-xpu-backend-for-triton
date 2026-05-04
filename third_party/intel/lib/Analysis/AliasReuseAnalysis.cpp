#include "intel/include/Analysis/AliasReuseAnalysis.h"

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
/// resolve to. An empty set is the pessimistic (uninitialized / unresolved)
/// value: treated as "unknown — MayAlias everything that is also unresolved".
///
/// `join` is set union (classic may-alias lattice). A non-empty set strictly
/// dominates the empty set.
class AliasInfo {
public:
  AliasInfo() = default;
  explicit AliasInfo(Value root) { roots.insert(root); }

  void insert(Value v) { roots.insert(v); }

  const DenseSet<Value> &getRoots() const { return roots; }
  bool empty() const { return roots.empty(); }

  bool operator==(const AliasInfo &other) const { return roots == other.roots; }

  /// Classic MayAlias join: set union.
  static AliasInfo join(const AliasInfo &lhs, const AliasInfo &rhs) {
    if (lhs == rhs)
      return lhs;
    AliasInfo ret;
    for (Value v : lhs.roots)
      ret.insert(v);
    for (Value v : rhs.roots)
      ret.insert(v);
    return ret;
  }

  void print(raw_ostream &os) const {
    os << "roots = {";
    llvm::interleaveComma(roots, os, [&](Value v) { v.print(os); });
    os << "}";
  }

private:
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

  /// Seeds the lattice for an entry state. When the anchor is a pointer-typed
  /// entry-block function argument, seed it with itself as its own root. All
  /// other entry states get the pessimistic empty set.
  void setToEntryState(dataflow::Lattice<AliasInfo> *lattice) override {
    Value anchor = lattice->getAnchor();
    if (isPointerLike(anchor.getType()) && isEntryBlockFuncArg(anchor)) {
      propagateIfChanged(lattice, lattice->join(AliasInfo(anchor)));
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

    // Any other op: pointer-typed results get the pessimistic empty set
    // (unresolved / opaque producer). Non-pointer results are irrelevant to
    // the alias question; we still push the pessimistic state to keep the
    // lattice monotone.
    setAllToEntryStates(results);
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
/// For descriptor ops, resolves the base pointer via the defining
/// tt.make_tensor_descriptor op. Returns null Value for anything else.
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
            if (auto makeDesc = desc.getDefiningOp<tt::MakeTensorDescOp>())
              return makeDesc.getBase();
            return Value();
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

/// Returns a reference to the root set for `ptr`, or a static empty set if
/// `ptr` is not in the map.
const DenseSet<Value> &getOrEmpty(const DenseMap<Value, DenseSet<Value>> &map,
                                  Value ptr) {
  static const DenseSet<Value> empty;
  auto it = map.find(ptr);
  return it != map.end() ? it->second : empty;
}

/// Two pointer root sets MayAlias iff:
///   - Both are empty (both opaque — unknown origin, conservatively MayAlias).
///   - Both are non-empty and their root sets intersect.
/// One empty, one non-empty: the non-empty pointer has a known distinct origin,
/// so we treat it as NoAlias vs. the opaque pointer.
static bool mayAlias(const DenseSet<Value> &a, const DenseSet<Value> &b) {
  if (a.empty() && b.empty())
    return true;
  if (a.empty() || b.empty())
    return false;
  for (const Value &v : a)
    if (b.count(v))
      return true;
  return false;
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
    DenseSet<Value> roots;
    if (lattice) {
      const AliasInfo &info = lattice->getValue();
      for (Value r : info.getRoots())
        roots.insert(r);
    }
    pointerRoots.try_emplace(ptr, std::move(roots));
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

  const DenseSet<Value> &qRoots = getOrEmpty(pointerRoots, qPtr);

  for (auto [mPtr, mOp] : llvm::zip(memOpPtrs, memOps)) {
    if (mOp == queryOp)
      continue;
    const DenseSet<Value> &mRoots = getOrEmpty(pointerRoots, mPtr);
    if (mayAlias(qRoots, mRoots))
      peers.push_back(mOp);
  }

  LDBG("getAliasingMemOps(" << *queryOp << "): " << peers.size() << " peer(s)");
  return peers;
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
