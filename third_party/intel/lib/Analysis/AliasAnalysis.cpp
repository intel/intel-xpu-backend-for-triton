#include "intel/include/Analysis/AliasAnalysis.h"

#include "intel/include/Utils/Utility.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "intel-alias"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::intel {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// True if `type` is a Triton pointer, a Triton tensor descriptor, or a
/// tensor of Triton pointers. Tensor descriptors are included so that the
/// dataflow framework propagates Unknown through descriptors that escape
/// unresolvable control flow (e.g., mismatched `scf.if` branches), keeping
/// the opaque-sentinel path in `getMemOpPointer` sound. Used by both the
/// dataflow seeding (see `AliasDataflow`) and the interface-tracked pointer
/// resolution in `getMemOpPointer`; keeping a single definition ensures
/// op-collection and dataflow seeding agree on what "pointer-like" means.
static bool isPointerLike(Type type) {
  if (isa<tt::PointerType, tt::TensorDescType>(type))
    return true;
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return isa<tt::PointerType>(tensorTy.getElementType());
  return false;
}

/// True iff `op` is one of the "modeled" memory-effect ops with a known
/// pointer-resolution rule: `tt.load/store/atomic_rmw/atomic_cas` and the
/// five `tt.descriptor_*` ops. Interface-tracked ops (any
/// `MemoryEffectOpInterface` op with a Read/Write effect) are additionally
/// handled via `hasReadOrWriteEffect` + `getMemOpPointer`'s default branch;
/// see the ctor.
static bool isTrackedMemOp(Operation *op) {
  return isa<tt::LoadOp, tt::StoreOp, tt::AtomicRMWOp, tt::AtomicCASOp,
             tt::DescriptorLoadOp, tt::DescriptorStoreOp,
             tt::DescriptorGatherOp, tt::DescriptorScatterOp,
             tt::DescriptorReduceOp>(op);
}

/// True iff `op` implements `MemoryEffectOpInterface` and has at least one
/// `MemoryEffects::Read` or `MemoryEffects::Write` effect. Used to identify
/// interface-tracked peers alongside the nine modeled op types.
static bool hasReadOrWriteEffect(Operation *op) {
  auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffects)
    return false;
  return memEffects.hasEffect<MemoryEffects::Read>() ||
         memEffects.hasEffect<MemoryEffects::Write>();
}

/// Returns the pointer operand for a tracked memory-effect op.
/// For the nine "modeled" op types (tt.load/store/atomic_rmw/atomic_cas and
/// the five tt.descriptor_* ops), returns the pointer operand directly. For
/// descriptor ops, traces the base pointer through SCF iter_args, yields,
/// `scf.if`, `arith.select`, and unrealized casts via
/// `findDefiningOpOfType<tt::MakeTensorDescOp>`. When that trace fails
/// (e.g., the descriptor comes from a call, a region with mismatched
/// branches, or any unmodeled producer), returns the descriptor Value
/// itself. That value is never seeded by the dataflow, so the snapshot
/// comes back as uninitialized and the op is conservatively treated as
/// Unknown — MayAlias everything — rather than being silently dropped.
///
/// For any other op implementing `MemoryEffectOpInterface` with a Read or
/// Write effect, returns the first pointer-like operand (per
/// `isPointerLike`). Returns a null Value if the op has no pointer-like
/// operand; the caller (ctor) still tracks such ops so they act as
/// universal MayAlias peers.
///
/// Returns a null Value for ops that are neither modeled nor interface-
/// tracked.
static Value getMemOpPointer(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case<tt::LoadOp>([](auto op) { return op.getPtr(); })
      .Case<tt::StoreOp>([](auto op) { return op.getPtr(); })
      .Case<tt::AtomicRMWOp>([](auto op) { return op.getPtr(); })
      .Case<tt::AtomicCASOp>([](auto op) { return op.getPtr(); })
      .Case<tt::DescriptorLoadOp, tt::DescriptorStoreOp, tt::DescriptorGatherOp,
            tt::DescriptorScatterOp, tt::DescriptorReduceOp>(
          [](auto op) -> Value {
            Value desc = op.getDesc();
            // TODO(#6862): once the worklist-based findAllMakeTensorDescOps
            // lands, iterate all reachable MakeTensorDescOps and union their
            // base pointers, degrading to Unknown on any irresolvable base.
            // The single-op resolution here may pick a stale init-args
            // descriptor for loop-carried descs.
            if (std::optional<tt::MakeTensorDescOp> makeDesc =
                    findDefiningOpOfType<tt::MakeTensorDescOp>(desc))
              return makeDesc->getBase();
            // Couldn't resolve to a MakeTensorDescOp — keep the op alive
            // by returning the descriptor itself as an opaque sentinel.
            return desc;
          })
      .Default([](Operation *op) -> Value {
        // Interface-tracked path: any op with a Read or Write effect. Find
        // the single pointer-like operand; if none, return null (the op is
        // still tracked by the ctor as a universal peer).
        auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
        if (!memEffects)
          return Value();
        if (!memEffects.hasEffect<MemoryEffects::Read>() &&
            !memEffects.hasEffect<MemoryEffects::Write>())
          return Value();
        auto isPtr = [](Value v) { return isPointerLike(v.getType()); };
        assert(llvm::count_if(op->getOperands(), isPtr) <= 1 &&
               "interface-tracked memory-effect op must have at most one "
               "pointer-like operand");
        auto it = llvm::find_if(op->getOperands(), isPtr);
        return it == op->operand_end() ? Value() : *it;
      });
}

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

} // anonymous namespace

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

AliasAnalysis::AliasAnalysis(tt::FuncOp func) {
  // Collect every memory-effect op in program order.
  func.walk([&](Operation *op) {
    bool modeled = isTrackedMemOp(op);
    bool interfaceTracked = !modeled && hasReadOrWriteEffect(op);
    if (!modeled && !interfaceTracked)
      return;
    Value ptr = getMemOpPointer(op);
    // For modeled ops, `getMemOpPointer` is guaranteed non-null: descriptor
    // ops fall back to the descriptor value as an opaque sentinel, and the
    // other four types (load/store/atomic_rmw/atomic_cas) directly return
    // a required operand. For interface-tracked ops, `ptr` may be null:
    // the op has no pointer-like operand and will be treated as a universal
    // MayAlias peer by `getAliasingMemOps`.
    assert((!modeled || ptr) && "modeled op returned null pointer");
    memOps.push_back(op);
    memOpPtrs.push_back(ptr); // may be null for interface-tracked ops
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
    // Interface-tracked ops with no pointer-like operand are tracked with a
    // null ptr; they act as universal MayAlias peers and are handled
    // purely by the short-circuit in `getAliasingMemOps`, not by the
    // pointer-roots map.
    if (!ptr)
      return;
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
AliasAnalysis::getAliasingMemOps(Operation *queryOp) const {
  auto [it, inserted] = resultCache.try_emplace(queryOp);
  if (!inserted)
    return it->second;

  SmallVector<Operation *> &peers = it->second;

  // A query op is "tracked" iff it was collected by the ctor into `memOps`.
  // This covers both modeled ops (the 9 specific types) and interface-
  // tracked ops (MemoryEffectOpInterface with Read/Write). Ops outside this
  // set (e.g., arith.addi) return an empty peer list.
  bool queryTracked = llvm::is_contained(memOps, queryOp);
  if (!queryTracked)
    return peers;

  Value qPtr = getMemOpPointer(queryOp);
  // qPtr may be null: interface-tracked op with no pointer-like operand.
  // Such an op acts as a universal MayAlias peer in both directions.

  if (pessimizeAll) {
    for (Operation *op : memOps)
      if (op != queryOp)
        peers.push_back(op);
    return peers;
  }

  for (auto [mPtr, mOp] : llvm::zip(memOpPtrs, memOps)) {
    if (mOp == queryOp)
      continue;
    // Null qPtr or null peer ptr: one side has no resolvable pointer, so
    // MayAlias conservatively. Otherwise consult the dataflow snapshot.
    if (!qPtr || !mPtr || mayAlias(qPtr, mPtr))
      peers.push_back(mOp);
  }

  LDBG("getAliasingMemOps(" << *queryOp << "): " << peers.size() << " peer(s)");
  return peers;
}

bool AliasAnalysis::mayAlias(Value a, Value b) const {
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
AliasAnalysis::getAliasingMemOps(tt::LoadOp loadOp) const {
  return getAliasingMemOps(loadOp.getOperation());
}

ArrayRef<Operation *>
AliasAnalysis::getAliasingMemOps(tt::DescriptorLoadOp loadOp) const {
  return getAliasingMemOps(loadOp.getOperation());
}

} // namespace mlir::triton::intel
