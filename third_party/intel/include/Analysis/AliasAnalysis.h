#ifndef TRITON_INTEL_ANALYSIS_ALIASANALYSIS_H
#define TRITON_INTEL_ANALYSIS_ALIASANALYSIS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
namespace triton {
class FuncOp;
class LoadOp;
class DescriptorLoadOp;
} // namespace triton
} // namespace mlir

namespace mlir::triton::intel {

/// Function-local alias analysis.
///
/// For a given memory-effect op, returns all other memory-effect ops in
/// the same function whose pointer operand may alias the query op's
/// pointer, in program order. Tracks:
///   - "Modeled" ops with known pointer resolution:
///     `tt.load/store/atomic_rmw/atomic_cas` and the five
///     `tt.descriptor_*` ops (base resolved through
///     `findDefiningOpOfType<MakeTensorDescOp>`).
///   - Any other op implementing `MemoryEffectOpInterface` with at
///     least one `MemoryEffects::Read` or `MemoryEffects::Write`
///     effect. For these, the pointer is the first pointer-like
///     operand; if none exists, the op acts as a universal peer
///     (MayAlias with every tracked op).
///
/// Construct once per `tt::FuncOp`; query per `tt::LoadOp`. Cross-function
/// aliasing (through call sites) is out of scope — Triton backends inline
/// before this analysis runs.
///
/// Internally, the analysis runs MLIR's sparse forward dataflow framework to
/// compute, for every pointer-typed SSA value, either:
///   - a set of "root" SSA values (entry-block pointer arguments) it may
///     resolve to, or
///   - an "unknown" marker, meaning the origin is opaque (e.g., passes
///     through `arith.select`, or any producer not in the passthrough set).
///
/// Two pointers MayAlias iff either has the unknown marker, or their root
/// sets intersect. An unknown pointer conservatively MayAlias every tracked
/// pointer.
///
/// If the dataflow solver fails to initialize, every query conservatively
/// returns the full set of memory-effect ops in the function (minus self).
class AliasAnalysis {
public:
  explicit AliasAnalysis(triton::FuncOp func);

  /// Returns all memory-effect ops in the same function whose pointer operand
  /// may alias `queryOp`'s pointer, in program order. Excludes `queryOp`
  /// itself. An empty result means no aliasing peer was found.
  /// `queryOp` must be a tracked op: one of the "modeled" types (tt.load,
  /// tt.store, tt.atomic_rmw, tt.atomic_cas, any tt.descriptor_*) or an op
  /// implementing `MemoryEffectOpInterface` with a Read or Write effect;
  /// passing any other op returns an empty ArrayRef.
  /// Callers use `isa<tt::StoreOp>(op)` / `isa<tt::AtomicRMWOp>(op)` /
  /// `isa<tt::DescriptorLoadOp>(op)` etc. to classify peers.
  llvm::ArrayRef<mlir::Operation *>
  getAliasingMemOps(mlir::Operation *queryOp) const;

  /// Convenience overloads for typed ops.
  llvm::ArrayRef<mlir::Operation *>
  getAliasingMemOps(triton::LoadOp loadOp) const;
  llvm::ArrayRef<mlir::Operation *>
  getAliasingMemOps(triton::DescriptorLoadOp loadOp) const;

private:
  /// Returns true if the snapshots for pointers `a` and `b` may alias.
  /// Either snapshot being `unknown` (opaque origin or missing from the map)
  /// forces MayAlias; otherwise their root sets must intersect.
  bool mayAlias(mlir::Value a, mlir::Value b) const;

  /// `true` when dataflow initialization failed — forces every query to return
  /// the full memOps list (minus self). This is a correctness fallback, not
  /// policy.
  bool pessimizeAll = false;

  /// Snapshot of the dataflow lattice element for a tracked pointer.
  struct RootSnapshot {
    /// Set of root SSA values this pointer may resolve to. Only meaningful
    /// when `unknown == false`.
    llvm::DenseSet<mlir::Value> roots;
    /// If true, this pointer has opaque/unresolved origin and must be
    /// treated as MayAlias-everything.
    bool unknown = false;
  };

  /// Maps a pointer-typed Value to its lattice snapshot.
  llvm::DenseMap<mlir::Value, RootSnapshot> pointerRoots;

  /// The pointer operand of every tracked memory-effect op in the function.
  /// Parallel to `memOps`.
  llvm::SmallVector<mlir::Value> memOpPtrs;

  /// Every memory-effect op in the function, in program order. Parallel to
  /// `memOpPtrs`.
  llvm::SmallVector<mlir::Operation *> memOps;

  /// Memoizes peer sets keyed by the query op. Lazily populated.
  mutable llvm::DenseMap<mlir::Operation *,
                         llvm::SmallVector<mlir::Operation *>>
      resultCache;
};

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_ALIASANALYSIS_H
