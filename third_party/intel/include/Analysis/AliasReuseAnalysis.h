#ifndef TRITON_INTEL_ANALYSIS_ALIASREUSEANALYSIS_H
#define TRITON_INTEL_ANALYSIS_ALIASREUSEANALYSIS_H

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

/// Function-local alias-reuse analysis.
///
/// For a given `tt.load`, returns all other memory-effect ops
/// (tt.load / tt.store / tt.atomic_rmw / tt.atomic_cas / tt.descriptor_load /
/// tt.descriptor_store / tt.descriptor_gather / tt.descriptor_scatter /
/// tt.descriptor_reduce) in the same function whose pointer operand may alias
/// the load's pointer, in program order.
///
/// Construct once per `tt::FuncOp`; query per `tt::LoadOp`. Cross-function
/// aliasing (through call sites) is out of scope — Triton backends inline
/// before this analysis runs.
///
/// Internally, the analysis runs MLIR's sparse forward dataflow framework to
/// compute, for every pointer-typed SSA value, the set of "root" values
/// (entry-block pointer arguments or unresolved opaque producers) it may
/// resolve to. Two pointers MayAlias iff their root sets intersect. An empty
/// root set means the pointer origin is opaque; any two opaque-pointer ops
/// MayAlias each other.
///
/// If the dataflow solver fails to initialize, every query conservatively
/// returns the full set of memory-effect ops in the function (minus self).
class AliasReuseAnalysis {
public:
  explicit AliasReuseAnalysis(triton::FuncOp func);

  /// Returns all memory-effect ops in the same function whose pointer operand
  /// may alias `queryOp`'s pointer, in program order. Excludes `queryOp`
  /// itself. An empty result means no aliasing peer was found.
  /// `queryOp` must be one of the tracked op types (tt.load, tt.store,
  /// tt.atomic_rmw, tt.atomic_cas, or any tt.descriptor_* op); passing any
  /// other op returns an empty ArrayRef.
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
  /// `true` when dataflow initialization failed — forces every query to return
  /// the full memOps list (minus self). This is a correctness fallback, not
  /// policy.
  bool pessimizeAll = false;

  /// Maps a pointer-typed Value to the set of root values it may resolve to.
  /// An empty set means "unresolved/opaque" — such pointers MayAlias any other
  /// pointer whose root set is also empty.
  llvm::DenseMap<mlir::Value, llvm::DenseSet<mlir::Value>> pointerRoots;

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

#endif // TRITON_INTEL_ANALYSIS_ALIASREUSEANALYSIS_H
