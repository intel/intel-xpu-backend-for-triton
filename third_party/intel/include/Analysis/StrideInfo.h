#ifndef TRITON_INTEL_ANALYSIS_STRIDEINFO_H
#define TRITON_INTEL_ANALYSIS_STRIDEINFO_H

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Analysis/Utility.h"

namespace mlir::triton::intel {

class ModuleAxisInfoAnalysis;

/// Per-dimension stride tracked by StrideAnalysis.
///   -1 = unknown, 0 = broadcast/constant, >0 = known stride.
///
/// In addition to the spatial stride (how elements within a single loaded
/// tile vary across tensor axes), StrideInfo also tracks a per-loop
/// *temporal* stride — how a value's address shifts per **unit of IV
/// increase** of a given `LoopLikeOpInterface`.  The IV-stride table is
/// keyed by the enclosing loop op.  **An absent entry is the canonical
/// and only encoding of "value does not depend on this loop's IV"** —
/// all-zero vectors are never stored.  A present entry is guaranteed to
/// carry at least one non-zero dimension.
///
/// NOTE on units: `getIVStride()` reports the delta per unit of IV, not
/// per iteration.  To obtain the per-iteration address delta, multiply
/// by the enclosing loop's step (when constant) — use
/// `getPerIterationIVStride()`, which folds this in and returns
/// `std::nullopt` for dynamic-step loops.  The native-unit convention
/// keeps the column meaningful even when the step is not a compile-time
/// constant, and mirrors the spatial column (which is also in native
/// element units).
class StrideInfo {
public:
  using DimVectorT = SmallVector<int64_t>;
  /// One entry per axis: the SSA value behind a runtime stride (see
  /// `getStrideValue`), or null where there isn't one.
  using StrideValueVectorT = SmallVector<Value>;

  StrideInfo() = default;
  explicit StrideInfo(ArrayRef<int64_t> stride) : stride(stride) {}
  StrideInfo(DimVectorT spatial,
             DenseMap<LoopLikeOpInterface, DimVectorT> ivStrides)
      : stride(std::move(spatial)), ivStrides(std::move(ivStrides)) {}
  StrideInfo(DimVectorT spatial,
             DenseMap<LoopLikeOpInterface, DimVectorT> ivStrides,
             StrideValueVectorT strideValues)
      : stride(std::move(spatial)), ivStrides(std::move(ivStrides)),
        strideValues(canonicalizeStrideValues(std::move(strideValues))) {}

  /// Spatial stride along `dim`: how many elements apart consecutive lanes
  /// along that tensor axis are within a single loaded tile.  Values follow
  /// the lattice convention: -1 = unknown, 0 = broadcast/constant along the
  /// axis, >0 = known element-count stride.
  int64_t getStride(size_t dim) const { return stride[dim]; }

  /// Full per-axis spatial-stride vector.  Size matches `getRank()`.
  const DimVectorT &getStride() const { return stride; }

  /// The SSA value behind a runtime stride along `dim`, or null if there is
  /// none. Set only when the constant stride is unknown (`getStride(dim) < 0`)
  /// but the value can still be named, so a consumer can use it instead of
  /// digging through the IR.
  Value getStrideValue(size_t dim) const {
    return dim < strideValues.size() ? strideValues[dim] : Value();
  }

  /// Full per-axis symbolic-stride vector.  Either empty (no symbolic source
  /// for any axis) or exactly `getRank()`-sized.
  const StrideValueVectorT &getStrideValues() const { return strideValues; }

  unsigned getRank() const { return stride.size(); }

  bool operator==(const StrideInfo &other) const {
    return stride == other.stride && ivStrides == other.ivStrides &&
           strideValues == other.strideValues;
  }

  /// Collapse an all-null vector to empty, so "no runtime stride" compares
  /// equal regardless of rank (keeps the lattice equality well-behaved).
  static StrideValueVectorT
  canonicalizeStrideValues(StrideValueVectorT strideValues) {
    if (llvm::all_of(strideValues, [](Value v) { return !v; }))
      return {};
    return strideValues;
  }

  static StrideInfo getPessimisticValueState(Value value);
  static StrideInfo join(const StrideInfo &lhs, const StrideInfo &rhs);

  void print(raw_ostream &os) const;

  /// Temporal stride of this value along `dim` with respect to `loop`'s IV,
  /// in IV units (delta per unit of IV increase).  Returns 0 when `loop` is
  /// not tracked for this value (the consumer interprets "not tracked" as
  /// "does not depend on this loop's IV", which is correct for values
  /// defined outside the loop and is the conservative answer for values
  /// we have not yet propagated into).
  int64_t getIVStride(LoopLikeOpInterface loop, size_t dim) const;

  /// Full per-axis IV-unit stride vector for `loop`.  Returns `nullptr` iff
  /// the value does not depend on `loop`'s IV.  A non-null result is
  /// guaranteed to have at least one non-zero dimension (asserted).
  const DimVectorT *getIVStride(LoopLikeOpInterface loop) const;

  /// Per-iteration address delta along `dim` with respect to `loop`.
  /// Equivalent to `getIVStride(loop, dim) * loop.step` when the step is
  /// a compile-time constant.  Returns `std::nullopt` when the step is
  /// dynamic, when the underlying IV-unit stride is unknown (-1), or when
  /// the constant-step multiplication would overflow `int64_t`.  Callers
  /// that need a concrete per-iteration delta should use this helper
  /// instead of multiplying by the step manually.
  std::optional<int64_t> getPerIterationIVStride(LoopLikeOpInterface loop,
                                                 size_t dim) const;

  /// Enumerate the loops with tracked IV-stride columns.  Used by visitors
  /// to fold over both operands' loop sets.
  const DenseMap<LoopLikeOpInterface, DimVectorT> &getIVStrides() const {
    return ivStrides;
  }

private:
  DimVectorT stride;                                   // spatial
  DenseMap<LoopLikeOpInterface, DimVectorT> ivStrides; // per-loop IV
  StrideValueVectorT strideValues; // symbolic spatial-stride source per axis
};

using StrideInfoMapT = DenseMap<Value, StrideInfo>;

class ModuleStrideAnalysis : public CallGraph<StrideInfoMapT> {
public:
  explicit ModuleStrideAnalysis(ModuleOp moduleOp,
                                ModuleAxisInfoAnalysis &axisInfo);

  StrideInfo *getStrideInfo(Value value);

private:
  void initialize(FunctionOpInterface funcOp);
  void update(CallOpInterface callOp, FunctionOpInterface funcOp);
  ModuleAxisInfoAnalysis &axisInfo;
};

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_STRIDEINFO_H
