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
/// *temporal* stride — how a value's address shifts between successive
/// iterations of a given `LoopLikeOpInterface`.  The IV-stride table is
/// keyed by the enclosing loop op; an absent entry is semantically
/// equivalent to an all-zero vector (i.e. the value does not depend on
/// that loop's IV).
class StrideInfo {
public:
  using DimVectorT = SmallVector<int64_t>;

  StrideInfo() = default;
  explicit StrideInfo(ArrayRef<int64_t> stride) : stride(stride) {}
  StrideInfo(DimVectorT spatial,
             DenseMap<LoopLikeOpInterface, DimVectorT> ivStrides)
      : stride(std::move(spatial)), ivStrides(std::move(ivStrides)) {}

  /// Spatial stride along `dim`: how many elements apart consecutive lanes
  /// along that tensor axis are within a single loaded tile.  Values follow
  /// the lattice convention: -1 = unknown, 0 = broadcast/constant along the
  /// axis, >0 = known element-count stride.
  int64_t getStride(size_t dim) const { return stride[dim]; }

  /// Full per-axis spatial-stride vector.  Size matches `getRank()`.
  const DimVectorT &getStride() const { return stride; }

  unsigned getRank() const { return stride.size(); }

  bool operator==(const StrideInfo &other) const {
    return stride == other.stride && ivStrides == other.ivStrides;
  }

  static StrideInfo getPessimisticValueState(Value value);
  static StrideInfo join(const StrideInfo &lhs, const StrideInfo &rhs);

  void print(raw_ostream &os) const;

  /// Temporal stride of this value along `dim` with respect to `loop`'s IV.
  /// Returns 0 when `loop` is not tracked for this value (the consumer
  /// interprets "not tracked" as "does not depend on this loop's IV", which
  /// is correct for values defined outside the loop and is the conservative
  /// answer for values we have not yet propagated into).
  int64_t getIVStride(LoopLikeOpInterface loop, size_t dim) const;

  /// Full per-axis vector for `loop`.  Returns `nullptr` when `loop` has no
  /// entry (treated as all-zero by consumers).
  const DimVectorT *getIVStride(LoopLikeOpInterface loop) const;

  /// Enumerate the loops with tracked IV-stride columns.  Used by visitors
  /// to fold over both operands' loop sets.
  const DenseMap<LoopLikeOpInterface, DimVectorT> &getIVStrides() const {
    return ivStrides;
  }

private:
  DimVectorT stride;                                   // spatial
  DenseMap<LoopLikeOpInterface, DimVectorT> ivStrides; // per-loop IV
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
