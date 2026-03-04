#ifndef TRITON_INTEL_ANALYSIS_STRIDEINFO_H
#define TRITON_INTEL_ANALYSIS_STRIDEINFO_H

#include "triton/Analysis/Utility.h"

namespace mlir::triton::intel {

class ModuleAxisInfoAnalysis;

/// Per-dimension stride tracked by StrideAnalysis.
///   -1 = unknown, 0 = broadcast/constant, >0 = known stride.
class StrideInfo {
public:
  using DimVectorT = SmallVector<int64_t>;

  StrideInfo() = default;
  explicit StrideInfo(ArrayRef<int64_t> stride) : stride(stride) {}

  int64_t getStride(size_t dim) const { return stride[dim]; }
  const DimVectorT &getStride() const { return stride; }
  unsigned getRank() const { return stride.size(); }

  bool operator==(const StrideInfo &other) const {
    return stride == other.stride;
  }

  static StrideInfo getPessimisticValueState(Value value);
  static StrideInfo join(const StrideInfo &lhs, const StrideInfo &rhs);

  void print(raw_ostream &os) const {
    os << "stride = [";
    llvm::interleaveComma(stride, os);
    os << "]";
  }

private:
  DimVectorT stride;
};

using StrideInfoMapT = DenseMap<Value, StrideInfo>;

class ModuleStrideAnalysis : public CallGraph<StrideInfoMapT> {
public:
  explicit ModuleStrideAnalysis(ModuleOp moduleOp,
                                ModuleAxisInfoAnalysis *axisInfo = nullptr);

  StrideInfo *getStrideInfo(Value value);

private:
  void initialize(FunctionOpInterface funcOp);
  void update(CallOpInterface callOp, FunctionOpInterface funcOp);
  ModuleAxisInfoAnalysis *axisInfo = nullptr;
};

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_STRIDEINFO_H
