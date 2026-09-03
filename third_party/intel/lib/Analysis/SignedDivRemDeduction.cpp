#include "intel/include/Analysis/SignedDivRemDeduction.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Analysis/AxisInfo.h"

#include <numeric>

using namespace mlir;
namespace tt = mlir::triton;

namespace {

//===----------------------------------------------------------------------===//
// The deduction mirror
//===----------------------------------------------------------------------===//
//
// The three helpers below mirror their counterparts in
// lib/Analysis/AxisInfo.cpp
// (`gcd`, `AxisInfoVisitor::isContiguousDim`, `AxisInfoVisitor::isConstantDim`)
// so that signedDivRemDeductionApplies() can restate the disabled deduction
// conditions verbatim.

/// Greatest common divisor treating zero as the identity, matching the variadic
/// `gcd` in lib/Analysis/AxisInfo.cpp.
template <typename... Args> int64_t gcd(int64_t a, int64_t b, Args... args) {
  if (a == 0)
    return b;
  if (b == 0)
    return a;
  if constexpr (sizeof...(args) == 0)
    return std::gcd(a, b);
  else
    return gcd(std::gcd(a, b), args...);
}

bool isContiguousDim(const tt::AxisInfo &info, ArrayRef<int64_t> shape,
                     int dim) {
  return info.getContiguity(dim) == shape[dim];
}

bool isConstantDim(const tt::AxisInfo &info, ArrayRef<int64_t> shape, int dim) {
  return info.getConstancy(dim) == shape[dim];
}

} // namespace

namespace mlir::triton::intel {

bool signedDivRemDeductionApplies(Operation *op, const AxisInfo &lhs,
                                  const AxisInfo &rhs) {
  if (!isa<arith::DivSIOp, arith::RemSIOp>(op))
    return false;

  auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resTy)
    return false;

  // Mirrors DivOpAxisInfoVisitor::getConstancy and
  // RemOpAxisInfoVisitor::getContiguity: both require a contiguous dividend and
  // a constant divisor along the dimension, and both deduce
  // gcd(contiguity, divisibility, divisibility). The sound value is 1 in either
  // case (a contiguous dimension has constancy 1), so the deduction is only
  // worth something when that gcd exceeds 1.
  ArrayRef<int64_t> shape = resTy.getShape();
  for (int d = 0, rank = shape.size(); d < rank; ++d) {
    if (isContiguousDim(lhs, shape, d) && isConstantDim(rhs, shape, d) &&
        gcd(lhs.getContiguity(d), lhs.getDivisibility(d),
            rhs.getDivisibility(d)) > 1)
      return true;
  }

  return false;
}

} // namespace mlir::triton::intel
