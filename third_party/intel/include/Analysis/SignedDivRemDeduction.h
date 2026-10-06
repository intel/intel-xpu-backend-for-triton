#ifndef TRITON_INTEL_ANALYSIS_SIGNEDDIVREMDEDUCTION_H
#define TRITON_INTEL_ANALYSIS_SIGNEDDIVREMDEDUCTION_H

#include "mlir/IR/Operation.h"

namespace mlir::triton {
class AxisInfo;
} // namespace mlir::triton

namespace mlir::triton::intel {

/// Returns true if `op` is an `arith.divsi` or `arith.remsi` that the AxisInfo
/// deduction would have drawn a stronger conclusion from, were its dividend
/// known to be non-negative.
///
/// The deduction is only sound for a non-negative dividend, so it is disabled
/// for the signed operations (see DivOpAxisInfoVisitor::getConstancy and
/// RemOpAxisInfoVisitor::getContiguity in lib/Analysis/AxisInfo.cpp). This
/// query mirrors those two sites, and exists so
/// TritonIntelSpeculateSignedDivRem can find the operations worth converting to
/// their unsigned form under a runtime check.
///
/// `lhs` and `rhs` are the AxisInfo of the operands.
bool signedDivRemDeductionApplies(Operation *op, const AxisInfo &lhs,
                                  const AxisInfo &rhs);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_SIGNEDDIVREMDEDUCTION_H
