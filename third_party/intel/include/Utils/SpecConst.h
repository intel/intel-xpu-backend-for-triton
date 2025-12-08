#ifndef TRITON_INTEL_UTILS_SPECCONST_H
#define TRITON_INTEL_UTILS_SPECCONST_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::intel {

FailureOr<Value> buildSpecConstBasedValue(Value baseValue, Operation *anchorOp,
                                          Location loc,
                                          PatternRewriter &rewriter);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_UTILS_SPECCONST_H
