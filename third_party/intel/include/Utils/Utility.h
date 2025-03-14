#ifndef TRITON_INTEL_UTILS_UTILITY_H
#define TRITON_INTEL_UTILS_UTILITY_H

#include <mlir/IR/Value.h>

namespace mlir::triton::intel {

// This function folds the `op` operation and returns the constant value if it
// has successfully folded to a constant. Otherwise, it returns `std::nullopt`.
std::optional<int64_t> getFoldedConstantValue(Operation *op);

// Return true if the `val` value is a constant containing a value equal to
// expected.
bool isConstant(Value val, int64_t expected);

mlir::Value getFinalValue(Value value);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_UTILS_UTILITY_H
