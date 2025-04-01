#ifndef TRITON_INTEL_UTILS_UTILITY_H
#define TRITON_INTEL_UTILS_UTILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::triton::intel {

// Lookup for a integer constant with the given value and bitwidth in the
// current block (before the builder insertion point). Return it if found,
// otherwise create a new one.
Value findOrCreateIntConstant(Location loc, int val, unsigned bitWidth,
                              OpBuilder &builder);

// This function folds the `op` operation and returns the constant value if it
// has successfully folded to a constant. Otherwise, it returns `std::nullopt`.
std::optional<int64_t> getFoldedConstantValue(Operation *op);

// Return true if the `val` value is a constant containing a value equal to
// expected.
bool isConstant(Value val, int64_t expected);

Value getFinalValue(Value value);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_UTILS_UTILITY_H
