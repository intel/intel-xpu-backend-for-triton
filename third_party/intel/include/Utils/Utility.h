#ifndef TRITON_INTEL_UTILS_UTILITY_H
#define TRITON_INTEL_UTILS_UTILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class FunctionOpInterface;
class LoopLikeOpInterface;
} // namespace mlir

namespace mlir::triton::intel {

Value findOrCreateCastOp(Value val, Type targetType);

// Lookup for a integer constant with the given value and bitwidth in the
// current block (before the builder insertion point). Return it if found,
// otherwise create a new one.
Value findOrCreateIntConstant(Location loc, int val, unsigned bitWidth,
                              OpBuilder &builder);

// This function folds the `v` value and returns the constant value if it
// has successfully folded to a constant. Otherwise, it returns `std::nullopt`.
std::optional<int64_t> getFoldedConstantValue(Value v, int depth = 8);

// Return true if the `val` value is a constant containing a value equal to
// expected.
bool isConstant(Value val, int64_t expected);

Value getFinalValue(Value value);

// Erase the operations in \p operations.
void eraseOperations(SmallPtrSetImpl<Operation *> &operations);

// Find all MakeTensorDescOps reachable from the given value.
// Traverses block arguments, loop yields, if/select branches, etc.
// Returns a deduplicated list of all reachable ops, or an empty list if any
// path leads to an untraceable value (e.g., function call, unknown op).
SmallVector<triton::MakeTensorDescOp> findAllMakeTensorDescOps(Value val);

// Find the unique MakeTensorDescOp for the given value.
// Returns the op only if all reachable paths lead to the same one.
std::optional<triton::MakeTensorDescOp> findMakeTensorDescOp(Value val);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_UTILS_UTILITY_H
