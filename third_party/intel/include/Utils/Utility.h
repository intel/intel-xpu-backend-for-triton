#ifndef TRITON_INTEL_UTILS_UTILITY_H
#define TRITON_INTEL_UTILS_UTILITY_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

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

// Look through value-preserving operations to the value that ultimately defines
// `value`, e.g. through casts, broadcasts and adding zero.
//
// An scf.for iteration argument resolves to its INIT operand: the yielded
// update is never inspected. That is intended for callers asking "what is this
// value outside the loop", but it makes the result UNSOUND for an alignment or
// divisibility proof, because `off = 0; off += 3` resolves to the constant 0.
// Issue #7990 was exactly that. For such a proof, query the raw value through
// ModuleAxisInfoAnalysis, which reaches a fixpoint over the loop, and conjoin
// this only as a conservative second opinion.
Value getFinalValue(Value value);

// Erase the operations in \p operations.
void eraseOperations(SmallPtrSetImpl<Operation *> &operations);

// The set of MakeTensorDescOps that may define a descriptor value.
//
// With more than one candidate the per-descriptor properties can disagree, so a
// property is only reachable through a helper that says what it does about the
// disagreement: consistentX() requires every candidate to match, allSatisfy()
// requires every candidate to pass a predicate. Both spellings are needed --
// padding and shape must be identical, whereas base and pitch legitimately
// differ as long as each is aligned.
class DescriptorDefinitions {
  SmallVector<triton::MakeTensorDescOp> ops;

public:
  explicit DescriptorDefinitions(SmallVector<triton::MakeTensorDescOp> ops)
      : ops(std::move(ops)) {}

  // Iteration but deliberately no operator[]: every raw use in tree is a
  // range-for or a range algorithm, so withholding indexing costs nothing and
  // makes `defs[0].getPadding()` a compile error instead of a review question.
  auto begin() const { return ops.begin(); }
  auto end() const { return ops.end(); }

  bool empty() const { return ops.empty(); }
  size_t size() const { return ops.size(); }

  // The padding shared by every candidate; nullopt if they disagree or the set
  // is empty.
  std::optional<triton::PaddingOption> consistentPadding() const;

  // The shape operands shared by every candidate; nullopt if they disagree or
  // the set is empty. Compares operand ranges by SSA identity, NOT by folded
  // value: descriptors built from separate `arith.constant 64` ops disagree.
  // That errs toward refusing the 2D block path, so it is safe -- but do not
  // read the name as semantic equality.
  std::optional<Operation::operand_range> consistentShape() const;

  // True iff `pred` holds for every candidate. False for an empty set: empty
  // means the trace failed, which is never a licence to proceed.
  bool
  allSatisfy(llvm::function_ref<bool(triton::MakeTensorDescOp)> pred) const;
};

// Find every MakeTensorDescOp reachable from `val`, tracing block arguments,
// loop yields, if/select branches and unrealized casts. Empty if any path
// reaches an untraceable value: a function entry-block argument, a tt.call
// result, a loop induction variable, or an unknown parent op. ub.poison is the
// one exception -- it is skipped rather than treated as untraceable, so a
// descriptor reached alongside a poison path still yields a non-empty set.
DescriptorDefinitions findDescriptorDefinitions(Value val);

// Find the unique MakeTensorDescOp for the given value.
// Returns the op only if all reachable paths lead to the same one.
std::optional<triton::MakeTensorDescOp> findMakeTensorDescOp(Value val);

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_UTILS_UTILITY_H
