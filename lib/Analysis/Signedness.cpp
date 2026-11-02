#include "triton/Analysis/Signedness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>

namespace mlir::triton {

namespace {

/// Chains deeper than this are declined rather than walked. A backstop against
/// pathological IR: the index arithmetic this fires on is a handful of
/// operations deep.
constexpr unsigned kMaxDepth = 12;

bool isNonNegative(Value v, Operation *useOp, unsigned depth);

/// True if `assume` is certain to have executed by the time `useOp` does.
///
/// Asks whether the assume precedes `useOp` in the assume's own block, taking
/// an enclosing operation when `useOp` is nested, so an assume ahead of a loop
/// covers its body. Missing what only a dominance tree establishes - an assume
/// in a predecessor block - costs precision, never soundness.
bool executesBefore(Operation *assume, Operation *useOp) {
  Block *assumeBlock = assume->getBlock();
  if (!assumeBlock)
    return false;
  Operation *ancestor = assumeBlock->findAncestorOpInBlock(*useOp);
  return ancestor && assume->isBeforeInBlock(ancestor);
}

/// True if `cmp` states that `v` is at or above some non-negative bound. Both
/// operand orders are recognised: `v >= c` and `c <= v`.
bool comparisonImpliesNonNegative(arith::CmpIOp cmp, Value v) {
  auto atLeast = [](Value bound, bool strict) {
    APInt c;
    if (!matchPattern(bound, m_ConstantInt(&c)))
      return false;
    // `v > c` needs c >= -1 to imply `v >= 0`; `v >= c` needs c >= 0.
    return strict ? c.sge(-1) : c.isNonNegative();
  };

  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::sge:
    return cmp.getLhs() == v && atLeast(cmp.getRhs(), /*strict=*/false);
  case arith::CmpIPredicate::sgt:
    return cmp.getLhs() == v && atLeast(cmp.getRhs(), /*strict=*/true);
  case arith::CmpIPredicate::sle:
    return cmp.getRhs() == v && atLeast(cmp.getLhs(), /*strict=*/false);
  case arith::CmpIPredicate::slt:
    return cmp.getRhs() == v && atLeast(cmp.getLhs(), /*strict=*/true);
  default:
    return false;
  }
}

/// Looks for an `llvm.intr.assume` of a comparison that bounds `v` below by a
/// non-negative constant. `tl.assume(x >= 0)` lowers to exactly that.
bool assumedNonNegative(Value v, Operation *useOp) {
  for (Operation *user : v.getUsers()) {
    auto cmp = dyn_cast<arith::CmpIOp>(user);
    if (!cmp || !comparisonImpliesNonNegative(cmp, v))
      continue;
    for (Operation *condUser : cmp.getResult().getUsers())
      if (condUser->getName().getStringRef() == "llvm.intr.assume" &&
          executesBefore(condUser, useOp))
        return true;
  }
  return false;
}

bool allElementsNonNegative(Attribute value) {
  if (auto intAttr = dyn_cast<IntegerAttr>(value))
    return intAttr.getValue().isNonNegative();
  auto denseAttr = dyn_cast<DenseElementsAttr>(value);
  if (!denseAttr || !denseAttr.getElementType().isSignlessInteger())
    return false;
  return llvm::all_of(denseAttr.getValues<APInt>(),
                      [](const APInt &v) { return v.isNonNegative(); });
}

bool isNonNegative(Value v, Operation *useOp, unsigned depth) {
  if (depth >= kMaxDepth)
    return false;
  if (assumedNonNegative(v, useOp))
    return true;

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return false;

  auto operandIsNonNegative = [&](Value operand) {
    return isNonNegative(operand, useOp, depth + 1);
  };

  return llvm::TypeSwitch<Operation *, bool>(defOp)
      // A program id is in [0, num_programs).
      .Case<GetProgramIdOp, GetNumProgramsOp>([](auto) { return true; })
      .Case<MakeRangeOp>([](MakeRangeOp op) {
        // getStart() is unsigned; the attribute carries the signed value.
        return op.getStartAttr().getInt() >= 0;
      })
      .Case<arith::ConstantOp>([](arith::ConstantOp op) {
        return allElementsNonNegative(op.getValue());
      })
      // Composition. Addition and multiplication preserve a non-negative sign
      // only in the absence of signed overflow, which is assumed; see the
      // header.
      .Case<arith::AddIOp, arith::MulIOp, arith::MinSIOp>([&](auto op) {
        return operandIsNonNegative(op.getLhs()) &&
               operandIsNonNegative(op.getRhs());
      })
      // max(a, b) >= a, so one non-negative operand is enough.
      .Case<arith::MaxSIOp>([&](arith::MaxSIOp op) {
        return operandIsNonNegative(op.getLhs()) ||
               operandIsNonNegative(op.getRhs());
      })
      // Either arm can be the one that runs.
      .Case<arith::SelectOp>([&](arith::SelectOp op) {
        return operandIsNonNegative(op.getTrueValue()) &&
               operandIsNonNegative(op.getFalseValue());
      })
      // A signed remainder takes the sign of its dividend, so the divisor's
      // sign is irrelevant: srem(7, -3) is 1.
      .Case<arith::RemSIOp>(
          [&](arith::RemSIOp op) { return operandIsNonNegative(op.getLhs()); })
      // A quotient's sign is the product of the operands': sdiv(7, -3) is -2.
      .Case<arith::DivSIOp>([&](arith::DivSIOp op) {
        return operandIsNonNegative(op.getLhs()) &&
               operandIsNonNegative(op.getRhs());
      })
      // Unsigned results and zero-extension have the top bit clear.
      .Case<arith::RemUIOp, arith::DivUIOp, arith::ExtUIOp>(
          [](auto) { return true; })
      // Sign extension preserves the value; an arithmetic shift right
      // preserves the sign bit.
      .Case<arith::ExtSIOp>(
          [&](arith::ExtSIOp op) { return operandIsNonNegative(op.getIn()); })
      .Case<arith::ShRSIOp>(
          [&](arith::ShRSIOp op) { return operandIsNonNegative(op.getLhs()); })
      // Shape operations replicate their source element-wise.
      .Case<SplatOp, BroadcastOp, ExpandDimsOp>(
          [&](auto op) { return operandIsNonNegative(op.getSrc()); })
      // Everything else is declined. Notably arith.subi, where `a >= 0` and
      // `b >= 0` say nothing about `a - b`, and arith.trunci, which can turn a
      // positive value negative by dropping the high bits.
      .Default([](Operation *) { return false; });
}

} // namespace

bool dividendIsNonNegative(Operation *op) {
  if (isa<arith::DivUIOp, arith::RemUIOp>(op))
    return true;
  assert((isa<arith::DivSIOp, arith::RemSIOp>(op)) &&
         "expected a division or remainder");
  return isNonNegative(op->getOperand(0), op, /*depth=*/0);
}

} // namespace mlir::triton
