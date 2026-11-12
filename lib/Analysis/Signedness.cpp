#include "triton/Analysis/Signedness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>

namespace mlir::triton {

namespace {

/// Chains deeper than this are declined rather than walked. A backstop against
/// pathological IR: the index arithmetic this fires on is a handful of
/// operations deep.
constexpr unsigned kMaxDepth = 12;

/// Block arguments taken as non-negative; see `blockArgIsNonNegative`. Passed
/// by value, so a hypothesis reaches only the sub-walk that introduced it:
/// shared, one branch of a `max` or `select` could draw on a hypothesis a
/// failed sibling left behind undischarged.
using Hypotheses = SmallPtrSet<Value, 4>;

bool isNonNegative(Value v, Operation *useOp, unsigned depth,
                   Hypotheses assumed);

/// True if `assume` is certain to have executed by the time `useOp` does.
///
/// Either the assume precedes `useOp` in its own block, taking an enclosing
/// operation when `useOp` is nested, so an assume ahead of a loop covers its
/// body; or it sits in the entry block of a region `useOp` is inside, which
/// runs before every other block of that region. The second is the form
/// `convert-scf-to-cf` leaves, and so what the LLVM lowering's AxisInfo sees.
/// Missing what only a dominance tree establishes - an assume in some other
/// predecessor block - costs precision, never soundness.
bool executesBefore(Operation *assume, Operation *useOp) {
  Block *assumeBlock = assume->getBlock();
  if (!assumeBlock)
    return false;
  if (Operation *ancestor = assumeBlock->findAncestorOpInBlock(*useOp))
    return assume->isBeforeInBlock(ancestor);

  Region *region = assumeBlock->getParent();
  // A graph region has no execution order between its blocks to appeal to.
  if (!region || !assumeBlock->isEntryBlock() || mayBeGraphRegion(*region))
    return false;
  return region->findAncestorOpInRegion(*useOp);
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
///
/// Matched by name rather than with `isa<LLVM::AssumeOp>`: this library does
/// not link the LLVM dialect.
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

/// True if every element of `value` satisfies `pred`.
bool allElements(Attribute value, function_ref<bool(const APInt &)> pred) {
  if (auto intAttr = dyn_cast<IntegerAttr>(value))
    return pred(intAttr.getValue());
  auto denseAttr = dyn_cast<DenseElementsAttr>(value);
  if (!denseAttr || !denseAttr.getElementType().isSignlessInteger())
    return false;
  return llvm::all_of(denseAttr.getValues<APInt>(), pred);
}

bool allElementsNonNegative(Attribute value) {
  return allElements(value, [](const APInt &v) { return v.isNonNegative(); });
}

/// True if `v` is a constant whose every element is 2 or more *read as
/// unsigned*. Dividing unsigned by such a divisor brings any dividend below
/// 2^(w-1), so the quotient's top bit is clear whatever the dividend's was.
bool isUnsignedDivisorAboveOne(Value v) {
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    if (auto splat = v.getDefiningOp<SplatOp>())
      return isUnsignedDivisorAboveOne(splat.getSrc());
    return false;
  }
  return allElements(constOp.getValue(),
                     [](const APInt &c) { return c.ugt(1); });
}

/// True if every value reaching `arg` along a CFG edge is non-negative, taking
/// `arg` itself as non-negative while proving them. This is the `scf.for`
/// counter once `convert-scf-to-cf` has turned it into a header block argument,
/// the form the LLVM lowering's AxisInfo sees.
///
/// Assuming `arg` to prove `arg` is induction over block entries, not
/// circularity: the value arriving on the n-th entry can only use the `arg` of
/// the (n-1)-th, and the first entry has to arrive along an edge whose value
/// does not use `arg` at all, since `arg` does not dominate that predecessor.
bool blockArgIsNonNegative(BlockArgument arg, unsigned depth,
                           Hypotheses assumed) {
  Block *block = arg.getOwner();
  // A function or region entry argument. Declining is what stops the hypothesis
  // below from standing in for every incoming value, there being none.
  if (block->hasNoPredecessors())
    return false;

  assumed.insert(arg);

  for (BlockOperand &edge : block->getUses()) {
    auto branch = dyn_cast<BranchOpInterface>(edge.getOwner());
    if (!branch)
      return false;
    SuccessorOperands operands =
        branch.getSuccessorOperands(edge.getOperandNumber());
    // Null for an operand the successor produces rather than receives.
    Value incoming = arg.getArgNumber() < operands.size()
                         ? operands[arg.getArgNumber()]
                         : Value();
    if (!incoming || !isNonNegative(incoming, branch, depth + 1, assumed))
      return false;
  }
  return true;
}

bool isNonNegative(Value v, Operation *useOp, unsigned depth,
                   Hypotheses assumed) {
  if (depth >= kMaxDepth)
    return false;
  if (assumed.contains(v))
    return true;
  if (assumedNonNegative(v, useOp))
    return true;

  Operation *defOp = v.getDefiningOp();
  if (!defOp) {
    // An `scf.for` induction variable takes the values `lb + k * step`, so
    // non-negative bounds make every one of them non-negative. Both bounds are
    // read at the loop, not at `useOp`, which is inside it.
    auto blockArg = cast<BlockArgument>(v);
    auto forOp =
        dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (forOp && blockArg == forOp.getInductionVar())
      return isNonNegative(forOp.getLowerBound(), forOp, depth + 1, assumed) &&
             isNonNegative(forOp.getStep(), forOp, depth + 1, assumed);
    return blockArgIsNonNegative(blockArg, depth, assumed);
  }

  auto operandIsNonNegative = [&](Value operand) {
    return isNonNegative(operand, useOp, depth + 1, assumed);
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
      // An unsigned remainder is at or below its dividend (as unsigned) and
      // below its divisor, so either operand having its top bit clear clears
      // the result's. remui(0xffffffc0, 0xfffffff0) is 0xffffffc0, i.e. -64.
      .Case<arith::RemUIOp>([&](arith::RemUIOp op) {
        return operandIsNonNegative(op.getLhs()) ||
               operandIsNonNegative(op.getRhs());
      })
      // An unsigned quotient is at or below its dividend, and a divisor of 2 or
      // more clears the top bit on its own. A divisor of 1 is the identity and
      // carries a negative dividend straight through.
      .Case<arith::DivUIOp>([&](arith::DivUIOp op) {
        return operandIsNonNegative(op.getLhs()) ||
               isUnsignedDivisorAboveOne(op.getRhs());
      })
      // Zero extension always widens, so the result's top bit is clear.
      .Case<arith::ExtUIOp>([](auto) { return true; })
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

bool valueIsNonNegative(Value v, Operation *useOp) {
  return isNonNegative(v, useOp, /*depth=*/0, /*assumed=*/{});
}

bool dividendIsNonNegative(Operation *op) {
  if (isa<arith::DivUIOp, arith::RemUIOp>(op))
    return true;
  assert((isa<arith::DivSIOp, arith::RemSIOp>(op)) &&
         "expected a division or remainder");
  return isNonNegative(op->getOperand(0), op, /*depth=*/0, /*assumed=*/{});
}

} // namespace mlir::triton
