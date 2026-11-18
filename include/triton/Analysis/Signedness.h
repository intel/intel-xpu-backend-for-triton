#ifndef TRITON_ANALYSIS_SIGNEDNESS_H
#define TRITON_ANALYSIS_SIGNEDNESS_H

#include "mlir/IR/Operation.h"

namespace mlir::triton {

/// Returns true if `op`, a signed or unsigned division or remainder, has a
/// dividend that is non-negative wherever `op` executes. A tensor dividend
/// qualifies only when every element does; an unsigned operation qualifies
/// unconditionally, since its dividend is read as unsigned.
///
/// "Could not tell" and "is negative" are both false, since a caller guarding
/// a deduction has to decline it either way.
///
/// Deliberately not a range query. The dividend of a swizzled index is of the
/// form `pid * BLOCK + arange`, whose range a sound analysis has to widen to
/// the whole signed domain, because a signed multiply can overflow. This
/// instead composes a sign over the index arithmetic *assuming no signed
/// overflow* - the same assumption the deduction it guards already makes, and
/// one that holds for any in-range index computation.
///
/// Facts come from operations whose sign is structural (`tt.get_program_id`,
/// `tt.make_range`, constants, a loop counter over non-negative bounds - as an
/// `scf.for` induction variable or as the block argument it lowers to) and from
/// `llvm.intr.assume`, which is how `tl.assume` reaches the compiler.
/// An assume counts only where it is certain to have executed first; see the
/// implementation for the (conservative) test.
///
/// This belongs here rather than in a backend: `tritongpu-coalesce` builds
/// `triton::ModuleAxisInfoAnalysis` directly, and that analysis' deductions
/// decide the `sizePerThread` a load is vectorized with, so a backend-side
/// implementation would never be consulted where it matters most. Needing
/// neither a solver nor dominance info also keeps it usable from an AxisInfo
/// running over partially converted IR, as the LLVM lowering's does.
bool dividendIsNonNegative(Operation *op);

/// Returns true if `v` is non-negative wherever `useOp` executes, by the same
/// structural composition (and the same no-signed-overflow assumption) as
/// `dividendIsNonNegative`. "Could not tell" is false.
bool valueIsNonNegative(Value v, Operation *useOp);

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_SIGNEDNESS_H
