// Core's triton-licm
// (lib/Dialect/Triton/Transforms/LoopInvariantCodeMotion.cpp) with one change:
// an op that carries `tt.no_licm` stays in its loop.
// TritonIntelGPUHoistLayoutConversions stamps that attribute on the conversions
// it decides to leave in place; core's pass does not know it. Keep the rest
// identical to core's; copied-from-sha256 is the hash of the core file it was
// copied from.
// clang-format off
// copied-from-sha256: 688fa8418ad8393d4e63835751e60db5adadf243989b8a63598bfce25a14c51b
// clang-format on
#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton::intel {

#define GEN_PASS_DEF_TRITONINTELLOOPINVARIANTCODEMOTION
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-intel-licm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class LoopInvariantCodeMotionPass
    : public impl::TritonIntelLoopInvariantCodeMotionBase<
          LoopInvariantCodeMotionPass> {

  DenseMap<LoopLikeOpInterface, bool> isLoopMemoryEffectFreeOrOnlyRead;

  bool isLICMDisabled(LoopLikeOpInterface loopLike) {
    auto loopMD = loopLike->getAttrOfType<LLVM::LoopAnnotationAttr>(
        LLVM::LoopAnnotationAttr::name);
    if (!loopMD)
      return false;

    auto licmMD = loopMD.getLicm();
    if (!licmMD)
      return false;

    auto disable = licmMD.getDisable();
    return disable && disable.getValue();
  }

  bool isMemoryEffectFreeOrOnlyRead(Operation *op) {
    std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
        getEffectsRecursively(op);
    if (!effects)
      return false;
    return llvm::all_of(*effects,
                        [&](const MemoryEffects::EffectInstance &effect) {
                          return isa<MemoryEffects::Read>(effect.getEffect());
                        });
  }

  void runOnOperation() override {
    // Walk through all loops in a function in innermost-loop-first order.
    // This way, we first LICM from the inner loop, and place the ops in the
    // outer loop, which in turn can be further LICM'ed.
    getOperation()->walk([&](LoopLikeOpInterface loopLike) {
      if (isLICMDisabled(loopLike))
        return;
      moveLoopInvariantCode(
          loopLike.getLoopRegions(),
          // isDefinedOutsideOfRegion
          [&](Value value, Region *region) {
            return loopLike.isDefinedOutsideOfLoop(value);
          },
          // shouldMoveOutOfRegion
          [&](Operation *op, Region *region) {
            if (op->hasAttr("tt.no_licm"))
              return false;
            if (!isa<LoadOp>(op))
              return isSpeculatable(op) && isMemoryEffectFree(op);
            if (!isLoopMemoryEffectFreeOrOnlyRead.contains(loopLike))
              isLoopMemoryEffectFreeOrOnlyRead[loopLike] =
                  isMemoryEffectFreeOrOnlyRead(loopLike);
            return isMemoryEffectFreeOrOnlyRead(op) &&
                   isLoopMemoryEffectFreeOrOnlyRead[loopLike];
          },
          // moveOutOfRegion
          [&](Operation *op, Region *) {
            // Create the new mask for load op.
            if (auto loadOp = dyn_cast<LoadOp>(op)) {
              IRRewriter rewriter(loopLike);
              Location loc = loopLike->getLoc();
              Value cond;
              if (auto forOp = dyn_cast<scf::ForOp>(loopLike.getOperation())) {
                cond = arith::CmpIOp::create(
                    rewriter, loc, arith::CmpIPredicate::slt,
                    forOp.getLowerBound(), forOp.getUpperBound());
              } else if (auto whileOp =
                             dyn_cast<scf::WhileOp>(loopLike.getOperation())) {
                // TODO: Support Load Op hoisting for while loop.
                return;
              } else {
                return;
              }
              Value newMask = getPredMask(rewriter, loadOp.getPtr().getType(),
                                          loadOp.getMask(), cond);
              loadOp.getMaskMutable().assign(newMask);
            }
            loopLike.moveOutOfLoop(op);
          });
    });
  }
};

} // namespace mlir::triton::intel
