#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUREDUCEDATADUPLICATION
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

/// Return true if \p op, or an op nested in it, may write memory, so a
/// shared-memory store must not be moved above it. This mirrors
/// `hasWriteSideEffect` in TritonGPUReorderInstructions, except that:
///  - writes to the `L2Cache` resource are ignored: `ttig.prefetch` and
///    `ttig.descriptor_prefetch` declare `MemWrite<L2Cache>` only to keep
///    CSE/DCE from removing them (see CodeSinking.cpp). The resource IDs are
///    compared directly because `isa<>` on resources is tree-based;
///  - unknown effects count as a write, so ops without a memory-effect
///    interface (e.g. `ttg.barrier`, `ttg.async_wait`) are not crossed.
bool mayWriteMemory(Operation *op) {
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  if (!effects)
    return true;
  return llvm::any_of(*effects, [](const MemoryEffects::EffectInstance &e) {
    if (isa<MemoryEffects::Read, MemoryEffects::Allocate, MemoryEffects::Free>(
            e.getEffect()))
      return false;
    return e.getResource()->getResourceID() !=
           triton::gpu::intel::L2Cache::getResourceID();
  });
}

/// Return true if an op that may write memory executes between \p ip and
/// \p end, where \p ip dominates \p end. Ops are scanned from \p ip up to and
/// including the ancestor of \p end in the block of \p ip; that ancestor
/// (e.g. the loop containing \p end) is checked as a whole, since all of its
/// body can run between \p ip and \p end on some iteration.
bool crossesMemoryWrite(OpBuilder::InsertPoint ip, Operation *end) {
  Block *block = ip.getBlock();
  Operation *ancestor = block->findAncestorOpInBlock(*end);
  if (!ancestor)
    return true;
  for (Operation &op : llvm::make_range(ip.getPoint(), block->end())) {
    if (&op == end)
      return false;
    if (mayWriteMemory(&op))
      return true;
    if (&op == ancestor)
      return false;
  }
  return true;
}

/// Return where the shared-memory staging buffer that replaces \p cvtOp
/// should be allocated, or an unset insertion point to allocate it at
/// \p cvtOp.
///
/// A conversion can sit in a loop while its source is defined outside it (an
/// earlier pass chose to keep the conversion in the loop, e.g. to limit the
/// dot-operand's register live range; only the local_load needs to stay
/// there). When that happens, the store into shared memory is loop
/// invariant. Allocating right after the source performs the store once and
/// ends the source's register live range early, without relying on
/// TritonGPUReorderInstructions to hoist the allocation later.
///
/// A block-argument source (e.g. a function argument, or an outer loop's
/// iter_arg) has no defining op to allocate after; the allocation is placed
/// right before the op of the argument's block that contains \p cvtOp, which
/// is outside the loop and only moves the store above that op.
OpBuilder::InsertPoint getLoopInvariantAllocPoint(ConvertLayoutOp cvtOp) {
  auto loop = cvtOp->getParentOfType<LoopLikeOpInterface>();
  if (!loop)
    return {};
  Value src = cvtOp.getSrc();
  // Rejects values defined in the loop, including its own region arguments.
  if (!loop.isDefinedOutsideOfLoop(src))
    return {};
  OpBuilder::InsertPoint ip;
  if (Operation *srcDef = src.getDefiningOp()) {
    // Staging a scalar-derived value for the whole loop costs shared memory
    // for no benefit; TritonGPUReorderInstructions skips these for the same
    // reason.
    if (isa<arith::ConstantOp, triton::SplatOp>(srcDef))
      return {};
    ip = OpBuilder::InsertPoint(srcDef->getBlock(),
                                std::next(srcDef->getIterator()));
  } else {
    Block *owner = cast<BlockArgument>(src).getOwner();
    Operation *ancestor = owner->findAncestorOpInBlock(*cvtOp);
    if (!ancestor)
      return {};
    ip = OpBuilder::InsertPoint(owner, ancestor->getIterator());
  }
  // Keep the store after ops that may write memory, e.g. waits that complete
  // earlier asynchronous reads of shared memory, as
  // TritonGPUReorderInstructions does when hoisting allocations.
  if (crossesMemoryWrite(ip, cvtOp))
    return {};
  return ip;
}

class TritonIntelGPUReduceDataDuplicationPass
    : public intel::impl::TritonIntelGPUReduceDataDuplicationBase<
          TritonIntelGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
      if (!cvtNeedsSharedMemory(cvtOp))
        return;
      auto srcOrder = triton::gpu::getOrder(srcType);
      auto rank = srcOrder.size(); // TODO: maybe we can use upstream code.
      if (auto srcDpasEncoding =
              dyn_cast<triton::gpu::intel::DpasEncodingAttr>(srcEncoding)) {
        auto opIdx =
            static_cast<intel::DpasEncodingAttr::OpIdx>(dstDotOp.getOpIdx());
        if ((opIdx == intel::DpasEncodingAttr::OpIdx::OperandA /* Operand A */
             && dstDotOp.getParent() == srcDpasEncoding &&
             srcDpasEncoding.getWarpsPerCTA()[rank - 1] ==
                 1 /* No parallel on N dim */) ||
            (opIdx == intel::DpasEncodingAttr::OpIdx::OperandB /* Operand B */
             && dstDotOp.getParent() == srcDpasEncoding &&
             srcDpasEncoding.getWarpsPerCTA()[rank - 2] ==
                 1 /* No parallel on M dim */))
          /* The destination dot layout has no duplication. */
          return;
      }
      SmallVector<unsigned> sharedOrder;
      if (rank == 3) {
        // add all elements except the element that is zero
        for (unsigned i = 0; i < rank; ++i)
          if (srcOrder[i] != 0)
            sharedOrder.emplace_back(srcOrder[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = std::move(srcOrder);
      }
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto tmpType = triton::gpu::MemDescType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SwizzledSharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(), sharedOrder,
              triton::gpu::getCGALayout(srcEncoding), srcType.getElementType()),
          sharedMemorySpace);
      if (OpBuilder::InsertPoint allocPoint = getLoopInvariantAllocPoint(cvtOp);
          allocPoint.isSet())
        builder.restoreInsertionPoint(allocPoint);
      auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(),
                                                   tmpType, cvtOp.getSrc());
      builder.setInsertionPoint(cvtOp);
      auto newConvert = triton::gpu::LocalLoadOp::create(
          builder, cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

} // namespace
