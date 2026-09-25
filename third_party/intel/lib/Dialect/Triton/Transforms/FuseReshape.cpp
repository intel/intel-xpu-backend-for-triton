#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/DefUseChain.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

#define DEBUG_TYPE "triton-intel-fuse-reshape"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELFUSERESHAPE
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

// Transform:
//   %desc = tt.make_tensor_descriptor %base, [%s0,%s1,%s2], [%a,%b,%c]
//                       : !tt.tensordesc<1x512x64xf16>
//   %load = tt.descriptor_load %desc[%x,%y,%z] -> tensor<1x512x64xf16>
//   %A = tt.reshape %load : tensor<1x512x64xf16> -> tensor<512x64xf16>
//   dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
// into:
//   %d = %a / %b
//   %i = max(min(%x, max(%s0,1)-1), 0)         // this load's collapsed index
//   %n = (%s0 > 0)                             // collapsed dimension not empty
//   %e = (%i*%d + %s1) * %n + %f * (1 - %n)    // merged extent, %f when empty
//   %g = (%x >= 0) & (%x < %s0)                // collapsed index in range
//   %j = (%x*%d + %y) * %g + (1 - %g) * %e     // merged index, %e when padding
//   %desc = tt.make_tensor_descriptor %base, [%e,%s2], [%b,%c]
//                       : !tt.tensordesc<512x64xf16>
//   %A = tt.descriptor_load %desc[%j,%z] -> tensor<512x64xf16>
//   dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
// A unit-extent middle dimension (e.g. a one-head tile of a contiguous
// (TOKENS, HEADS, HEAD_DIM) tensor) is collapsed the same way, with %d = %b/%c,
// %g guarding %y against %s1 and %j merging %y and %z:
//   %desc = tt.make_tensor_descriptor %base, [%s0,%e], [%a,%c]
//   %A = tt.descriptor_load %desc[%x,%j] -> tensor<64x128xf16>
// The merged extent is per *load*, not per dimension: bounding it with the
// whole collapsed dimension ((%s0-1)*%d+%s1) is tight only for the last index
// of that dimension and lets every other index read rows the rank-3 form pads
// (issue #8001). `%f` is the smallest legal surface extent, declared when the
// collapsed dimension is empty at runtime: a zero extent is emitted as `0 - 1`
// and reads back as a 16MB surface, so the empty case declares a legal one and
// relies on `%g` forcing the load out of range to make it pad.
class FuseReshapeWithLoad : public tt::intel::Fuser {
public:
  void run(ModuleOp moduleOp) {
    // Collect def-use chains originating at a `MakeTensorDescOp` operation
    // and terminating at a candidate `tt::ReshapeOp` operation.
    // Note: A candidate `reshapeOp` must use the result of a `loadOp` using a
    // descriptor created by the `MakeTensorDescOp` rooting the def-use chain.
    DefUseChainManager manager;
    moduleOp.walk([&](tt::ReshapeOp reshapeOp) {
      if (isCandidate(reshapeOp)) {
        Operation *srcOp = reshapeOp.getSrc().getDefiningOp();
        assert(srcOp && "Expected a valid source operation");

        llvm::TypeSwitch<Operation *>(srcOp)
            .Case<tt::DescriptorLoadOp>([&](auto descLoadOp) {
              auto makeTensorDescOp =
                  *tt::intel::findMakeTensorDescOp(descLoadOp.getDesc());
              manager.createChains(makeTensorDescOp, reshapeOp);
            })
            .Default([](Operation *) {});
      }
    });

    if (manager.getChains().empty())
      return;

    LLVM_DEBUG(llvm::dbgs() << "[Initial set of chains]:\n" << manager << "\n");

    // Prune chains that overlap with other chains (except at the root).
    unsigned numChainsCollected = manager.getChains().size();
    bool includeStart = false;
    manager.pruneOverlappingChains(includeStart);
    if (manager.getChains().empty())
      return;

    LLVM_DEBUG({
      if (manager.getChains().size() != numChainsCollected)
        llvm::dbgs() << "[After pruning]:\n" << manager << "\n";
    });

    // Prune chains that cannot be fused.
    pruneInvalid(manager.getChainsMutable());
    if (manager.getChains().empty())
      return;

    LLVM_DEBUG(llvm::dbgs() << "[Before fusion]:\n" << manager << "\n");

    // Fuse tt.LoadOp->tt.ReshapeOp operations.
    Fuser::fuse(manager.getChains());

    // Remove operations that are no longer used.
    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);
  }

private:
  /// Return the unit-extent dimension collapsed into the one following it:
  /// 1xNxM or Nx1xM -> NxM. Dimension 0 wins, so `1x1xM` keeps its behavior.
  static std::optional<unsigned> getCollapsedDim(ArrayRef<int64_t> shape) {
    if (shape.size() != 3)
      return std::nullopt;
    if (shape[0] == 1)
      return 0;
    if (shape[1] == 1)
      return 1;
    return std::nullopt;
  }

  /// Return \p values without the element at index \p dim.
  template <typename RangeT>
  static auto dropDim(RangeT &&values, unsigned dim) {
    auto res = llvm::to_vector(values);
    res.erase(res.begin() + dim);
    return res;
  }

  void fuse(const DefUseChain &chain) final {
    assert(isa<tt::ReshapeOp>(chain.getEnd()) &&
           "Expecting 'chain' to be terminated by a 'tt.reshape' operation");

    llvm::TypeSwitch<Operation *>(chain.getStart())
        .Case<tt::MakeTensorDescOp>([&](auto makeTensorDescOp) {
          fuseMakeTensorDescOp(chain, makeTensorDescOp);
        })
        .Default([](Operation *) {
          llvm_unreachable("Unexpected 'chain' root operation kind");
        });
  }

  void fuseMakeTensorDescOp(const DefUseChain &chain,
                            tt::MakeTensorDescOp makeTensorDescOp) {
    assert(chain.getStart() == makeTensorDescOp &&
           "Unexpected 'chain' start operation");
    assert(isa<tt::ReshapeOp>(chain.getEnd()) &&
           "Expecting 'chain' to be terminated by a 'tt.reshape' operation");
    assert(chain.getOps().size() == 3 &&
           "Expecting 'chain' to have exactly 3 operations");

    auto reshapeOp = cast<tt::ReshapeOp>(chain.getEnd());
    auto descLoadOp =
        cast<tt::DescriptorLoadOp>(reshapeOp.getSrc().getDefiningOp());
    LLVM_DEBUG(llvm::dbgs() << "Fusing:\n  " << reshapeOp << "\nwith:\n  "
                            << descLoadOp << "\n");

    // Create a MakeTensorDescOp yielding a 2-dim tensor descriptor.
    std::optional<unsigned> dim =
        getCollapsedDim(makeTensorDescOp.getType().getBlockType().getShape());
    assert(dim && "Result shape should have extent equal to 1 in either the "
                  "outermost or the middle dimension");
    const unsigned collapsedDim = *dim;
    const unsigned mergedDim = collapsedDim + 1;

    auto tensorType = cast<RankedTensorType>(reshapeOp.getType());
    auto newDescType = tt::TensorDescType::get(
        tensorType.getShape(), tensorType.getElementType(), mlir::Attribute{});

    Location loc = makeTensorDescOp.getLoc();
    OperandRange shapes = makeTensorDescOp.getShape();
    OperandRange strides = makeTensorDescOp.getStrides();
    OperandRange offsets = descLoadOp.getIndices();
    Value collapsedShape = shapes[collapsedDim];
    Value collapsedOffset = offsets[collapsedDim];
    Type indexTy = collapsedShape.getType();

    // The merged extent depends on this load's collapsed index, so the new
    // descriptor can only stay where the original one was if that index is
    // available there; otherwise it is rebuilt at the load. Reconstructing the
    // descriptor per load is cheap: `LowerTo2DBlockLoad` re-extracts every
    // shape/stride field per load anyway.
    Operation *insertionPoint =
        domInfo.properlyDominates(collapsedOffset, makeTensorDescOp)
            ? static_cast<Operation *>(makeTensorDescOp)
            : static_cast<Operation *>(descLoadOp);
    OpBuilder builder(insertionPoint);

    // An index pair (i,j) in `collapsedDim`/`mergedDim` addresses the same
    // element as the single index i * d + j, with d = the stride ratio. Erasing
    // `collapsedDim` leaves the merged entry at index `collapsedDim`, so shape
    // [s0,s1,s2] / stride [a,b,c] yields [e, s2] / [b,c] or [s0, e] / [a,c],
    // with `e` the merged extent computed below.
    // Signed, to agree with the divisibility proof in `isCandidate`.
    Value ratio = builder.createOrFold<arith::TruncIOp>(
        loc, indexTy,
        builder.createOrFold<arith::DivSIOp>(loc, strides[collapsedDim],
                                             strides[mergedDim]));
    auto merge = [&](Value hi, Value lo) -> Value {
      return builder.createOrFold<arith::AddIOp>(
          loc, builder.createOrFold<arith::MulIOp>(loc, hi, ratio), lo);
    };

    SmallVector<Value> newShape = dropDim(shapes, collapsedDim);
    SmallVector<Value> newStrides = dropDim(strides, collapsedDim);
    unsigned indexBitWidth = indexTy.getIntOrFloatBitWidth();
    Value zero =
        tt::intel::findOrCreateIntConstant(loc, 0, indexBitWidth, builder);
    Value one =
        tt::intel::findOrCreateIntConstant(loc, 1, indexBitWidth, builder);

    // Bound the *load*, not the collapsed dimension: merge this load's own
    // collapsed index, clamped into [0, shapes[cd]-1]. The clamp is two-sided
    // because the extent now depends on a load operand, and a negative
    // `offsets[cd]` must not be able to drive it non-positive. Clamp the shape
    // before subtracting: `maxsi(shapes[cd]-1, 0)` would compute INT32_MIN-1,
    // wrap to INT32_MAX and preserve the garbage instead of clamping it.
    Value lastIdx = builder.createOrFold<arith::SubIOp>(
        loc, builder.createOrFold<arith::MaxSIOp>(loc, collapsedShape, one),
        one);
    Value clampedIdx = builder.createOrFold<arith::MaxSIOp>(
        loc,
        builder.createOrFold<arith::MinSIOp>(loc, collapsedOffset, lastIdx),
        zero);
    Value merged = merge(clampedIdx, newShape[collapsedDim]);

    // An empty collapsed dimension cannot keep an extent of 0: every surface
    // field is emitted as `extent - 1`, so a zero declares a 16MB surface over
    // a pointer to nothing instead of padding. Declare the smallest legal
    // surface and let the index guard below force the load out of range, which
    // is what makes it pad - on the 2D path as well as the generic one. An
    // empty merged dimension needs the same: `isCandidate` declines only a
    // folded one, and at runtime the extent `clampedIdx * ratio + 0` is 0
    // whenever `offsets[cd] <= 0`.
    //
    // Write both branches as multiplies and not as
    // `select(nonEmpty, merged, floor)`: on the middle branch the merged
    // extent is the stride-one dimension, and a top-level `arith.select` is
    // invisible to `ttgi::isDivisible`, which would cost the load its
    // `block_io` attribute for 16-bit and narrower types (for a 32-bit element
    // the divisor is 1, which that helper answers unconditionally). Its `muli`
    // case is an OR over the operands and its `addi` case an AND, so both terms
    // have to stay divisible by the block extent: `muli(merged, nonEmpty)`
    // reduces to the query it answers today, and the floor is a multiple of the
    // block extent by construction.
    std::optional<int64_t> floorExtent = emptyExtentFloor(
        makeTensorDescOp.getType().getBlockType(), collapsedDim);
    assert(floorExtent && "isCandidate should have declined this fusion");
    Value collapsedNonEmpty = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, collapsedShape, zero);
    Value mergedNonEmpty = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, shapes[mergedDim], zero);
    Value nonEmpty = builder.createOrFold<arith::ExtUIOp>(
        loc, indexTy,
        builder.createOrFold<arith::AndIOp>(loc, collapsedNonEmpty,
                                            mergedNonEmpty));
    // `canDeclareLegalSurface` has declined anything above the 24-bit surface
    // limit, so the floor fits an `int`.
    Value floor = tt::intel::findOrCreateIntConstant(
        loc, static_cast<int>(*floorExtent), indexBitWidth, builder);
    // One term per statement: ops appear in the order they are created, and C++
    // leaves the evaluation order of a call's arguments unspecified, so
    // building this as one nested expression makes the emitted order
    // compiler-dependent.
    Value isEmpty = builder.createOrFold<arith::SubIOp>(loc, one, nonEmpty);
    Value keepExtent =
        builder.createOrFold<arith::MulIOp>(loc, merged, nonEmpty);
    Value emptyExtent =
        builder.createOrFold<arith::MulIOp>(loc, floor, isEmpty);
    newShape[collapsedDim] =
        builder.createOrFold<arith::AddIOp>(loc, keepExtent, emptyExtent);

    Value newDesc = tt::MakeTensorDescOp::create(
        builder, loc, newDescType, makeTensorDescOp.getBase(), newShape,
        newStrides, makeTensorDescOp.getPadding());
    LLVM_DEBUG(llvm::dbgs() << "new MakeTensorDescOp:\n  " << newDesc << "\n");

    // Merge the load indices the same way, unclamped: they are indices.
    builder.setInsertionPoint(descLoadOp);
    SmallVector<Value> newOffsets = dropDim(offsets, collapsedDim);
    Value mergedIdx = merge(offsets[collapsedDim], newOffsets[collapsedDim]);

    // The collapsed dimension's block extent is 1, so the rank-3 form pads the
    // whole block exactly when `offsets[cd]` is out of range, whereas the
    // merged coordinate loses that check: `offsets[md]` (or the per-element
    // tile coordinate, which is added before the bounds check) can lift it back
    // into range (issue #8070). Force it to the merged extent, which no row of
    // the block can fall below, so the fused load pads too. An empty merged
    // dimension makes the rank-3 load pure padding at any index, so it counts
    // as out of range as well: an in-range `offsets[cd]` can still merge to an
    // index inside the floor (`0 * ratio + 0` is row 0) and read real data.
    // Written as multiplies for the same `isDivisible` reason as `nonEmpty`
    // above.
    // One term per statement here too, for the emission order reason above.
    Value nonNegative = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, collapsedOffset, zero);
    Value belowExtent = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, collapsedOffset, collapsedShape);
    Value inBounds =
        builder.createOrFold<arith::AndIOp>(loc, nonNegative, belowExtent);
    Value inRange = builder.createOrFold<arith::ExtUIOp>(
        loc, indexTy,
        builder.createOrFold<arith::AndIOp>(loc, inBounds, mergedNonEmpty));
    Value outOfRange = builder.createOrFold<arith::SubIOp>(loc, one, inRange);
    Value keepIdx =
        builder.createOrFold<arith::MulIOp>(loc, mergedIdx, inRange);
    Value padIdx = builder.createOrFold<arith::MulIOp>(loc, outOfRange,
                                                       newShape[collapsedDim]);
    newOffsets[collapsedDim] =
        builder.createOrFold<arith::AddIOp>(loc, keepIdx, padIdx);

    auto resType = cast<tt::TensorDescType>(newDesc.getType()).getBlockType();
    auto newDescLoadOp = tt::DescriptorLoadOp::create(
        builder, descLoadOp.getLoc(), resType, newDesc, newOffsets,
        descLoadOp.getCachePolicyAttr());
    newDescLoadOp->setAttrs(descLoadOp->getAttrs());

    LLVM_DEBUG(llvm::dbgs() << "newDescLoadOp:\n  " << newDescLoadOp << "\n");

    // Propagate the new descriptor load result.
    IRMapping mapping;
    propagateToUser(newDescLoadOp->getResult(0), descLoadOp.getResult(),
                    reshapeOp, reshapeOp, mapping);

    cleanUp.insert(descLoadOp);
    cleanUp.insert(makeTensorDescOp);
  }

  // Candidate is a reshape operation of having one of the following forms:
  //   - tt.dot(tt.reshape(tt.load(..., )))
  //   - tt.dot(tt.reshape(tt.descriptor_load(..., )))
  // Where:
  //  - the reshape operation drops the outermost or the middle dimension of the
  //    operand, which is a 3-dim tensor whose dropped dimension has extent one
  //  - the reshape result is used by a dot operation
  //  - the reshape operation uses the result of a 3-dim load operation on a
  //    tensor descriptor (transitively) defined by a `make_tensor_descriptor`
  //  - the descriptor's block shape equals the loaded shape (the *tensor*
  //    extent on the dropped dimension is arbitrary)
  //  - the collapse can declare a legal 2D block surface (see the
  //    `DescriptorLoadOp` overload)
  bool isCandidate(tt::ReshapeOp reshapeOp) const {
    assert(reshapeOp && "Expecting a valid reshape operation");

    ArrayRef<int64_t> reshapeOperandShape =
        reshapeOp.getSrc().getType().getShape();
    std::optional<unsigned> collapsedDim = getCollapsedDim(reshapeOperandShape);
    if (!collapsedDim)
      return false;

    // The reshape must drop exactly the unit-extent dimension.
    if (!llvm::equal(dropDim(reshapeOperandShape, *collapsedDim),
                     reshapeOp.getType().getShape()))
      return false;

    // Check whether \p reshapeOp is used by a `dotOp`.
    auto usedByDotOp = [](tt::ReshapeOp reshapeOp) {
      if (!reshapeOp->hasOneUse())
        return false;

      Operation *user = *reshapeOp->getUsers().begin();
      while (user) {
        if (isa<tt::DotOp>(user))
          return true;
        if (!user->hasOneUse())
          break;
        user = *user->getUsers().begin();
      }

      return false;
    };

    if (!usedByDotOp(reshapeOp))
      return false;

    Operation *defOp = reshapeOp.getSrc().getDefiningOp();
    if (!defOp)
      return false;
    if (auto descLoadOp = dyn_cast<tt::DescriptorLoadOp>(defOp))
      return isCandidate(descLoadOp);

    return false;
  }

  bool isCandidate(tt::DescriptorLoadOp descLoadOp) const {
    if (!descLoadOp->hasOneUse())
      return false;

    std::optional<tt::MakeTensorDescOp> makeTensorDescOp =
        tt::intel::findMakeTensorDescOp(descLoadOp.getDesc());
    if (!makeTensorDescOp)
      return false;

    tt::TensorDescType descTy = makeTensorDescOp->getResult().getType();
    auto tensorTy = cast<RankedTensorType>(descTy.getBlockType());
    // `tt.descriptor_load` only requires a matching element type and count, so
    // the block shape may differ from the loaded shape (e.g. a rank-reducing
    // load). The collapsed dimension is found in the loaded shape while the
    // fusion indexes the descriptor's shape/strides, so the two must agree.
    if (!llvm::equal(tensorTy.getShape(), descLoadOp.getType().getShape()))
      return false;

    std::optional<unsigned> dim = getCollapsedDim(tensorTy.getShape());
    if (!dim)
      return false;
    const unsigned collapsedDim = *dim;
    const unsigned mergedDim = collapsedDim + 1;

    auto decline = [&](StringRef reason) {
      LLVM_DEBUG(llvm::dbgs() << "Declining to fuse:\n  " << *descLoadOp
                              << "\n  reason: " << reason << "\n");
      return false;
    };

    OperandRange strides = makeTensorDescOp->getStrides();
    OperandRange shapes = makeTensorDescOp->getShape();

    // A value that folds to a non-positive constant. Non-foldable values are
    // trusted, as everywhere else on this path.
    auto isProvablyNonPositive = [](Value v) {
      std::optional<int64_t> cst = tt::intel::getFoldedConstantValue(v);
      return cst && *cst <= 0;
    };

    // The stride ratio is an unsigned division and `isDivisible` compares its
    // operands as unsigned, so a negative stride would otherwise fuse and turn
    // the whole block into padding (issue #8001, case 2). Reject before
    // `isProvablyDivisible` so no non-positive value ever reaches it.
    if (isProvablyNonPositive(strides[collapsedDim]) ||
        isProvablyNonPositive(strides[mergedDim]))
      return decline("non-positive stride on a collapsed dimension");

    // An empty collapsed dimension makes the rank-3 load pure padding, which no
    // rank-2 surface can express, and a non-positive merged extent wraps to a
    // huge surface (the field is emitted as `extent - 1`). A zero `shapes[md]`
    // passes the divisibility check below, since `0 % n == 0`. Only a folded
    // value is declined: a runtime-empty dimension of either kind fuses, and
    // `fuseMakeTensorDescOp`'s floor and index guard make the load pad.
    if (isProvablyNonPositive(shapes[collapsedDim]))
      return decline("empty collapsed dimension");
    if (isProvablyNonPositive(shapes[mergedDim]))
      return decline("non-positive extent on the merged dimension");

    // Collapsing the middle dimension promotes `shapes[0]` to the surface
    // height and `strides[0]` to its pitch; neither was a surface field before
    // fusion and nothing downstream checks their sign.
    if (collapsedDim == 1 &&
        (isProvablyNonPositive(shapes[0]) || isProvablyNonPositive(strides[0])))
      return decline("non-positive promoted surface height or pitch");

    // The fusion divides strides[collapsedDim] by strides[mergedDim], so it is
    // only valid when that division is exact (e.g. not for padded strides).
    if (!isProvablyDivisible(strides[collapsedDim], strides[mergedDim]))
      return decline("strides are not provably divisible");

    // Fusion replaces the per-dimension bounds check with a single check on the
    // merged dimension, which is only sound if a block load can never straddle
    // a boundary between two "rows" of the collapsed dimension, e.g. a
    // ragged/padded last block (issues/7464).
    int64_t blockExtent = tensorTy.getDimSize(mergedDim);
    // `isDivisible` requires a positive divisor, and a zero-extent block
    // reaches it as 0.
    if (blockExtent <= 0)
      return decline("merged-dimension block extent is not positive");
    if (!mlir::triton::gpu::intel::isDivisible(shapes[mergedDim], blockExtent))
      return decline("merged extent is not provably a multiple of the block");

    // The merged dimension keeps its lower bound only through the merged
    // coordinate, which `offsets[cd]*ratio` can lift back into range (#8070).
    // Guarding it like the collapsed index is not an option: the block spans
    // this dimension, so a negative index pads only its leading rows, which one
    // rank-2 coordinate cannot express. Hence a decline - and only for a folded
    // constant, so a computed or runtime negative stays open in #8070.
    std::optional<int64_t> mergedOffset =
        tt::intel::getFoldedConstantValue(descLoadOp.getIndices()[mergedDim]);
    if (mergedOffset && *mergedOffset < 0)
      return decline("negative index on the merged dimension");

    return canDeclareLegalSurface(descLoadOp, *makeTensorDescOp, tensorTy,
                                  collapsedDim, decline);
  }

  /// `triton_gen.2Dblockload` bounds, in bytes for the byte-valued fields.
  static constexpr int64_t MaxSurfaceExtent = 1 << 24;
  static constexpr int64_t MinSurfaceBytes = 64;

  /// Return the extent the fusion emits when the collapsed or the merged
  /// dimension turns out to be empty at runtime, or `std::nullopt` if it cannot
  /// be computed without overflow. A zero extent is not an option: every
  /// surface field is emitted as `extent - 1`, so a zero reads back as
  /// 0xFFFFFF, i.e. a 16MB surface over a pointer to nothing. The empty case
  /// therefore declares the smallest legal surface and forces the load out of
  /// range instead, which is what makes it pad on both lowering paths.
  ///
  /// The floor is a multiple of the merged dimension's block extent so that
  /// `DescriptorLoadOpConversion`'s static mask classification is unchanged: it
  /// asks `ttgi::isDivisible` about the emitted shape and index, `arith.addi`
  /// is an AND over its operands there, and a floor that is not a multiple of
  /// the block would demote the dimension to per-element masking for every
  /// fused load with a dynamic collapsed shape - not just the empty ones.
  static std::optional<int64_t> emptyExtentFloor(RankedTensorType tensorTy,
                                                 unsigned collapsedDim) {
    int64_t blockExtent = tensorTy.getDimSize(collapsedDim + 1);
    // `isCandidate` has already declined a non-positive block extent; keep the
    // helper total anyway, because it divides by this below.
    if (blockExtent <= 0)
      return std::nullopt;
    // Collapsing dimension 0 makes the extent the surface height in rows, which
    // has no lower bound; one block of rows is enough.
    if (collapsedDim == 0)
      return blockExtent;

    // Collapsing the middle dimension makes it the surface width, which must be
    // at least `MinSurfaceBytes` wide. Every operand of the two `divideCeil`s
    // is positive, so the unsigned overload they resolve to is exact.
    unsigned elemBitWidth = tensorTy.getElementTypeBitWidth();
    if (elemBitWidth < 8)
      return std::nullopt;
    int64_t elemBytes = elemBitWidth / 8;
    int64_t minElems = llvm::divideCeil(MinSurfaceBytes, elemBytes);
    int64_t blocks = llvm::divideCeil(minElems, blockExtent);
    int64_t floorElems;
    if (llvm::MulOverflow(blockExtent, blocks, floorElems))
      return std::nullopt;
    return floorElems;
  }

  /// Return true if the collapse of \p collapsedDim can declare a legal 2D
  /// block surface for every foldable input of \p descLoadOp. Everything here
  /// is foldable-only: a field built from function arguments is trusted, as
  /// `LowerTo2DBlockLoad`'s own checks and the `triton_gen.2Dblockload`
  /// verifier trust it. All comparisons are signed, because a folded value can
  /// be negative and an unsigned compare would turn a negative pitch into a
  /// huge legal-looking one.
  ///
  /// Residual, knowingly accepted: when either of the two shapes or strides
  /// being merged does not fold there is no bracket and the width rules below
  /// stay dead. This is a stronger assumption than trusting a dynamic
  /// descriptor operand, because the merged extent is *derived*: the stride
  /// ratio can amplify individually sane inputs past the field limits. Three
  /// ways it can go wrong at runtime, none of them statically visible:
  ///   - the extent exceeds the 24-bit surface field, so the `- 1` encoding
  ///     truncates it;
  ///   - `clampedIdx * ratio + shapeMd` overflows the i32 that the surface
  ///     fields are computed in;
  ///   - the emitted `trunci(divui(...))` ratio truncates, or that overflow
  ///     wraps the extent to a small value - including back to zero, which
  ///     re-creates the `0 - 1 == 0xFFFFFF` surface that the floor below exists
  ///     to prevent, by a route no static check can see. That last one is worse
  ///     than an over-large field, not merely different.
  /// Declining instead would lose the motivating dynamic case (a `batch*heads`
  /// collapse), and clamping at runtime would turn real data into padding.
  template <typename DeclineFn>
  static bool canDeclareLegalSurface(tt::DescriptorLoadOp descLoadOp,
                                     tt::MakeTensorDescOp makeTensorDescOp,
                                     RankedTensorType tensorTy,
                                     unsigned collapsedDim, DeclineFn decline) {
    const unsigned mergedDim = collapsedDim + 1;
    OperandRange shapes = makeTensorDescOp.getShape();
    OperandRange strides = makeTensorDescOp.getStrides();
    OperandRange offsets = descLoadOp.getIndices();
    auto folded = [](Value v) { return tt::intel::getFoldedConstantValue(v); };

    // The ratio is divided in i64 and truncated to the i32 shape type, and the
    // truncated value scales both the merged extent and the merged offset.
    // Validate it first and unconditionally: it is consumed independently of
    // the extent, so this check must not become conditional on the extent
    // folding.
    std::optional<int64_t> strideCd = folded(strides[collapsedDim]);
    std::optional<int64_t> strideMd = folded(strides[mergedDim]);
    std::optional<int64_t> ratio;
    if (strideCd && strideMd && *strideMd > 0) {
      // Both are positive here (non-positive folded strides are declined
      // earlier), so the signed division matches the emitted `divui`.
      ratio = *strideCd / *strideMd;
      if (!llvm::isInt<32>(*ratio))
        return decline("stride ratio does not fit i32");
    }

    // The merged offset is computed in i32 from the *unclamped* collapsed
    // index, so it can wrap while the clamped extent and every surface field
    // stay small and legal. Check the multiply and the add separately: that is
    // stricter than "the final offset fits", which is accepted - the cost is a
    // lost fusion on a bizarre input, not unsafety.
    std::optional<int64_t> offCd = folded(offsets[collapsedDim]);
    std::optional<int64_t> offMd = folded(offsets[mergedDim]);
    if (ratio && offCd && offMd) {
      int64_t scaled, mergedOffset;
      if (llvm::MulOverflow(*offCd, *ratio, scaled) ||
          !llvm::isInt<32>(scaled) ||
          llvm::AddOverflow(scaled, *offMd, mergedOffset) ||
          !llvm::isInt<32>(mergedOffset))
        return decline("merged load offset does not fit i32");
    }

    // The extent this fusion will emit, bracketed over the clamp range. Do not
    // gate this on the collapsed index folding: the emitted extent is
    // `clamp(offCd, 0, max(shapeCd,1)-1) * ratio + shapeMd`, monotone
    // non-decreasing in the clamped index because `ratio` is non-negative here
    // (non-positive folded strides are declined earlier), so the endpoints of
    // the clamp range bracket every extent this load can emit. Gating on
    // `offCd` instead leaves every rule below dead for a dynamic collapsed
    // index while the load still fuses - i.e. it declares a surface nothing
    // ever checks. A folded index collapses the bracket to a point and
    // reproduces the exact check. Both shapes fold positive here, so the
    // empty-dimension branch of the emitted extent is dead and `lastIdx` is
    // `shapeCd - 1`; that branch's floor is checked separately below, because
    // it is live whenever either shape does not fold.
    std::optional<int64_t> shapeCd = folded(shapes[collapsedDim]);
    std::optional<int64_t> shapeMd = folded(shapes[mergedDim]);
    std::optional<int64_t> minExtent, maxExtent;
    bool extentIsExact = false;
    if (ratio && shapeCd && shapeMd) {
      int64_t lastIdx = std::max<int64_t>(*shapeCd, 1) - 1;
      int64_t loIdx = 0, hiIdx = lastIdx;
      if (offCd)
        loIdx = hiIdx = std::clamp<int64_t>(*offCd, 0, lastIdx);
      auto mergedExtent = [&](int64_t idx) -> std::optional<int64_t> {
        int64_t scaled, value;
        if (llvm::MulOverflow(idx, *ratio, scaled) ||
            llvm::AddOverflow(scaled, *shapeMd, value) ||
            !llvm::isInt<32>(value))
          return std::nullopt;
        return value;
      };
      minExtent = mergedExtent(loIdx);
      maxExtent = mergedExtent(hiIdx);
      if (!minExtent || !maxExtent)
        return decline("merged extent does not fit i32");
      extentIsExact = (loIdx == hiIdx);
    }

    // The extent emitted for an empty dimension is a compile-time
    // constant, so it is checked unconditionally: unlike the bracket above,
    // nothing about it depends on an input that folds. Only the floor-vs-pitch
    // rule below can actually fire today, because `verifyTensorSize`
    // (lib/Dialect/Triton/IR/Traits.cpp) requires a power-of-two element count
    // and caps it at 2^20: the block extent is therefore a power of two, which
    // makes the floor either exactly `MinSurfaceBytes` or a power-of-two
    // multiple of the element size above it - aligned, and at most 2^23 bytes
    // either way. The magnitude and alignment rules are kept because they cost
    // one comparison each and the alternative is a field derived by this pass
    // that nothing ever checks.
    std::optional<int64_t> floorExtent =
        emptyExtentFloor(tensorTy, collapsedDim);
    if (!floorExtent)
      return decline("empty-dimension extent floor cannot be computed");

    if (collapsedDim == 0) {
      // The merged extent becomes the surface height, in rows: only the 24-bit
      // cap applies, and `wouldOverflow` never checks the height.
      if (maxExtent && *maxExtent > MaxSurfaceExtent)
        return decline("merged extent exceeds the surface height limit");
      if (*floorExtent > MaxSurfaceExtent)
        return decline("empty-dimension height floor exceeds its limit");
      return true;
    }

    // The merged extent becomes the surface width, and the collapse
    // simultaneously promotes `shapes[0]` to the height and `strides[0]` to the
    // pitch.
    if (std::optional<int64_t> height = folded(shapes[0]);
        height && *height > MaxSurfaceExtent)
      return decline("promoted surface height exceeds its limit");

    // A sub-byte element type has no byte-granular width this can validate:
    // `elemBitWidth / 8` is 0, which is how `LowerTo2DBlockLoad` would size the
    // surface too. Decline rather than declare a field nothing checked.
    unsigned elemBitWidth = tensorTy.getElementTypeBitWidth();
    if (elemBitWidth < 8)
      return decline("sub-byte element type has no checkable surface width");
    const int64_t elemBytes = elemBitWidth / 8;

    // Compute the byte fields with checked arithmetic: `strides[0] * elemBytes`
    // alone can exceed int64_t for a hostile constant descriptor.
    auto toBytes = [&](int64_t elems, int64_t &bytes) {
      return !llvm::MulOverflow(elems, elemBytes, bytes);
    };
    int64_t minExtentBytes = 0, maxExtentBytes = 0;
    bool haveExtentBytes = false;
    if (minExtent && maxExtent) {
      if (!toBytes(*minExtent, minExtentBytes) ||
          !toBytes(*maxExtent, maxExtentBytes))
        return decline("merged surface width in bytes overflows");
      haveExtentBytes = true;
    }
    int64_t floorBytes = 0;
    if (!toBytes(*floorExtent, floorBytes))
      return decline("empty-dimension width floor in bytes overflows");
    int64_t pitchBytes = 0;
    bool havePitchBytes = false;
    if (std::optional<int64_t> pitch = folded(strides[0])) {
      if (!toBytes(*pitch, pitchBytes))
        return decline("promoted surface pitch in bytes overflows");
      havePitchBytes = true;
    }

    // Per field, never gated on every input folding: the verifier tests the
    // pitch rules independently of the width, so an all-or-nothing fold gate
    // would let a known-bad pitch through whenever anything else is dynamic.
    // The `%` rules duplicate a `MaterializeBlockPointer` check that merely
    // withholds `block_io`; the magnitude rules are the load-bearing ones.
    //
    // Which end of the width bracket binds depends on the rule: the lower bound
    // binds at its minimum, the 24-bit cap and the pitch at its maximum. The
    // `%` rule cannot be decided from endpoints at all, so it applies only to
    // an exact bracket. Skipping it otherwise cannot admit a misaligned width
    // that the hardware ever sees: `MaterializeBlockPointer` grants `block_io`
    // only when the stride-one extent is a multiple of `ceil(32/elemBitWidth)`
    // elements, which is exactly 4 bytes for 8- and 16-bit types, while a 32-
    // or 64-bit element is 4-byte aligned on its own. Sub-byte types are
    // declined above, so a width that fails this rule on a non-exact bracket
    // belongs to a load that never got `block_io` in the first place.
    if (haveExtentBytes &&
        (minExtentBytes < MinSurfaceBytes ||
         maxExtentBytes > MaxSurfaceExtent ||
         (extentIsExact &&
          maxExtentBytes % std::max<int64_t>(4, elemBytes) != 0)))
      return decline("merged surface width is not legal");
    if (floorBytes < MinSurfaceBytes || floorBytes > MaxSurfaceExtent ||
        floorBytes % std::max<int64_t>(4, elemBytes) != 0)
      return decline("empty-dimension width floor is not legal");
    if (havePitchBytes &&
        (pitchBytes < MinSurfaceBytes || pitchBytes > MaxSurfaceExtent ||
         pitchBytes % 16 != 0))
      return decline("promoted surface pitch is not legal");
    if (havePitchBytes && floorBytes > pitchBytes)
      return decline("empty-dimension width floor exceeds its pitch");
    if (haveExtentBytes && havePitchBytes && maxExtentBytes > pitchBytes)
      return decline("merged surface width exceeds its pitch");

    return true;
  }

  /// Return true if \p numerator is provably divisible by \p denominator.
  static bool isProvablyDivisible(Value numerator, Value denominator) {
    // If both are the same value, trivially divisible.
    if (numerator == denominator)
      return true;

    // If numerator is defined by arith.muli and one operand is the
    // denominator, it is divisible.
    if (auto mulOp = numerator.getDefiningOp<arith::MulIOp>()) {
      if (mulOp.getLhs() == denominator || mulOp.getRhs() == denominator)
        return true;
    }

    // If both fold, the remainder is exact at any magnitude, so answer here
    // rather than narrowing the denominator to `unsigned` below. Both are
    // required positive because this models the emitted `divui`, for which a
    // negative operand is a large unsigned value that a signed `%` does not
    // answer for (the caller already declines non-positive strides).
    std::optional<int64_t> numCst =
        tt::intel::getFoldedConstantValue(numerator);
    std::optional<int64_t> denCst =
        tt::intel::getFoldedConstantValue(denominator);
    if (numCst && denCst && *numCst > 0 && *denCst > 0)
      return *numCst % *denCst == 0;

    // If denominator is a constant, use isDivisible which leverages
    // tt.divisibility attributes on function arguments and constants.
    APInt denVal;
    if (matchPattern(denominator, m_ConstantInt(&denVal)))
      if (std::optional<int64_t> den = denVal.trySExtValue())
        return *den > 0 &&
               mlir::triton::gpu::intel::isDivisible(numerator, *den);

    return false;
  }

  // If \p user is not \p sentinel, propagate \p newVal to \p user. Otherwise
  // terminate the propagation.
  virtual void propagateToUser(Value newVal, Value origVal, Operation *user,
                               Operation *sentinel, IRMapping &mapping) final {
    assert(user && sentinel && "Expecting valid operations");
    assert(llvm::is_contained(origVal.getUsers(), user) && "Invalid usage");

    LLVM_DEBUG({
      llvm::dbgs() << "In " << __func__ << "\n";
      llvm::dbgs() << "user of: ";
      if (origVal.getDefiningOp()) {
        llvm::dbgs() << "\n  " << *origVal.getDefiningOp() << "\n";
      } else {
        origVal.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << " ";
      }
      llvm::dbgs() << "is:\n  ";
      user->dumpPretty();
    });

    if (user == sentinel) {
      LLVM_DEBUG(llvm::dbgs() << "Reached sentinel\n");
      sentinel->replaceAllUsesWith(newVal.getDefiningOp());
      cleanUp.insert(sentinel);
      return;
    }

    Location loc = user->getLoc();
    if (auto loadOp = dyn_cast<tt::LoadOp>(user)) {
      OpBuilder rewriter(loadOp);
      auto newLoadOp = tt::LoadOp::create(rewriter, loadOp.getLoc(), newVal,
                                          loadOp.getMask(), loadOp.getOther(),
                                          loadOp.getCachePolicyAttr(),
                                          loadOp.getIsVolatile());
      newLoadOp->setAttrs(loadOp->getAttrs());
      mapping.map(static_cast<Operation *>(loadOp),
                  static_cast<Operation *>(newLoadOp));
      LLVM_DEBUG(llvm::dbgs().indent(2) << "newLoadOp: " << newLoadOp << "\n");
      cleanUp.insert(loadOp);
      return propagateToUsers(newLoadOp, loadOp.getResult(), loadOp, sentinel,
                              mapping);
    }

    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      int opNum = -1;
      for (OpOperand &operand : yieldOp->getOpOperands()) {
        if (operand.get() == origVal) {
          opNum = operand.getOperandNumber();
          yieldOp->setOperand(operand.getOperandNumber(), newVal);
          break;
        }
      }

      // Update the yield's parent operation result type.
      Operation *parentOp = yieldOp->getParentOp();
      OpResult res = parentOp->getOpResult(opNum);
      res.setType(newVal.getType());
      return;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user))
      return propagateToLoop(newVal, origVal, forOp, sentinel, mapping);

    llvm_unreachable("Unexpected kind of user");
  }

  /// Queried once per fused chain. The pass creates one fuser per run, so this
  /// is built once per run. Reusing it across fusions is safe because the
  /// rewrites add and erase operations but create no blocks or regions, and
  /// MLIR keeps the intra-block ordering `properlyDominates` relies on up to
  /// date.
  DominanceInfo domInfo;
};

struct TritonIntelFuseReshape
    : tt::intel::impl::TritonIntelFuseReshapeBase<TritonIntelFuseReshape> {
public:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    FuseReshapeWithLoad fuser;
    fuser.run(moduleOp);
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
