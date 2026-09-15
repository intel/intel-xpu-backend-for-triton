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
#include <limits>

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
//   %e = (%i*%d + %s1) * (%s0 > 0)             // merged extent
//   %desc = tt.make_tensor_descriptor %base, [%e,%s2], [%b,%c]
//                       : !tt.tensordesc<512x64xf16>
//   %A = tt.descriptor_load %desc[%x*%d+%y,%z] -> tensor<512x64xf16>
//   dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
// A unit-extent middle dimension (e.g. a one-head tile of a contiguous
// (TOKENS, HEADS, HEAD_DIM) tensor) is collapsed the same way, with %d = %b/%c:
//   %desc = tt.make_tensor_descriptor %base, [%s0,%e], [%a,%c]
//   %A = tt.descriptor_load %desc[%x,%y*%d+%z] -> tensor<64x128xf16>
// The merged extent is per *load*, not per dimension: bounding it with the
// whole collapsed dimension ((%s0-1)*%d+%s1) is tight only for the last index
// of that dimension and lets every other index read rows the rank-3 form pads
// (issue #8001).
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
    DominanceInfo domInfo;
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
    Value ratio = builder.createOrFold<arith::TruncIOp>(
        loc, indexTy,
        builder.createOrFold<arith::DivUIOp>(loc, strides[collapsedDim],
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

    // An empty collapsed dimension keeps an extent of 0, which is what makes
    // the load pad on the generic path, as the rank-3 form does. Write it as a
    // multiply and not as `select(shapes[cd] > 0, merged, 0)`: on the middle
    // branch the merged extent is the stride-one dimension, and a top-level
    // `arith.select` is invisible to `ttgi::isDivisible`, which would cost the
    // load its `block_io` attribute. That helper's `muli` case is an OR over
    // the operands, so `muli(merged, nonEmpty)` reduces to the query it
    // answers today.
    Value nonEmpty = builder.createOrFold<arith::ExtUIOp>(
        loc, indexTy,
        builder.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                            collapsedShape, zero));
    newShape[collapsedDim] =
        builder.createOrFold<arith::MulIOp>(loc, merged, nonEmpty);

    Value newDesc = tt::MakeTensorDescOp::create(
        builder, loc, newDescType, makeTensorDescOp.getBase(), newShape,
        newStrides, makeTensorDescOp.getPadding());
    LLVM_DEBUG(llvm::dbgs() << "new MakeTensorDescOp:\n  " << newDesc << "\n");

    // Merge the load indices the same way, unclamped: they are indices.
    builder.setInsertionPoint(descLoadOp);
    SmallVector<Value> newOffsets = dropDim(offsets, collapsedDim);
    newOffsets[collapsedDim] =
        merge(offsets[collapsedDim], newOffsets[collapsedDim]);

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
    // passes the divisibility check below, since `0 % n == 0`.
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
    // `isDivisible` takes an `unsigned` divisor and divides by it unguarded, so
    // it must be positive: a zero-extent block reaches it as 0 and raises
    // SIGFPE, and a value wider than `unsigned` narrows to 0 (same) or to 1 (a
    // false "divisible"). Only the zero is reachable today - the tensor
    // verifier caps a block at 2^20 elements - but the narrowing is silent.
    if (blockExtent <= 0 || !fitsUnsigned(blockExtent))
      return decline("merged-dimension block extent is not a positive "
                     "`unsigned`");
    if (!mlir::triton::gpu::intel::isDivisible(shapes[mergedDim], blockExtent))
      return decline("merged extent is not provably a multiple of the block");

    return canDeclareLegalSurface(descLoadOp, *makeTensorDescOp, tensorTy,
                                  collapsedDim, decline);
  }

  /// Return true if the collapse of \p collapsedDim can declare a legal 2D
  /// block surface for every foldable input of \p descLoadOp. Everything here
  /// is foldable-only: a field built from function arguments is trusted, as
  /// `LowerTo2DBlockLoad`'s own checks and the `triton_gen.2Dblockload`
  /// verifier trust it. All comparisons are signed, because a folded value can
  /// be negative and an unsigned compare would turn a negative pitch into a
  /// huge legal-looking one.
  template <typename DeclineFn>
  static bool canDeclareLegalSurface(tt::DescriptorLoadOp descLoadOp,
                                     tt::MakeTensorDescOp makeTensorDescOp,
                                     RankedTensorType tensorTy,
                                     unsigned collapsedDim, DeclineFn decline) {
    // `triton_gen.2Dblockload` bounds, in bytes for the byte-valued fields.
    constexpr int64_t MaxSurfaceExtent = 1 << 24;
    constexpr int64_t MinSurfaceBytes = 64;

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
      if (!fitsInt32(*ratio))
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
      if (llvm::MulOverflow(*offCd, *ratio, scaled) || !fitsInt32(scaled) ||
          llvm::AddOverflow(scaled, *offMd, mergedOffset) ||
          !fitsInt32(mergedOffset))
        return decline("merged load offset does not fit i32");
    }

    // The extent this fusion will emit, where it folds.
    std::optional<int64_t> shapeCd = folded(shapes[collapsedDim]);
    std::optional<int64_t> shapeMd = folded(shapes[mergedDim]);
    std::optional<int64_t> extent;
    if (ratio && shapeCd && shapeMd && offCd) {
      int64_t lastIdx = std::max<int64_t>(*shapeCd, 1) - 1;
      int64_t clampedIdx = std::clamp<int64_t>(*offCd, 0, lastIdx);
      int64_t scaled, value;
      if (llvm::MulOverflow(clampedIdx, *ratio, scaled) ||
          llvm::AddOverflow(scaled, *shapeMd, value) || !fitsInt32(value))
        return decline("merged extent does not fit i32");
      extent = value;
    }

    if (collapsedDim == 0) {
      // The merged extent becomes the surface height, in rows: only the 24-bit
      // cap applies, and `wouldOverflow` never checks the height.
      if (extent && *extent > MaxSurfaceExtent)
        return decline("merged extent exceeds the surface height limit");
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
    int64_t extentBytes = 0;
    bool haveExtentBytes = false;
    if (extent) {
      if (llvm::MulOverflow(*extent, elemBytes, extentBytes))
        return decline("merged surface width in bytes overflows");
      haveExtentBytes = true;
    }
    int64_t pitchBytes = 0;
    bool havePitchBytes = false;
    if (std::optional<int64_t> pitch = folded(strides[0])) {
      if (llvm::MulOverflow(*pitch, elemBytes, pitchBytes))
        return decline("promoted surface pitch in bytes overflows");
      havePitchBytes = true;
    }

    // Per field, never gated on every input folding: the verifier tests the
    // pitch rules independently of the width, so an all-or-nothing fold gate
    // would let a known-bad pitch through whenever anything else is dynamic.
    // The `%` rules duplicate a `MaterializeBlockPointer` check that merely
    // withholds `block_io`; the magnitude rules are the load-bearing ones.
    if (haveExtentBytes &&
        (extentBytes < MinSurfaceBytes || extentBytes > MaxSurfaceExtent ||
         extentBytes % std::max<int64_t>(4, elemBytes) != 0))
      return decline("merged surface width is not legal");
    if (havePitchBytes &&
        (pitchBytes < MinSurfaceBytes || pitchBytes > MaxSurfaceExtent ||
         pitchBytes % 16 != 0))
      return decline("promoted surface pitch is not legal");
    if (haveExtentBytes && havePitchBytes && extentBytes > pitchBytes)
      return decline("merged surface width exceeds its pitch");

    return true;
  }

  /// Return true if \p value is representable as an `int32_t`.
  static bool fitsInt32(int64_t value) {
    return value >= std::numeric_limits<int32_t>::min() &&
           value <= std::numeric_limits<int32_t>::max();
  }

  /// Return true if \p value is representable as an `unsigned`.
  static bool fitsUnsigned(int64_t value) {
    return value >= 0 &&
           static_cast<uint64_t>(value) <= std::numeric_limits<unsigned>::max();
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
    // required positive: the caller declines non-positive strides, and a
    // negative operand would make `%` implementation-defined in sign.
    std::optional<int64_t> numCst =
        tt::intel::getFoldedConstantValue(numerator);
    std::optional<int64_t> denCst =
        tt::intel::getFoldedConstantValue(denominator);
    if (numCst && denCst && *numCst > 0 && *denCst > 0)
      return *numCst % *denCst == 0;

    // If denominator is a constant, use isDivisible which leverages
    // tt.divisibility attributes on function arguments and constants.
    APInt denVal;
    if (matchPattern(denominator, m_ConstantInt(&denVal)) && !denVal.isZero()) {
      // `isDivisible` takes an `unsigned` divisor and this denominator is an
      // i64 stride: narrowing a wider value would divide by zero, or hit that
      // helper's `divisor == 1` early return and report a divisibility that
      // does not hold. Decline instead of narrowing. The guard lives here, and
      // not ahead of the whole function, because the cases above are exact for
      // a stride of any magnitude.
      uint64_t den = denVal.getZExtValue();
      if (den > std::numeric_limits<unsigned>::max())
        return false;
      return mlir::triton::gpu::intel::isDivisible(numerator, den);
    }

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
