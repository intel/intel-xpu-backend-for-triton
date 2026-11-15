#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/DefUseChain.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
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
//   %i = max(min(%x, max(%s0,1)-1), 0)         // collapsed index, clamped
//   %p = tt.addptr %base, %i * %a              // the slice at that index
//   %n = (%s1 > 0)                             // merged dimension not empty
//   %e = %s1 * %n + %f * (1 - %n)              // one block (%f) when empty
//   %g = (%x >= 0) & (%x < %s0) & %n           // the rank-3 load reads data
//   %j = %y * %g + %e * (1 - %g)               // %e (pads) otherwise
//   %desc = tt.make_tensor_descriptor %p, [%e,%s2], [%b,%c]
//                       : !tt.tensordesc<512x64xf16>
//   %A = tt.descriptor_load %desc[%j,%z] -> tensor<512x64xf16>
//   dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
// %n and %e are only emitted for a %s1 that does not fold; otherwise %e is
// %s1. A unit-extent middle dimension (e.g. a one-head tile of a contiguous
// (TOKENS, HEADS, HEAD_DIM) tensor) is collapsed the same way, with %i and %g
// taken from %y, %n and %e from %s2, and the guard on %z:
//   %desc = tt.make_tensor_descriptor %p, [%s0,%e], [%a,%c]
//   %A = tt.descriptor_load %desc[%x,%j] -> tensor<64x128xf16>
// The collapsed index goes into the base rather than into a neighbouring
// coordinate, so the new descriptor's strides and (but for the empty case)
// shapes are the original ones. Merging it instead (row %x*(%a/%b) + %y over
// an extent derived from the strides) declared surfaces the rank-3 descriptor
// never had: a negative runtime stride turned the block into padding (issue
// #8072), and a large one pushed the derived extent past the 24-bit surface
// field.
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

    // The base depends on this load's collapsed index, so the new descriptor
    // can only stay where the original one was if that index is available
    // there; otherwise it is rebuilt at the load. A loop-invariant index still
    // gets it hoisted back out of the loop by the TTGIR LICM, and
    // `LowerTo2DBlockLoad` re-extracts every shape/stride field per load
    // anyway.
    Operation *insertionPoint =
        domInfo.properlyDominates(collapsedOffset, makeTensorDescOp)
            ? static_cast<Operation *>(makeTensorDescOp)
            : static_cast<Operation *>(descLoadOp);
    OpBuilder builder(insertionPoint);

    unsigned indexBitWidth = indexTy.getIntOrFloatBitWidth();
    Value zero =
        tt::intel::findOrCreateIntConstant(loc, 0, indexBitWidth, builder);
    Value one =
        tt::intel::findOrCreateIntConstant(loc, 1, indexBitWidth, builder);

    // Offset the base by the collapsed index, clamped into
    // [0, shapes[cd]-1] so that an out-of-range index - which the guard below
    // turns into padding - cannot move the base outside the tensor. Clamp the
    // shape before subtracting: `maxsi(shapes[cd]-1, 0)` would compute
    // INT32_MIN-1, wrap to INT32_MAX and preserve the garbage instead of
    // clamping it. The product is formed in the stride's i64, as the rank-3
    // load forms its own address, so it keeps the stride's sign and range.
    Value lastIdx = builder.createOrFold<arith::SubIOp>(
        loc, builder.createOrFold<arith::MaxSIOp>(loc, collapsedShape, one),
        one);
    Value clampedIdx = builder.createOrFold<arith::MaxSIOp>(
        loc,
        builder.createOrFold<arith::MinSIOp>(loc, collapsedOffset, lastIdx),
        zero);
    Value collapsedStride = strides[collapsedDim];
    Value wideIdx = builder.createOrFold<arith::ExtSIOp>(
        loc, collapsedStride.getType(), clampedIdx);
    Value baseOffset =
        builder.createOrFold<arith::MulIOp>(loc, wideIdx, collapsedStride);
    // `make_tensor_descriptor` requires every leading stride to be a multiple
    // of 16 bytes, and `MaterializeBlockPointer` trusts that for a stride it
    // knows nothing else about. AxisInfo cannot carry it through the product:
    // for a stride without a `tt.divisibility` hint it proves only the element
    // size for the new base, which the 4-byte base check then rejects - while
    // the unfused load, its leading index folded into the base by
    // `LowerTo2DBlockLoad`, kept 2D block I/O. So state the contract on the
    // product. A folded stride needs no hint, and the check on the operands
    // keeps it off a value the folder returned instead of a new op.
    auto offsetOp = baseOffset.getDefiningOp<arith::MulIOp>();
    unsigned elemBytes =
        getIntOrFloatOrPtrBitWidth(
            makeTensorDescOp.getType().getBlockType().getElementType()) /
        8;
    if (offsetOp && llvm::is_contained(offsetOp->getOperands(), wideIdx) &&
        llvm::is_contained(offsetOp->getOperands(), collapsedStride) &&
        !tt::intel::getFoldedConstantValue(collapsedStride) && elemBytes &&
        16 % elemBytes == 0)
      offsetOp->setAttr("tt.divisibility",
                        builder.getI32IntegerAttr(16 / elemBytes));
    Value base = makeTensorDescOp.getBase();
    Value newBase = builder.createOrFold<tt::AddPtrOp>(loc, base.getType(),
                                                       base, baseOffset);

    // A merged dimension that is empty at runtime would leave the rank-2 view
    // with a zero extent, which the 2D block lowering emits as `0 - 1`: a 16MB
    // surface that reads real data instead of padding. Declare one block
    // instead and let the guard below force the load out of range. A folded
    // extent needs neither, since `LowerTo2DBlockLoad` sees it and keeps the
    // load off 2D block I/O. Written as multiplies for the same `isDivisible`
    // reason as the guard; the floor is the block extent itself, so the sum
    // keeps whatever that query answered for `shapes[md]`. Like any width,
    // the floor is not held to the hardware's 64-byte minimum, which
    // `LowerTo2DBlockLoad` deliberately does not enforce either.
    //
    // The other dimension kept is passed through as is - on the middle
    // branch `shapes[0]` and `strides[0]` become the surface height and pitch
    // - and so is exactly as trusted as in a rank-2 descriptor: a runtime
    // zero height or negative pitch is not caught here.
    SmallVector<Value> newShape = dropDim(shapes, collapsedDim);
    Value mergedShape = shapes[mergedDim];
    Value mergedNonEmpty;
    if (!tt::intel::getFoldedConstantValue(mergedShape)) {
      int64_t blockExtent =
          makeTensorDescOp.getType().getBlockType().getDimSize(mergedDim);
      Value floor = tt::intel::findOrCreateIntConstant(
          loc, static_cast<int>(blockExtent), indexBitWidth, builder);
      // One term per statement: ops appear in the order they are created, and
      // C++ leaves the evaluation order of a call's arguments unspecified, so
      // building this as one nested expression makes the emitted order
      // compiler-dependent.
      mergedNonEmpty = builder.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, mergedShape, zero);
      Value nonEmpty =
          builder.createOrFold<arith::ExtUIOp>(loc, indexTy, mergedNonEmpty);
      Value isEmpty = builder.createOrFold<arith::SubIOp>(loc, one, nonEmpty);
      Value keepExtent =
          builder.createOrFold<arith::MulIOp>(loc, mergedShape, nonEmpty);
      Value emptyExtent =
          builder.createOrFold<arith::MulIOp>(loc, floor, isEmpty);
      newShape[collapsedDim] =
          builder.createOrFold<arith::AddIOp>(loc, keepExtent, emptyExtent);
    }

    Value newDesc = tt::MakeTensorDescOp::create(
        builder, loc, newDescType, newBase, newShape,
        dropDim(strides, collapsedDim), makeTensorDescOp.getPadding());
    LLVM_DEBUG(llvm::dbgs() << "new MakeTensorDescOp:\n  " << newDesc << "\n");

    // The collapsed dimension's block extent is 1, so the rank-3 form pads the
    // whole block exactly when `offsets[cd]` is out of range, a check the
    // rank-2 view no longer has. Force the index of the dimension that
    // followed it to that dimension's extent, which no element of the block
    // can fall below, so the fused load pads too (issue #8070). Written as
    // multiplies and not as `select(inRange, idx, extent)`: a top-level
    // `arith.select` is invisible to `ttgi::isDivisible`, which
    // `DescriptorLoadOpConversion` asks about the emitted index to classify
    // the mask statically, and its `muli` case is an OR over the operands and
    // its `addi` case an AND. An empty merged dimension makes the rank-3 load
    // pure padding, so it counts as out of range too: an in-range index would
    // otherwise read the floor declared above.
    builder.setInsertionPoint(descLoadOp);
    SmallVector<Value> newOffsets = dropDim(offsets, collapsedDim);
    Value nonNegative = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, collapsedOffset, zero);
    Value belowExtent = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, collapsedOffset, collapsedShape);
    Value inBounds =
        builder.createOrFold<arith::AndIOp>(loc, nonNegative, belowExtent);
    if (mergedNonEmpty)
      inBounds =
          builder.createOrFold<arith::AndIOp>(loc, inBounds, mergedNonEmpty);
    Value inRange =
        builder.createOrFold<arith::ExtUIOp>(loc, indexTy, inBounds);
    Value outOfRange = builder.createOrFold<arith::SubIOp>(loc, one, inRange);
    Value keepIdx = builder.createOrFold<arith::MulIOp>(
        loc, newOffsets[collapsedDim], inRange);
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

    if (!getCollapsedDim(tensorTy.getShape()))
      return false;

    // Offsetting the base by `offsets[cd] * strides[cd]` elements needs an
    // element that is a whole number of bytes: the 2D block path packs a
    // sub-byte element, while `tt.addptr` would advance by its storage size.
    // Such a load keeps its rank-3 form.
    if (getIntOrFloatOrPtrBitWidth(tensorTy.getElementType()) < 8) {
      LLVM_DEBUG(llvm::dbgs() << "Declining to fuse:\n  " << *descLoadOp
                              << "\n  reason: sub-byte element type\n");
      return false;
    }

    return true;
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

/// AxisInfo reads these at their stored length, so a rank change would leave
/// them stale.
static bool hasRankIndexedHint(Operation *op) {
  return op->hasAttr("tt.contiguity") || op->hasAttr("tt.divisibility") ||
         op->hasAttr("tt.constancy");
}

static RankedTensorType dropUnitDim(Type type) {
  auto tensorTy = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorTy.getShape().drop_front(),
                               tensorTy.getElementType());
}

/// A dense tensor constant or a splat, which `rebuildAtRank2` can rebuild.
static bool isRebuildable(Value v) {
  auto cstOp = v.getDefiningOp<arith::ConstantOp>();
  return (cstOp && isa<DenseElementsAttr>(cstOp.getValue())) ||
         v.getDefiningOp<tt::SplatOp>();
}

/// Rebuild an `isRebuildable` value at rank 2 right after its definition. The
/// original may have other users, so it is rebuilt, not retyped.
static Value rebuildAtRank2(OpBuilder &builder, Value v) {
  Operation *defOp = v.getDefiningOp();
  builder.setInsertionPointAfter(defOp);
  RankedTensorType newTy = dropUnitDim(v.getType());
  if (auto cstOp = dyn_cast<arith::ConstantOp>(defOp))
    return arith::ConstantOp::create(
        builder, cstOp.getLoc(),
        cast<DenseElementsAttr>(cstOp.getValue()).reshape(newTy));
  return tt::SplatOp::create(builder, defOp->getLoc(), newTy,
                             cast<tt::SplatOp>(defOp).getSrc());
}

/// The one tensor operand of `op` that is not `isRebuildable`, or null if
/// there is none or more than one.
static Value singleNonRebuildableOperand(Operation *op) {
  Value found;
  for (Value operand : op->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()) || isRebuildable(operand))
      continue;
    if (found)
      return {};
    found = operand;
  }
  return found;
}

/// The ops from `reshapeOp`'s `tt.descriptor_load` up to the reshape, load
/// first, or an empty list if the reshape cannot be hoisted.
static SmallVector<Operation *> matchHoistChain(tt::ReshapeOp reshapeOp) {
  RankedTensorType srcTy = reshapeOp.getSrc().getType();
  if (reshapeOp->use_empty() || srcTy.getRank() != 3 ||
      srcTy.getDimSize(0) != 1 ||
      reshapeOp.getType().getShape() != srcTy.getShape().drop_front())
    return {};

  SmallVector<Operation *> chain;
  for (Value cur = reshapeOp.getSrc();;) {
    Operation *defOp = cur.getDefiningOp();
    if (!defOp || !cur.hasOneUse() || hasRankIndexedHint(defOp))
      return {};
    chain.push_back(defOp);
    if (isa<tt::DescriptorLoadOp>(defOp))
      break;
    if (auto transOp = dyn_cast<tt::TransOp>(defOp)) {
      if (transOp.getOrder()[0] != 0)
        return {};
      cur = transOp.getSrc();
      continue;
    }
    // Dropping a leading unit dim keeps the element order, so it commutes
    // with any single-result elementwise op but inline asm, which receives
    // its elements grouped by thread.
    if (!defOp->hasTrait<OpTrait::Elementwise>() ||
        defOp->getNumResults() != 1 || isa<tt::ElementwiseInlineAsmOp>(defOp))
      return {};
    cur = singleNonRebuildableOperand(defOp);
    if (!cur)
      return {};
  }
  if (chain.size() < 2)
    return {};
  std::reverse(chain.begin(), chain.end());
  return chain;
}

/// Rewrite a `matchHoistChain` chain at rank 2, and queue the reshape, the
/// replaced `tt.trans` ops and the rebuilt originals in `cleanUp`.
static void rewriteHoistChain(tt::ReshapeOp reshapeOp,
                              ArrayRef<Operation *> chain,
                              SmallPtrSetImpl<Operation *> &cleanUp) {
  OpBuilder builder(reshapeOp.getContext());
  builder.setInsertionPointAfter(chain.front());
  Value prev = chain.front()->getResult(0);
  Value cur =
      tt::ReshapeOp::create(builder, reshapeOp.getLoc(),
                            dropUnitDim(prev.getType()).getShape(), prev);
  for (Operation *op : llvm::drop_begin(chain)) {
    if (auto transOp = dyn_cast<tt::TransOp>(op)) {
      ArrayRef<int32_t> order = transOp.getOrder();
      builder.setInsertionPoint(transOp);
      cur = tt::TransOp::create(builder, transOp.getLoc(), cur,
                                {order[1] - 1, order[2] - 1});
      cleanUp.insert(transOp);
      prev = transOp.getResult();
      continue;
    }
    for (OpOperand &operand : op->getOpOperands()) {
      Value v = operand.get();
      if (v == prev) {
        operand.set(cur);
        continue;
      }
      if (!isa<RankedTensorType>(v.getType()))
        continue;
      operand.set(rebuildAtRank2(builder, v));
      cleanUp.insert(v.getDefiningOp());
    }
    prev = cur = op->getResult(0);
    cur.setType(dropUnitDim(cur.getType()));
  }
  reshapeOp.replaceAllUsesWith(cur);
  cleanUp.insert(reshapeOp);
}

/// Move each `tt.reshape` dropping the leading unit dim of a 3-D tensor up to
/// its `tt.descriptor_load`, where `FuseReshapeWithLoad` or Combine folds it,
/// through single-use elementwise ops (retyped in place; other tensor operands
/// must be constants or splats, rebuilt at rank 2) and dim-0-preserving
/// `tt.trans` ops (rebuilt at rank 2). Inline asm, multi-result ops, and
/// tt.contiguity/divisibility/constancy on those ops or the load stop it.
static void hoistUnitDimReshapesToDescriptorLoads(ModuleOp moduleOp) {
  SmallVector<std::pair<tt::ReshapeOp, SmallVector<Operation *>>> chains;
  moduleOp.walk([&](tt::ReshapeOp reshapeOp) {
    SmallVector<Operation *> chain = matchHoistChain(reshapeOp);
    if (!chain.empty())
      chains.emplace_back(reshapeOp, std::move(chain));
  });

  SmallPtrSet<Operation *, 8> cleanUp;
  for (auto &[reshapeOp, chain] : chains)
    rewriteHoistChain(reshapeOp, chain, cleanUp);
  tt::intel::eraseOperations(cleanUp);
}

/// Store `%x` directly where a `tt.descriptor_store` stores a `tt.reshape(%x)`
/// that only prepends unit dims, making the store rank-reduced. A rank-1 `%x`
/// is not folded: the 2D block store needs a rank-2 value.
static void foldReshapeIntoDescriptorStores(ModuleOp moduleOp) {
  moduleOp.walk([](tt::DescriptorStoreOp storeOp) {
    auto reshapeOp = storeOp.getSrc().getDefiningOp<tt::ReshapeOp>();
    if (!reshapeOp)
      return;

    ArrayRef<int64_t> srcShape = reshapeOp.getSrc().getType().getShape();
    ArrayRef<int64_t> blockShape =
        storeOp.getDesc().getType().getBlockType().getShape();
    const size_t rank = srcShape.size();
    // Matching element counts don't imply unit leading dims for an empty
    // tensor (`0x16` -> `2x0x16`), so they are checked explicitly.
    if (rank < 2 || reshapeOp.getType().getShape() != blockShape ||
        blockShape.take_back(rank) != srcShape ||
        llvm::any_of(
            blockShape.drop_back(rank), [](int64_t dim) { return dim != 1; }))
      return;

    storeOp.getSrcMutable().assign(reshapeOp.getSrc());
    if (reshapeOp->use_empty())
      reshapeOp->erase();
  });
}

struct TritonIntelFuseReshape
    : tt::intel::impl::TritonIntelFuseReshapeBase<TritonIntelFuseReshape> {
public:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    hoistUnitDimReshapesToDescriptorLoads(moduleOp);
    FuseReshapeWithLoad fuser;
    fuser.run(moduleOp);
    foldReshapeIntoDescriptorStores(moduleOp);
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
