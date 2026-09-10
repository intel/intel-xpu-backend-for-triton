#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/DefUseChain.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include "llvm/Support/raw_ostream.h"

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
//   %desc = tt.make_tensor_descriptor %base, [(%s0-1)*%d+%s1,%s2], [%b,%c]
//                       : !tt.tensordesc<512x64xf16>
//   %A = tt.descriptor_load %desc[%x*%d+%y,%z] -> tensor<512x64xf16>
//   dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
// A unit-extent middle dimension (e.g. a one-head tile of a contiguous
// (TOKENS, HEADS, HEAD_DIM) tensor) is collapsed the same way, with %d = %b/%c:
//   %desc = tt.make_tensor_descriptor %base, [%s0,(%s1-1)*%d+%s2], [%a,%c]
//   %A = tt.descriptor_load %desc[%x,%y*%d+%z] -> tensor<64x128xf16>
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

    OpBuilder builder(makeTensorDescOp);
    Location loc = makeTensorDescOp.getLoc();
    OperandRange shapes = makeTensorDescOp.getShape();
    OperandRange strides = makeTensorDescOp.getStrides();

    // An index pair (i,j) in `collapsedDim`/`mergedDim` addresses the same
    // element as the single index i * d + j, with d = the stride ratio. Erasing
    // `collapsedDim` leaves the merged entry at index `collapsedDim`, so shape
    // [s0,s1,s2] / stride [a,b,c] yields [(s0-1)*a/b+s1, s2] / [b,c] or
    // [s0, (s1-1)*b/c+s2] / [a,c].
    auto div = arith::DivUIOp::create(builder, loc, strides[collapsedDim],
                                      strides[mergedDim]);
    Value ratio = builder.createOrFold<arith::TruncIOp>(
        loc, shapes[collapsedDim].getType(), div);
    auto merge = [&](Value hi, Value lo) -> Value {
      return arith::AddIOp::create(
          builder, loc, arith::MulIOp::create(builder, loc, hi, ratio), lo);
    };

    // The extent merges the largest *index* of the collapsed dimension, hence
    // s-1: over-declaring it would loosen the descriptor's bounds check and,
    // for an innermost `mergedDim`, push the block surface width past the
    // pitch.
    SmallVector<Value> newShape = dropDim(shapes, collapsedDim);
    SmallVector<Value> newStrides = dropDim(strides, collapsedDim);
    Value one = arith::ConstantIntOp::create(builder, loc,
                                             shapes[collapsedDim].getType(), 1);
    newShape[collapsedDim] =
        merge(arith::SubIOp::create(builder, loc, shapes[collapsedDim], one),
              newShape[collapsedDim]);

    Value newDesc = tt::MakeTensorDescOp::create(
        builder, loc, newDescType, makeTensorDescOp.getBase(), newShape,
        newStrides, makeTensorDescOp.getPadding());
    LLVM_DEBUG(llvm::dbgs() << "new MakeTensorDescOp:\n  " << newDesc << "\n");

    // Merge the load indices the same way, without the s-1: they are indices.
    builder.setInsertionPoint(descLoadOp);
    OperandRange offsets = descLoadOp.getIndices();
    SmallVector<Value> newOffsets = dropDim(offsets, collapsedDim);
    newOffsets[collapsedDim] =
        merge(offsets[collapsedDim], newOffsets[collapsedDim]);

    auto resType = cast<tt::TensorDescType>(newDesc.getType()).getBlockType();
    auto newDescLoadOp = tt::DescriptorLoadOp::create(
        builder, descLoadOp.getLoc(), resType, newDesc, newOffsets,
        descLoadOp.getCache(), descLoadOp.getEvict());
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
  //  - the load operation doesn't have boundary checks on either of the
  //    dimensions collapsed
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

    // The fusion divides strides[collapsedDim] by strides[mergedDim], so it is
    // only valid when that division is exact (e.g. not for padded strides).
    OperandRange strides = makeTensorDescOp->getStrides();
    if (!isProvablyDivisible(strides[collapsedDim], strides[mergedDim]))
      return false;

    // Fusion replaces the per-dimension bounds check with a single check on the
    // merged dimension, which is only sound if a block load can never straddle
    // a boundary between two "rows" of the collapsed dimension, e.g. a
    // ragged/padded last block (issues/7464).
    OperandRange shapes = makeTensorDescOp->getShape();
    int64_t blockExtent = tensorTy.getDimSize(mergedDim);
    if (!mlir::triton::gpu::intel::isDivisible(shapes[mergedDim], blockExtent))
      return false;

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

    // If denominator is a constant, use isDivisible which leverages
    // tt.divisibility attributes on function arguments and constants.
    APInt denVal;
    if (matchPattern(denominator, m_ConstantInt(&denVal)) && !denVal.isZero())
      return mlir::triton::gpu::intel::isDivisible(numerator,
                                                   denVal.getZExtValue());

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
                                          loadOp.getCache(), loadOp.getEvict(),
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
