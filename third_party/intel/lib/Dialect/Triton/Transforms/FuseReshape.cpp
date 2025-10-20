#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/DefUseChain.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
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
//   %ptr = tt.make_tensor_ptr %base, [%s0,%s1,%s2], [%a,%b,%c], [%x,%y,%z]
//                       {order = array<i32: 2, 1, 0>} : <tensor<1x512x64xf16>>
//   %load = tt.load %ptr : !tt.ptr<tensor<1x512x64xf16>>
//   %A = tt.reshape %load : tensor<1x512x64xf16> -> tensor<512x64xf16>
//   dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
// into:
//   %d = %a / %b
//   %ptr = tt.make_tensor_ptr %base, [%s0*%d+%s1,%s2], [%b,%c], [%x*%div+%y,%z]
//                       {order = array<i32: 1, 0>} : <tensor<512x64xf16>>
//   %A = tt.load %ptr : !tt.ptr<tensor<512x64xf16>>
//   dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
class FuseReshapeWithLoad : public tt::intel::Fuser {
public:
  void run(ModuleOp moduleOp) {
    // Collect def-use chains originating at a `MakeTensorPtrOp` operation
    // and terminating at a candidate `tt::ReshapeOp` operation.
    // Note: A candidate `reshapeOp` must use the result of a `loadOp` using a
    // ptr created by the `MakeTensorPtrOp` rooting the def-use chain.
    DefUseChainManager manager;
    moduleOp.walk([&](tt::ReshapeOp reshapeOp) {
      if (isCandidate(reshapeOp)) {
        auto loadOp = cast<tt::LoadOp>(reshapeOp.getSrc().getDefiningOp());
        tt::MakeTensorPtrOp makeTensorPtrOp =
            *triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
        manager.createChains(makeTensorPtrOp, reshapeOp);
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
  void fuse(const DefUseChain &chain) final {
    assert(
        isa<tt::MakeTensorPtrOp>(chain.getStart()) &&
        "Expecting 'chain' to be rooted by a 'tt.make_tensor_ptr' operation");
    assert(isa<tt::ReshapeOp>(chain.getEnd()) &&
           "Expecting 'chain' to be terminated by a 'tt.reshape' operation");

    auto makeTensorPtrOp = cast<tt::MakeTensorPtrOp>(chain.getStart());
    auto reshapeOp = cast<tt::ReshapeOp>(chain.getEnd());
    auto loadOp = cast<tt::LoadOp>(reshapeOp.getSrc().getDefiningOp());
    LLVM_DEBUG(llvm::dbgs() << "Fusing:\n  " << reshapeOp << "\nwith:\n  "
                            << loadOp << "\n");

    // Create a MakeTensorPtrOp yielding a 2-dim block pointer.
    auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
    [[maybe_unused]] ArrayRef<int64_t> resShape =
        cast<RankedTensorType>(ptrType.getPointeeType()).getShape();
    assert(resShape[0] == 1 && "Result shape should have extent equal to 1 in "
                               "the outermost dimension");

    auto tensorType = cast<RankedTensorType>(reshapeOp.getType());
    auto newPtrType =
        tt::PointerType::get(tensorType, ptrType.getAddressSpace());

    // Compute the index of the innermost dimension.
    ArrayRef<int> order = makeTensorPtrOp.getOrder();
    assert(order.size() == 3 && order[0] == 2 && "Invalid order");

    unsigned innermostDimIdx = 0;
    for (int elem : makeTensorPtrOp.getOrder()) {
      if (elem == 0)
        break;
      ++innermostDimIdx;
    }

    OpBuilder builder(makeTensorPtrOp);
    Location loc = makeTensorPtrOp.getLoc();
    OperandRange shapes = makeTensorPtrOp.getShape();
    OperandRange strides = makeTensorPtrOp.getStrides();
    OperandRange offsets = makeTensorPtrOp.getOffsets();

    // Collapse the 3-dim tensor into a 2-dim tensor.
    // Given a make_tensor_ptr with:
    //   shape  [s0, s1, s2]
    //   stride [a, b, c]
    //   offset [x, y, z]
    //   order  [2, 1, 0]
    // We create a make_tensor_ptr with:
    //   shape  [s0 * a / b + s1, s2]
    //   stride [b, c]
    //   offset [x * a / b + y, z]
    //   order  [1, 0]
    SmallVector<Value> newShape(makeTensorPtrOp.getShape().drop_front());
    SmallVector<Value> newStrides(makeTensorPtrOp.getStrides().drop_front());
    SmallVector<Value> newOffsets(makeTensorPtrOp.getOffsets().drop_front());

    unsigned newInnermostDimIdx = (innermostDimIdx - 1);
    unsigned newOutermostDimIdx = !newInnermostDimIdx;
    auto div = builder.create<arith::DivUIOp>(loc, strides[0],
                                              newStrides[newOutermostDimIdx]);

    newShape[newOutermostDimIdx] = builder.create<arith::AddIOp>(
        loc, builder.create<arith::MulIOp>(loc, shapes[0], div),
        newShape[newOutermostDimIdx]);
    newOffsets[newOutermostDimIdx] = builder.create<arith::AddIOp>(
        loc,
        builder.create<arith::MulIOp>(
            loc, offsets[0],
            builder.create<arith::TruncIOp>(loc, offsets[0].getType(), div)),
        newOffsets[newOutermostDimIdx]);

    Value ptr = builder.create<tt::MakeTensorPtrOp>(
        loc, newPtrType, makeTensorPtrOp.getBase(), newShape, newStrides,
        newOffsets,
        DenseI32ArrayAttr::get(
            builder.getContext(),
            makeTensorPtrOp.getOrderAttr().asArrayRef().drop_front()));

    LLVM_DEBUG(llvm::dbgs() << "newMakeTensorPtrOp:\n  " << ptr << "\n");

    // Propagate the new ptr through the def-use chain.
    IRMapping mapping;
    propagateToUsers(ptr, chain, mapping);
    cleanUp.insert(makeTensorPtrOp);

    // We have collapsed 2 dimensions into one, therefore we need to adjust the
    // boundary check of the new load.
    auto newLoadOp =
        cast<tt::LoadOp>(mapping.lookup(static_cast<Operation *>(loadOp)));
    ArrayRef<int> boundaryCheck = newLoadOp.getBoundaryCheck();
    for (int idx : boundaryCheck) {
      assert(idx == (newInnermostDimIdx + 1) &&
             "Unexpected boundary check idx");
      newLoadOp.setBoundaryCheck({static_cast<int>(newInnermostDimIdx)});
    }
  }

  // Candidate is of the form:
  //   tt.dot(tt.reshape(tt.load(..., )))
  // Where:
  //  - the reshape operation drops the outermost dimension of the operand,
  //    which is a 3-dim tensor with outermost dimension extent equal to one
  //  - the reshape result is used by a dot operation
  //  - the reshape operation uses the result of a 3-dim load operation on a
  //    block pointer (transitively) defined by a `make_tensor_ptr` operation
  //  - the block pointer points to a tensor that has extent equal to 1 on the
  //    outermost dimension
  //  - the load operation doesn't have boundary checks on either of the
  //    dimensions collapsed
  bool isCandidate(tt::ReshapeOp reshapeOp) const {
    assert(reshapeOp && "Expecting a valid reshape operation");

    ArrayRef<int64_t> reshapeOperandShape =
        reshapeOp.getSrc().getType().getShape();
    if (reshapeOperandShape.size() != 3 || reshapeOperandShape.front() != 1)
      return false;

    ArrayRef<int64_t> reshapeResultShape = reshapeOp.getType().getShape();
    if (reshapeResultShape.size() != reshapeOperandShape.size() - 1)
      return false;

    for (auto pair :
         llvm::zip(reshapeOperandShape.drop_front(), reshapeResultShape)) {
      if (std::get<0>(pair) != std::get<1>(pair))
        return false;
    }

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

    // The reshape operation uses the result of a load operation.
    Operation *defOp = reshapeOp.getSrc().getDefiningOp();
    if (!defOp || !isa<tt::LoadOp>(defOp))
      return false;

    auto loadOp = cast<tt::LoadOp>(defOp);
    if (!loadOp->hasOneUse())
      return false;

    // The load uses a 3-dim block pointer defined by a make_tensor_ptr
    // operation.
    Type ptrType = loadOp.getPtr().getType();
    if (!tt::isTensorPointerType(ptrType))
      return false;

    std::optional<tt::MakeTensorPtrOp> makeTensorPtrOp =
        triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
    if (!makeTensorPtrOp)
      return false;

    tt::PointerType ptrTy = makeTensorPtrOp->getResult().getType();
    auto tensorTy = cast<RankedTensorType>(ptrTy.getPointeeType());
    assert((tensorTy.getRank() == 3 && tensorTy.getDimSize(0) == 1) &&
           "Unexpected tensor type");

    // Ensure the outermost dimension is the one with highest order.
    ArrayRef<int> order = makeTensorPtrOp->getOrder();
    if (order.front() != tensorTy.getRank() - 1)
      return false;

    unsigned innermostDimIdx = 0;
    for (int idx : order) {
      if (idx == 0)
        break;
      ++innermostDimIdx;
    }

    // Ensure the load operation checks at most the innermost dimension.
    return llvm::all_of(loadOp.getBoundaryCheck(),
                        [&](int idx) { return idx == innermostDimIdx; });
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
    if (auto advanceOp = dyn_cast<tt::AdvanceOp>(user)) {
      OpBuilder rewriter(advanceOp);
      SmallVector<Value> newOffsets(advanceOp.getOffsets().drop_front());
      auto newAdvanceOp = rewriter.create<tt::AdvanceOp>(loc, newVal.getType(),
                                                         newVal, newOffsets);
      mapping.map(static_cast<Operation *>(advanceOp),
                  static_cast<Operation *>(newAdvanceOp));
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "newAdvanceOp: " << newAdvanceOp << "\n");
      cleanUp.insert(advanceOp);
      return propagateToUsers(newAdvanceOp, advanceOp.getResult(), advanceOp,
                              sentinel, mapping);
    }

    if (auto loadOp = dyn_cast<tt::LoadOp>(user)) {
      OpBuilder rewriter(loadOp);
      auto newLoadOp = rewriter.create<tt::LoadOp>(
          loadOp.getLoc(), newVal, loadOp.getMask(), loadOp.getOther(),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
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
