#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/DefUseChain.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-intel-fuse-transpose"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELFUSETRANSPOSE
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

// Transform:
//   %ptr = make_tensor_ptr [shapeM, shapeK], [strideM, strideK], [offM, offK]
//        : tt.ptr<tensor<MxKxf16>>
//   %load = tt.load %ptr : tt.ptr<tensor<MxKxf16>>
//   %trans = tt.trans %load : tensor<MxKxf16> -> tensor<KxMxf16>
//   tt.dot(%trans, %b, ...)
// into:
//   %ptr = make_tensor_ptr [shapeK, shapeM], [strideK, strideM], [offK, offM]
//        : tt.ptr<tensor<KxMxf16>>
//   %load = tt.load %ptr : tt.ptr<tensor<KxMxf16>>
//   tt.dot(%load, %b, ...)
//
// Also handles the tensor descriptor variant:
//   %desc = make_tensor_desc %base, [shapeM, shapeK], [strideM, strideK]
//         : tensor_desc<MxKxf16>
//   %load = descriptor_load %desc, [idxM, idxK] : tensor<MxKxf16>
//   %trans = tt.trans %load : tensor<MxKxf16> -> tensor<KxMxf16>
//   tt.dot(%trans, %b, ...)
// into:
//   %desc = make_tensor_desc %base, [shapeK, shapeM], [strideK, strideM]
//         : tensor_desc<KxMxf16>
//   %load = descriptor_load %desc, [idxK, idxM] : tensor<KxMxf16>
//   tt.dot(%load, %b, ...)
class FuseTransposeWithLoad : public tt::intel::Fuser {
public:
  void run(ModuleOp moduleOp) {
    // Collect def-use chains originating at a MakeTensorPtrOp or
    // MakeTensorDescOp and terminating at a candidate TransOp.
    DefUseChainManager manager;
    moduleOp.walk([&](tt::TransOp transOp) {
      if (isCandidate(transOp)) {
        Operation *srcOp = transOp.getSrc().getDefiningOp();
        assert(srcOp && "Expected a valid source operation");

        llvm::TypeSwitch<Operation *>(srcOp)
            .Case<tt::LoadOp>([&](auto loadOp) {
              auto maybeMakeTensorPtrOp =
                  tt::intel::findDefiningOpOfType<tt::MakeTensorPtrOp>(
                      loadOp.getPtr());
              assert(maybeMakeTensorPtrOp &&
                     "isCandidate should have verified this");
              manager.createChains(*maybeMakeTensorPtrOp, transOp);
            })
            .Case<tt::DescriptorLoadOp>([&](auto descLoadOp) {
              auto maybeMakeTensorDescOp =
                  tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(
                      descLoadOp.getDesc());
              assert(maybeMakeTensorDescOp &&
                     "isCandidate should have verified this");
              manager.createChains(*maybeMakeTensorDescOp, transOp);
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

    // Fuse load->trans operations.
    Fuser::fuse(manager.getChains());

    // Remove operations that are no longer used.
    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);
  }

private:
  void fuse(const DefUseChain &chain) final override {
    assert(isa<tt::TransOp>(chain.getEnd()) &&
           "Expecting 'chain' to be terminated by a 'tt.trans' operation");

    llvm::TypeSwitch<Operation *>(chain.getStart())
        .Case<tt::MakeTensorPtrOp>([&](auto makeTensorPtrOp) {
          fuseMakeTensorPtrOp(chain, makeTensorPtrOp);
        })
        .Case<tt::MakeTensorDescOp>([&](auto makeTensorDescOp) {
          fuseMakeTensorDescOp(chain, makeTensorDescOp);
        })
        .Default([](Operation *) {
          llvm_unreachable("Unexpected 'chain' root operation kind");
        });
  }

  void fuseMakeTensorPtrOp(const DefUseChain &chain,
                           tt::MakeTensorPtrOp makeTensorPtrOp) {
    assert(chain.getStart() == makeTensorPtrOp &&
           "Unexpected 'chain' start operation");

    auto transOp = cast<tt::TransOp>(chain.getEnd());
    auto loadOp = cast<tt::LoadOp>(transOp.getSrc().getDefiningOp());
    LLVM_DEBUG(llvm::dbgs()
               << "Fusing:\n  " << transOp << "\nwith:\n  " << loadOp << "\n");

    // Create a MakeTensorPtrOp yielding a block pointer to the transposed
    // tensor.
    auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
    auto tensorType = cast<RankedTensorType>(transOp.getType());
    auto newPtrType =
        tt::PointerType::get(tensorType, ptrType.getAddressSpace());
    SmallVector<Value> newShape(llvm::reverse(makeTensorPtrOp.getShape()));
    SmallVector<Value> newStrides(llvm::reverse(makeTensorPtrOp.getStrides()));
    SmallVector<Value> newOffsets(llvm::reverse(makeTensorPtrOp.getOffsets()));

    OpBuilder builder(makeTensorPtrOp);
    Value ptr = tt::MakeTensorPtrOp::create(
        builder, makeTensorPtrOp.getLoc(), newPtrType,
        makeTensorPtrOp.getBase(), newShape, newStrides, newOffsets,
        makeTensorPtrOp.getOrderAttr());
    LLVM_DEBUG(llvm::dbgs() << "newMakeTensorPtrOp:\n  " << ptr << "\n");

    // Propagate the new ptr through the def-use chain.
    IRMapping mapping;
    propagateToUsers(ptr, chain, mapping);
    cleanUp.insert(makeTensorPtrOp);
  }

  void fuseMakeTensorDescOp(const DefUseChain &chain,
                            tt::MakeTensorDescOp makeTensorDescOp) {
    assert(chain.getStart() == makeTensorDescOp &&
           "Unexpected 'chain' start operation");

    auto transOp = cast<tt::TransOp>(chain.getEnd());
    auto descLoadOp =
        cast<tt::DescriptorLoadOp>(transOp.getSrc().getDefiningOp());
    LLVM_DEBUG(llvm::dbgs() << "Fusing:\n  " << transOp << "\nwith:\n  "
                            << descLoadOp << "\n");

    // Create a MakeTensorDescOp with swapped shape/strides for the transposed
    // tensor.
    auto tensorType = cast<RankedTensorType>(transOp.getType());
    auto newDescType =
        tt::TensorDescType::get(transOp->getContext(), tensorType);

    OpBuilder builder(makeTensorDescOp);
    Location loc = makeTensorDescOp.getLoc();
    SmallVector<Value> newShape(llvm::reverse(makeTensorDescOp.getShape()));
    SmallVector<Value> newStrides(llvm::reverse(makeTensorDescOp.getStrides()));

    Value newDesc = tt::MakeTensorDescOp::create(
        builder, loc, newDescType, makeTensorDescOp.getBase(), newShape,
        newStrides, makeTensorDescOp.getPadding());
    // Preserve discardable (non-inherent) attributes from the original op.
    for (auto attr : makeTensorDescOp->getDiscardableAttrs())
      newDesc.getDefiningOp()->setDiscardableAttr(attr.getName(),
                                                  attr.getValue());
    LLVM_DEBUG(llvm::dbgs() << "newMakeTensorDescOp:\n  " << newDesc << "\n");

    // Create a new DescriptorLoadOp with reversed indices.
    builder.setInsertionPoint(descLoadOp);
    SmallVector<Value> newIndices(llvm::reverse(descLoadOp.getIndices()));
    auto newDescLoadOp = tt::DescriptorLoadOp::create(
        builder, descLoadOp.getLoc(), tensorType, newDesc, newIndices,
        descLoadOp.getCache(), descLoadOp.getEvict());
    for (auto attr : descLoadOp->getDiscardableAttrs())
      newDescLoadOp->setDiscardableAttr(attr.getName(), attr.getValue());
    LLVM_DEBUG(llvm::dbgs() << "newDescLoadOp:\n  " << newDescLoadOp << "\n");

    // Propagate the new descriptor load result to the trans op sentinel.
    IRMapping mapping;
    propagateToUser(newDescLoadOp->getResult(0), descLoadOp.getResult(),
                    transOp, transOp, mapping);

    cleanUp.insert(descLoadOp);
    cleanUp.insert(makeTensorDescOp);
  }

  // Candidate is a transpose operation of the form:
  //   tt.dot(tt.trans(tt.load(...)))
  //   tt.dot(tt.trans(tt.descriptor_load(...)))
  // Where:
  //  - the transpose is a simple 2D transpose (order = [1, 0])
  //  - the transpose is not contained in a while loop
  //  - the transpose result is used by a dot operation
  //  - the transpose operation uses the result of a 2-dim load operation on a
  //    block pointer (transitively) defined by a `make_tensor_ptr` operation,
  //    or the result of a 2-dim descriptor load operation (transitively)
  //    defined by a `make_tensor_desc` operation.
  bool isCandidate(tt::TransOp transOp) const {
    assert(transOp && "Expecting a valid transpose operation");

    if (transOp->getParentOfType<scf::WhileOp>())
      return false;

    // Only handle simple 2D transposes.
    if (transOp.getOrder() != ArrayRef<int32_t>{1, 0})
      return false;

    // Check whether \p transOp is used by a `dotOp` (directly or indirectly).
    auto usedByDotOp = [](tt::TransOp transOp) {
      if (!transOp->hasOneUse())
        return false;

      Operation *user = *transOp->getUsers().begin();
      while (user) {
        if (isa<tt::DotOp>(user))
          return true;
        if (!user->hasOneUse())
          break;
        user = *user->getUsers().begin();
      }

      return false;
    };

    if (!usedByDotOp(transOp))
      return false;

    Operation *defOp = transOp.getSrc().getDefiningOp();
    if (!defOp)
      return false;

    if (auto loadOp = dyn_cast<tt::LoadOp>(defOp))
      return isCandidate(loadOp);
    if (auto descLoadOp = dyn_cast<tt::DescriptorLoadOp>(defOp))
      return isCandidate(descLoadOp);

    return false;
  }

  bool isCandidate(tt::LoadOp loadOp) const {
    if (!loadOp->hasOneUse())
      return false;

    Type ptrType = loadOp.getPtr().getType();
    if (!tt::isTensorPointerType(ptrType) ||
        cast<RankedTensorType>(cast<tt::PointerType>(ptrType).getPointeeType())
                .getRank() != 2)
      return false;

    std::optional<tt::MakeTensorPtrOp> makeTensorPtrOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorPtrOp>(loadOp.getPtr());
    return makeTensorPtrOp.has_value();
  }

  bool isCandidate(tt::DescriptorLoadOp descLoadOp) const {
    if (!descLoadOp->hasOneUse())
      return false;

    // Validate indices match the 2D descriptor block.
    if (descLoadOp.getIndices().size() != 2)
      return false;

    std::optional<tt::MakeTensorDescOp> makeTensorDescOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(
            descLoadOp.getDesc());
    if (!makeTensorDescOp)
      return false;

    tt::TensorDescType descTy = makeTensorDescOp->getResult().getType();
    auto tensorTy = cast<RankedTensorType>(descTy.getBlockType());
    if (tensorTy.getRank() != 2)
      return false;

    return true;
  }

  // If \p user is not \p sentinel, propagate \p newVal to \p user. Otherwise
  // terminate the propagation.
  // \p mapping is provided by the base class infrastructure for potential
  // future extensions (e.g., complex cloning scenarios).
  void propagateToUser(Value newVal, Value origVal, Operation *user,
                       Operation *sentinel, IRMapping &mapping) final override {
    assert(user && sentinel && "Expecting valid operations");
    assert(llvm::is_contained(origVal.getUsers(), user) && "Invalid usage");

    LLVM_DEBUG({
      llvm::dbgs() << "In " << __func__ << "\n";
      llvm::dbgs() << "user of:";
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
      SmallVector<Value> newOffsets(llvm::reverse(advanceOp.getOffsets()));
      auto newAdvanceOp = tt::AdvanceOp::create(rewriter, loc, newVal.getType(),
                                                newVal, newOffsets);
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "newAdvanceOp: " << newAdvanceOp << "\n");
      cleanUp.insert(advanceOp);
      return propagateToUsers(newAdvanceOp, advanceOp.getResult(), advanceOp,
                              sentinel, mapping);
    }

    if (auto loadOp = dyn_cast<tt::LoadOp>(user)) {
      OpBuilder rewriter(loadOp);
      auto newLoadOp = tt::LoadOp::create(
          rewriter, loc, newVal, loadOp.getMask(), loadOp.getOther(),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
      for (auto attr : loadOp->getDiscardableAttrs())
        newLoadOp->setDiscardableAttr(attr.getName(), attr.getValue());
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
      assert(opNum >= 0 && "origVal not found in yield operands");

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

struct TritonIntelFuseTranspose
    : tt::intel::impl::TritonIntelFuseTransposeBase<TritonIntelFuseTranspose> {
public:
  void runOnOperation() final override {
    ModuleOp moduleOp = getOperation();
    FuseTransposeWithLoad fuser;
    fuser.run(moduleOp);
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
