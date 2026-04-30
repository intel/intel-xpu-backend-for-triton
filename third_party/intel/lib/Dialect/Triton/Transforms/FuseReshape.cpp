#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/DefUseChain.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
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
//   %desc = tt.make_tensor_descriptor %base, [%s0*%d+%s1,%s2], [%b,%c]
//                       : !tt.tensordesc<512x64xf16>
//   %A = tt.descriptor_load %desc[%x*%d+%y,%z] -> tensor<512x64xf16>
//   dot %A, ... : tensor<512x64xf16> x tensor<64x32xf16> -> tensor<512x32xf16>
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
                  *tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(
                      descLoadOp.getDesc());
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
    auto descType = cast<tt::TensorDescType>(makeTensorDescOp.getType());
    [[maybe_unused]] ArrayRef<int64_t> resShape =
        cast<RankedTensorType>(descType.getBlockType()).getShape();
    assert(resShape[0] == 1 && "Result shape should have extent equal to 1 in "
                               "the outermost dimension");

    auto tensorType = cast<RankedTensorType>(reshapeOp.getType());
    auto newDescType = tt::TensorDescType::get(
        tensorType.getShape(), tensorType.getElementType(), mlir::Attribute{});

    OpBuilder builder(makeTensorDescOp);
    Location loc = makeTensorDescOp.getLoc();
    OperandRange shapes = makeTensorDescOp.getShape();
    OperandRange strides = makeTensorDescOp.getStrides();

    // Collapse the 3-dim tensor into a 2-dim tensor.
    // Given a make_tensor_descriptor with:
    //   shape  [s0, s1, s2]
    //   stride [a, b, c]
    // Create a make_tensor_descriptor with:
    //   shape  [s0 * a / b + s1, s2]
    //   stride [b, c]
    SmallVector<Value> newShape(makeTensorDescOp.getShape().drop_front());
    SmallVector<Value> newStrides(makeTensorDescOp.getStrides().drop_front());

    const unsigned innermostDimIdx = shapes.size() - 1;
    const unsigned newInnermostDimIdx = (innermostDimIdx - 1);
    const unsigned newOutermostDimIdx = !newInnermostDimIdx;
    auto div = arith::DivUIOp::create(builder, loc, strides[0],
                                      newStrides[newOutermostDimIdx]);
    auto trunc =
        builder.createOrFold<arith::TruncIOp>(loc, shapes[0].getType(), div);

    newShape[newOutermostDimIdx] = arith::AddIOp::create(
        builder, loc, arith::MulIOp::create(builder, loc, shapes[0], trunc),
        newShape[newOutermostDimIdx]);

    Value newDesc = tt::MakeTensorDescOp::create(
        builder, loc, newDescType, makeTensorDescOp.getBase(), newShape,
        newStrides, makeTensorDescOp.getPadding());
    LLVM_DEBUG(llvm::dbgs() << "new MakeTensorDescOp:\n  " << newDesc << "\n");

    // Adjust the descriptor load operation indices.
    // Given a make_tensor_descriptor with shape/strides:
    //   shape  [s0, s1, s2]
    //   stride [a, b, c]
    // And a descriptor_load with offsets:
    //   offset [x, y, z]
    // Create a new descriptor_load operation with indices:
    //   offset [x * a / b + y, z]
    builder.setInsertionPoint(descLoadOp);
    OperandRange offsets = descLoadOp.getIndices();
    SmallVector<Value> newOffsets(offsets.drop_front());
    newOffsets[newOutermostDimIdx] = arith::AddIOp::create(
        builder, loc, arith::MulIOp::create(builder, loc, offsets[0], trunc),
        newOffsets[newOutermostDimIdx]);

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
  //  - the reshape operation drops the outermost dimension of the operand,
  //    which is a 3-dim tensor with outermost dimension extent equal to one
  //  - the reshape result is used by a dot operation
  //  - the reshape operation uses the result of a 3-dim load operation on a
  //    tensor descriptor (transitively) defined by a `make_tensor_descriptor`
  //  - the tensor descriptor refers to a tensor that has extent
  //    equal to 1 on the outermost dimension
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
        tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(
            descLoadOp.getDesc());
    if (!makeTensorDescOp)
      return false;

    tt::TensorDescType descTy = makeTensorDescOp->getResult().getType();
    auto tensorTy = cast<RankedTensorType>(descTy.getBlockType());
    assert((tensorTy.getRank() == 3 && tensorTy.getDimSize(0) == 1) &&
           "Unexpected tensor type");

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
