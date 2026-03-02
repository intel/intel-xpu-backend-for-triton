#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Utils/DefUseChain.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

#define DEBUG_TYPE "tritonintelgpu-optimize-dot-operands"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEDOTOPERANDS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// Transform:
//   %ptr = make_block_ptr [shapeN, shapeK], [strideN, strideK], [offN, offK]
//        : tt.ptr<tensor<NxK, enc>
//   %load = tt.load %ptr, {blockIO=<row_major|column_major>}
//         : tt.ptr<tensor<NxK, enc>
//   %trans = tt.trans %load : tt.ptr<tensor<KxN, dotEnc>>
//   tt.dot(%a, %trans)
// into:
//   %ptr = make_block_ptr [shapeK, shapeN], [strideK, strideN], [offK, offN]
//        : tt.ptr<tensor<KxN, dotEnc>
//   %load = tt.load %ptr, {blockIO=<column_major|row_major>}
//         : tt.ptr<tensor<KxN, dotEnc>
//   tt.dot(%a, %load)
class FuseTransWithLoad : public tt::intel::Fuser {
public:
  void run(ModuleOp moduleOp) {
    // Collect def-use chains originating at a `MakeTensorPtrOp` operation
    // and terminating at a candidate `tt::TransOp` operation.
    // Note: A candidate `TransOp` must use the result of a `LoadOp` using a ptr
    // created the `MakeTensorPtrOp` rooting the def-use chain.
    DefUseChainManager manager;
    moduleOp.walk([&](tt::TransOp transOp) {
      if (isCandidate(transOp)) {
        auto loadOp = cast<tt::LoadOp>(transOp.getSrc().getDefiningOp());
        auto makeTensorPtrOp =
            *tt::intel::findDefiningOpOfType<tt::MakeTensorPtrOp>(
                loadOp.getPtr());
        manager.createChains(makeTensorPtrOp, transOp);
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

    // Fuse tt.LoadOp->tt.TransOp operations.
    Fuser::fuse(manager.getChains());

    // Remove operations that are no longer used.
    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);
  }

private:
  void fuse(const DefUseChain &chain) {
    assert(
        isa<tt::MakeTensorPtrOp>(chain.getStart()) &&
        "Expecting 'chain' to be rooted by a 'tt.make_tensor_ptr' operation");
    assert(isa<tt::TransOp>(chain.getEnd()) &&
           "Expecting 'chain' to be terminated by a 'tt.trans' operation");

    auto makeTensorPtrOp = cast<tt::MakeTensorPtrOp>(chain.getStart());
    auto transOp = cast<tt::TransOp>(chain.getEnd());
    auto loadOp = cast<tt::LoadOp>(transOp.getSrc().getDefiningOp());
    LLVM_DEBUG(llvm::dbgs()
               << "Fusing:\n  " << transOp << "\nwith:\n  " << loadOp << "\n");

    // Create a MakeTensorPtrOp yielding a block pointer to the transposed
    // tensor...
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

    // ... and propagate it through the def-use chain.
    IRMapping mapping;
    propagateToUsers(ptr, chain, mapping);
    cleanUp.insert(makeTensorPtrOp);
  }

  // Candidate is of the form:
  //   tt.dot(tt.trans(tt.load(..., {blockIO=...})))
  // Where:
  //  - the transpose is not contained in a while loop
  //  - the transpose result is used by the dot operation, and
  //  - the transpose operation uses the result of a 2-dim load operation on a
  //    block pointer (transitively) defined by a `make_tensor_ptr` operation.
  bool isCandidate(tt::TransOp transOp) const {
    assert(transOp && "Expecting a valid transpose operation");

    if (transOp->getParentOfType<scf::WhileOp>())
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

    Attribute transOpEncoding = transOp.getType().getEncoding();
    if (!usedByDotOp(transOp) || !transOpEncoding ||
        !isa<ttg::DotOperandEncodingAttr>(transOpEncoding))
      return false;

    Operation *defOp = transOp.getSrc().getDefiningOp();
    if (!defOp || !isa<tt::LoadOp>(defOp))
      return false;

    auto loadOp = cast<tt::LoadOp>(defOp);
    bool loadOpHasBlockIOAttr = loadOp->hasAttrOfType<StringAttr>(
        ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
    if (!loadOp->hasOneUse() || !loadOpHasBlockIOAttr)
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

  // Determine whether all operations in the given def-use chain have a single
  // user.
  // Note: we allow an operation in the def-use chain to have an additional user
  // if the operation is in a for loop, and the additional user is the loop
  // yield operation, provided that the result yielded is not used after the
  // loop.
  // Example:
  //   make_tensor_ptr -> advance -> load (OK)
  //   make_tensor_ptr -> for init_arg -> advance -> load (OK)
  //                                   -> yield (OK)
  //   make_tensor_ptr -> for init_arg -> advance -> load (OK)
  //                                              -> yield -> load (NOT OK)
  //
  bool validateChain(const DefUseChain &chain) const {
    auto validateOperation = [](Operation *op, Operation *&nextOp) {
      assert(nextOp == nullptr);
      if (op->hasOneUse())
        return true;
      if (!op->getParentOfType<LoopLikeOpInterface>())
        return false;

      auto loopOp = op->getParentOfType<LoopLikeOpInterface>();
      auto yieldOp = cast<scf::YieldOp>(
          loopOp.getYieldedValues()[0].getParentBlock()->getTerminator());

      SmallVector<Operation *> users(op->getUsers());
      if (users.size() > 2 || llvm::none_of(users, [&](Operation *user) {
            return user == yieldOp;
          }))
        return false;

      auto yieldedValUsedAfterLoop = [&op, &yieldOp]() {
        auto it =
            llvm::find_if(yieldOp->getOpOperands(), [&op](OpOperand &operand) {
              return operand.get() == op->getResult(0);
            });
        assert(it != yieldOp->getOpOperands().end());
        OpOperand &operand = *it;
        auto loopOp = cast<LoopLikeOpInterface>(yieldOp->getParentOp());
        OpResult res = loopOp->getResult(operand.getOperandNumber());
        return !res.getUsers().empty();
      };

      if (yieldedValUsedAfterLoop())
        return false;

      nextOp = *llvm::find_if(
          users, [](Operation *user) { return !isa<scf::YieldOp>(user); });
      return true;
    };

    Operation *currentOp = chain.getStart();
    while (currentOp != chain.getEnd()) {
      Operation *user = nullptr;
      if (!validateOperation(currentOp, user)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Fails safety checks: " << *currentOp << "\n");
        return false;
      }

      user = (!user) ? user = *currentOp->getUsers().begin() : user;
      if (user->getNumRegions() == 0) {
        currentOp = user;
        continue;
      }

      // Current limitation: give up if the use is a branch.
      if (isa<scf::IfOp>(user))
        return false;

      [[maybe_unused]] Operation *oldCurrentOp = currentOp;

      // Find the next operation in the def-use chain inside the loop body.
      if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
        for (auto [arg, init] :
             llvm::zip(loopOp.getRegionIterArgs(), loopOp.getInits())) {
          if (init == currentOp->getResult(0)) {
            if (!arg.hasOneUse())
              return false;

            currentOp = *arg.getUsers().begin();
            break;
          }
        }
      }

      assert(currentOp != oldCurrentOp && "Infinite loop detected!");
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
          rewriter, loadOp.getLoc(), newVal, loadOp.getMask(),
          loadOp.getOther(), loadOp.getBoundaryCheckAttr(),
          loadOp.getPaddingAttr(), loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());

      StringRef blockIOAttrName =
          ttgi::TritonIntelGPUDialect::getBlockIOAttrName();
      StringAttr attr = loadOp->getAttrOfType<StringAttr>(blockIOAttrName);
      StringAttr newAttr =
          (attr == "row_major")
              ? StringAttr::get(loadOp->getContext(), "column_major")
          : (attr == "column_major")
              ? StringAttr::get(loadOp->getContext(), "row_major")
              : nullptr;
      assert(newAttr && "Expecting a valid blockIO attribute");

      newLoadOp->setAttrs(loadOp->getAttrs());
      newLoadOp->setAttr(blockIOAttrName, newAttr);
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

// Fuse tt.trans into tt.descriptor_load for dot operands.
//
// The descriptor is always row-major (stride-1 on the last dimension) and is
// never modified. Instead, the DescriptorLoadOp is replaced with one whose
// result type has transposed dimensions. The "column_major" block_io attribute
// signals to the lowering that the result dimensions are transposed relative
// to the descriptor's block shape, so it must swap indices and request a
// hardware-transposed 2D block load.
//
// Transform:
//   %desc = tt.make_tensor_desc %base, [%N, %K], [%K_stride, %1]
//         : <tensor<BNxBKxf16>>
//   %load = tt.descriptor_load %desc[%n, %k] : tensor<BNxBKxf16>
//   %trans = tt.trans %load : tensor<BKxBN, blocked_trans>
//   %cvt = ttg.convert_layout %trans : tensor<BKxBN, dotEnc>
//   tt.dot(%a, %cvt)
// into:
//   %desc = tt.make_tensor_desc %base, [%N, %K], [%K_stride, %1]
//         : <tensor<BNxBKxf16>> (unchanged)
//   %load = tt.descriptor_load %desc[%n, %k] {block_io = "column_major"}
//         : !tt.tensordesc<tensor<BNxBKxf16>> -> tensor<BKxBN, dotEnc>
//   tt.dot(%a, %load)
class FuseTransWithDescriptorLoad {
public:
  void run(ModuleOp moduleOp) {
    moduleOp.walk([&](tt::TransOp transOp) {
      FusionCandidate candidate;
      if (isCandidate(transOp, candidate))
        fuse(transOp, candidate);
    });
    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);
  }

private:
  struct FusionCandidate {
    Attribute targetEncoding;
    Operation *lastOpBeforeDot;
  };

  SmallPtrSet<Operation *, 8> cleanUp;

  bool isCandidate(tt::TransOp transOp, FusionCandidate &out) const {
    assert(transOp && "Expecting a valid transpose operation");

    if (transOp->getParentOfType<scf::WhileOp>())
      return false;

    if (!transOp->hasOneUse())
      return false;

    // Walk through ConvertLayoutOps to find DotOp and the target encoding.
    Operation *user = *transOp->getUsers().begin();
    Attribute targetEncoding = transOp.getType().getEncoding();
    Operation *lastOp = transOp;

    while (auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(user)) {
      if (!cvtOp->hasOneUse())
        return false;
      targetEncoding = cvtOp.getType().getEncoding();
      lastOp = cvtOp;
      user = *cvtOp->getUsers().begin();
    }

    if (!isa<tt::DotOp, tt::DotScaledOp>(user))
      return false;

    if (!targetEncoding || !isa<ttg::DotOperandEncodingAttr>(targetEncoding))
      return false;

    // Source must be DescriptorLoadOp with single use and rank 2.
    auto descLoadOp = dyn_cast_or_null<tt::DescriptorLoadOp>(
        transOp.getSrc().getDefiningOp());
    if (!descLoadOp || !descLoadOp->hasOneUse())
      return false;

    if (cast<RankedTensorType>(descLoadOp.getType()).getRank() != 2)
      return false;

    // Must be able to find the defining MakeTensorDescOp.
    auto makeTensorDescOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(
            descLoadOp.getDesc());
    if (!makeTensorDescOp.has_value())
      return false;

    out.targetEncoding = targetEncoding;
    out.lastOpBeforeDot = lastOp;
    return true;
  }

  void fuse(tt::TransOp transOp, const FusionCandidate &candidate) {
    auto descLoadOp =
        cast<tt::DescriptorLoadOp>(transOp.getSrc().getDefiningOp());

    // Collect ops to clean up before modifying IR.
    SmallVector<Operation *> opsToClean{transOp};
    Operation *cur = *transOp->getUsers().begin();
    while (cur != candidate.lastOpBeforeDot) {
      opsToClean.push_back(cur);
      cur = *cur->getUsers().begin();
    }
    if (candidate.lastOpBeforeDot != transOp)
      opsToClean.push_back(candidate.lastOpBeforeDot);

    // Keep the original descriptor — do NOT reverse it.
    // The descriptor is always row-major (stride-1 on last dim).
    auto descType = cast<tt::TensorDescType>(descLoadOp.getDesc().getType());
    RankedTensorType blockType = descType.getBlockType();
    SmallVector<int64_t> transposedShape(llvm::reverse(blockType.getShape()));

    // Create new DescriptorLoadOp with transposed result type + target
    // encoding. The verifier allows result shape to differ from descriptor
    // block shape as long as the total element count matches.
    OpBuilder builder(descLoadOp);
    auto newResultType = RankedTensorType::get(
        transposedShape, blockType.getElementType(), candidate.targetEncoding);
    auto newLoad = tt::DescriptorLoadOp::create(
        builder, descLoadOp.getLoc(), newResultType, descLoadOp.getDesc(),
        descLoadOp.getIndices(), descLoadOp.getCache(), descLoadOp.getEvict());

    // Copy any discardable attributes from the original load.
    for (auto attr : descLoadOp->getDiscardableAttrs())
      newLoad->setDiscardableAttr(attr.getName(), attr.getValue());

    // Set block_io = "column_major": signals that the result type dimensions
    // are transposed relative to the descriptor's block shape dimensions.
    newLoad->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                     StringAttr::get(transOp.getContext(), "column_major"));

    // Replace uses and schedule cleanup.
    candidate.lastOpBeforeDot->replaceAllUsesWith(
        ValueRange{newLoad.getResult()});

    for (Operation *op : opsToClean)
      cleanUp.insert(op);
    cleanUp.insert(descLoadOp);
    // Do NOT clean up MakeTensorDescOp — it's unchanged, may have other uses
  }
};

} // namespace

class TritonIntelGPUOptimizeDotOperandsPass
    : public ttgi::impl::TritonIntelGPUOptimizeDotOperandsBase<
          TritonIntelGPUOptimizeDotOperandsPass> {

public:
  using ttgi::impl::TritonIntelGPUOptimizeDotOperandsBase<
      TritonIntelGPUOptimizeDotOperandsPass>::
      TritonIntelGPUOptimizeDotOperandsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    FuseTransWithLoad fuser;
    fuser.run(moduleOp);
    FuseTransWithDescriptorLoad descFuser;
    descFuser.run(moduleOp);
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};
