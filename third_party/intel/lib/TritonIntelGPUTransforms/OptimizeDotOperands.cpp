#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LayoutUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
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
//   %ptr = make_block_ptr [shX, shX], [stX, stY], [offX, offY]
//        : tt.ptr<tensor<MxN, enc>
//   %load = tt.load %ptr, {blockIO=<row_major|column_major>}
//         : tt.ptr<tensor<MxN, encoding>
//   %trans = tt.trans %load : tt.ptr<tensor<NxM, dotEnc>>
//   tt.dot(%a, %trans)
// into:
//   %ptr = make_block_ptr [shX, shX], [stX, stY], [offX, offY]
//        : tt.ptr<tensor<NxM, dotEnc>
//   %load = tt.load %ptr, {blockIO=<column_major|row_major>}
//         : tt.ptr<tensor<NxM, dotEnc>
//   tt.dot(%a, %load)
class FuseTransWithLoad {
private:
  tt::FuncOp funcOp;
  SmallPtrSet<Operation *, 8> cleanUp;

public:
  FuseTransWithLoad(tt::FuncOp funcOp) : funcOp(funcOp) {}

  void run() {
    funcOp.walk([&](tt::TransOp transOp) {
      if (isCandidate(transOp))
        fuse(transOp);
    });

    if (!cleanUp.empty())
      finalize();

    [[maybe_unused]] auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

  void fuse(tt::TransOp transOp) {
    LLVM_DEBUG(llvm::dbgs() << "Found candidate:\n\t" << transOp << "\n");
    auto loadOp = cast<tt::LoadOp>(transOp.getSrc().getDefiningOp());
    tt::MakeTensorPtrOp makeTensorPtrOp =
        *triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
    LLVM_DEBUG(llvm::dbgs()
               << "makeTensorPtrOp:\n\t" << makeTensorPtrOp << "\n");

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
    Value ptr = builder.create<tt::MakeTensorPtrOp>(
        makeTensorPtrOp.getLoc(), newPtrType, makeTensorPtrOp.getBase(),
        newShape, newStrides, newOffsets, makeTensorPtrOp.getOrderAttr());
    assert(makeTensorPtrOp->hasOneUse() && "Expecting single user");
    LLVM_DEBUG(llvm::dbgs() << "newMakeTensorPtrOp:\n\t" << ptr << "\n");

    // ... and propagate it through the def-use chain.
    propagateToUsers(ptr, makeTensorPtrOp, makeTensorPtrOp, transOp);
  }

private:
  // Candidate is of the form:
  //   tt.dot(tt.trans(tt.load(..., {blockIO=...})))
  // Where:
  //  - the transpose result is used only by the dot operation, and
  //  - the transpose operation uses the result of a 2-dim load operation on a
  //    block pointer (transitively) defined by a `make_tensor_ptr` in the same
  //    function, and
  //  - each operation in the def-use chain origination at the `make_tensor_ptr`
  //    and terminating at the load has a single user.
  bool isCandidate(tt::TransOp transOp) const {
    assert(transOp && "Expecting a valid transpose operation");

    bool transOpUsedOnlyByDotOp =
        transOp->hasOneUse() &&
        isa<triton::DotOp>(*transOp->getUsers().begin());
    Attribute transOpEncoding = transOp.getType().getEncoding();
    if (!transOpUsedOnlyByDotOp || !transOpEncoding ||
        !isa<ttg::DotOperandEncodingAttr>(transOpEncoding))
      return false;

    Operation *defOp = transOp.getSrc().getDefiningOp();
    if (!defOp || !isa<tt::LoadOp>(defOp))
      return false;

    return isCandidate(cast<tt::LoadOp>(defOp));
  }

  bool isCandidate(tt::LoadOp loadOp) const {
    assert(loadOp && "Expecting a valid load operation");

    bool loadOpHasBlockIOAttr = loadOp->hasAttrOfType<StringAttr>(
        ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
    if (!loadOp->hasOneUse() || !loadOpHasBlockIOAttr)
      return false;

    auto ptrType = cast<tt::PointerType>(loadOp.getPtr().getType());
    if (!isTensorPointerType(ptrType) ||
        cast<RankedTensorType>(ptrType.getPointeeType()).getRank() != 2)
      return false;

    std::optional<tt::MakeTensorPtrOp> defOp =
        triton::intel::findDefiningMakeTensorPtrOp(loadOp.getPtr());
    if (!defOp || !singleUsersInChain(*defOp, loadOp))
      return false;

    return true;
  }

  // Determine whether all operations in the def-use chain from \p start to
  // \p end have a single user.
  // Note: we allow an operation in the def-use chain to have an additional user
  // if the operation is in a for loop, and the additional user is the yield
  // operation, provided that the result yielded is not used after the loop.
  // Example:
  //   make_tensor_ptr -> advance -> load (OK)
  //   make_tensor_ptr -> for init_arg -> advance -> load (OK)
  //                                   -> yield (OK)
  //   make_tensor_ptr -> for init_arg -> advance -> load (OK)
  //                                              -> yield -> load (NOT OK)
  //
  bool singleUsersInChain(Operation *start, Operation *end) const {
    assert(start && end && "Expecting valid operations");
    Operation *currentOp = start;

    auto validate = [](Operation *op, Operation *&nextOp) {
      assert(nextOp == nullptr);

      if (op->hasOneUse())
        return true;
      if (!op->getParentOfType<scf::ForOp>())
        return false;

      SmallVector<Operation *> users(op->getUsers());
      if (users.size() > 2 || llvm::none_of(users, [](Operation *op) {
            return isa<scf::YieldOp>(op);
          }))
        return false;

      auto yieldOp = cast<scf::YieldOp>(*llvm::find_if(
          users, [](Operation *user) { return isa<scf::YieldOp>(user); }));
      auto yieldedValUsedAfterLoop = [&op, &yieldOp]() {
        auto it =
            llvm::find_if(yieldOp->getOpOperands(), [&op](OpOperand &operand) {
              return operand.get() == op->getResult(0);
            });
        assert(it != yieldOp->getOpOperands().end());
        OpOperand &operand = *it;
        auto forOp = cast<scf::ForOp>(yieldOp->getParentOp());
        OpResult res = forOp->getResult(operand.getOperandNumber());
        return !res.getUsers().empty();
      };
      if (yieldedValUsedAfterLoop())
        return false;

      nextOp = *llvm::find_if(
          users, [](Operation *user) { return !isa<scf::YieldOp>(user); });
      return true;
    };

    while (currentOp != end) {
      Operation *user = nullptr;
      if (!validate(currentOp, user)) {
        LLVM_DEBUG(llvm::dbgs() << *currentOp << " fails safety checks\n");
        return false;
      }

      user = (!user) ? user = *currentOp->getUsers().begin() : user;
      if (user->getNumRegions() == 0) {
        currentOp = user;
        continue;
      }

      // Find the next operation in the def-use chain inside the loop body.
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (BlockArgument arg : forOp.getRegionIterArgs()) {
          Value initArg = forOp.getInitArgs()[arg.getArgNumber() - 1];
          if (initArg == currentOp->getResult(0)) {
            if (!arg.hasOneUse())
              return false;

            currentOp = *arg.getUsers().begin();
            break;
          }
        }
      }
    }

    return true;
  }

  // Recursively update the operands in a chain of AdvanceOps, after setting the
  // pointer operand of the first one.
  tt::AdvanceOp updateAdvanceOpChain(tt::AdvanceOp advanceOp, tt::LoadOp loadOp,
                                     Value ptr) const {
    assert(advanceOp->hasOneUse() && "Expecting single user");
    assert(tt::isTensorPointerType(ptr.getType()) &&
           "Expecting a block pointer");

    Operation *user = *advanceOp->getUsers().begin();
    if (auto loadUser = dyn_cast<tt::LoadOp>(user)) {
      assert(loadUser == loadOp &&
             "chain should be terminated by candidate load");
      OpBuilder rewriter(advanceOp);
      SmallVector<Value> newOffsets(llvm::reverse(advanceOp.getOffsets()));
      return rewriter.create<tt::AdvanceOp>(advanceOp.getLoc(), ptr.getType(),
                                            ptr, newOffsets);
    }

    if (auto advanceOp = dyn_cast<tt::AdvanceOp>(user)) {
      OpBuilder rewriter(advanceOp);
      SmallVector<Value> newOffsets(llvm::reverse(advanceOp.getOffsets()));
      ptr = rewriter.create<tt::AdvanceOp>(advanceOp.getLoc(), ptr.getType(),
                                           ptr, newOffsets);
      return updateAdvanceOpChain(advanceOp, loadOp, ptr);
    }

    // TODO: add support for loops (advanceOp cound be consumed by a loop
    // init_arg).

    llvm_unreachable("Unexpected user");
    return nullptr;
  }

  // Propagate \p newVal to users of \p origOp.
  void propagateToUsers(Value newVal, Value origVal, Operation *origOp,
                        Operation *sentinel) {
    assert(origOp && sentinel && "Expecting valid operations");
    const SmallVector<Operation *> users(origOp->getUsers());
    for (Operation *user : users)
      propagateToUser(newVal, origVal, user, sentinel);
  }

  // If \p user is not \p sentinel, propagate \p newVal to \p user. Otherwise
  // terminate the propagation.
  void propagateToUser(Value newVal, Value origVal, Operation *user,
                       Operation *sentinel) {
    assert(user && sentinel && "Expecting valid operations");
    assert(llvm::is_contained(origVal.getUsers(), user) && "Invalid usage");

    LLVM_DEBUG({
      llvm::dbgs() << "In " << __func__ << "\n";
      llvm::dbgs() << "user of ";
      if (origVal.getDefiningOp()) {
        llvm::dbgs() << "\n\t" << *origVal.getDefiningOp() << "\n";
      } else {
        origVal.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << " ";
      }
      llvm::dbgs() << "is:\n\t";
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
      auto newAdvanceOp = rewriter.create<tt::AdvanceOp>(loc, newVal.getType(),
                                                         newVal, newOffsets);
      LLVM_DEBUG(llvm::dbgs() << "\tnewAdvanceOp: " << newAdvanceOp << "\n");
      cleanUp.insert(advanceOp);
      return propagateToUsers(newAdvanceOp, advanceOp.getResult(), advanceOp,
                              sentinel);
    }

    if (auto loadOp = dyn_cast<tt::LoadOp>(user)) {
      OpBuilder rewriter(loadOp);
      auto newLoadOp = rewriter.create<tt::LoadOp>(
          loadOp.getLoc(), newVal, loadOp.getMask(), loadOp.getOther(),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());

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

      newLoadOp->setAttr(blockIOAttrName, newAttr);
      LLVM_DEBUG(llvm::dbgs() << "\tnewLoadOp: " << newLoadOp << "\n");
      cleanUp.insert(loadOp);
      return propagateToUsers(newLoadOp, loadOp.getResult(), loadOp, sentinel);
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
      for (OpResult res : parentOp->getOpResults()) {
        int resNum = res.getResultNumber();
        if (resNum == opNum)
          res.setType(newVal.getType());
      }
      return;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user))
      return propagateToLoop(newVal, origVal, forOp, sentinel);
  }

  void propagateToLoop(Value newVal, Value origVal, LoopLikeOpInterface loopOp,
                       Operation *sentinel) {
    assert(sentinel && sentinel != loopOp && "Unexpected sentinel kind");
    LLVM_DEBUG({
      llvm::dbgs() << "In " << __func__ << "\n";
      llvm::dbgs() << "newVal: " << newVal << "\n";
    });

    for (auto [initArg, rgnInitArg, yieldVal, loopRes] :
         llvm::zip(loopOp.getInitsMutable(), loopOp.getRegionIterArgs(),
                   loopOp.getYieldedValues(), loopOp->getResults())) {
      if (initArg.get() == origVal) {
        initArg.set(newVal);
        rgnInitArg.setType(initArg.get().getType());
        const SmallVector<Operation *> users(rgnInitArg.getUsers());
        for (Operation *user : users)
          propagateToUser(rgnInitArg, rgnInitArg, user, sentinel);
      }
    }
  }

  // Cleanup unused operations.
  void finalize() {
    bool erasedOperation;
    do {
      erasedOperation = false;
      SmallPtrSet<Operation *, 8> erased;
      for (Operation *op : cleanUp) {
        if (!op->getUsers().empty() || !op->getRegions().empty())
          continue;

        erased.insert(op);
        op->erase();
        erasedOperation = true;
      }
      cleanUp.remove_if([&](Operation *op) { return erased.contains(op); });
    } while (erasedOperation);

    // Remove operations that contain a region.
    for (Operation *op : cleanUp) {
      if (!op->getUsers().empty())
        continue;
      op->erase();
    }
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
    moduleOp.walk([](tt::FuncOp funcOp) {
      FuseTransWithLoad fuser(funcOp);
      fuser.run();
    });
  }
};
