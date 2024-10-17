#include "intel/include/Analysis/AxisInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Utils.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tritonintelgpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUCOALESCE
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace {

RankedTensorType getRankedTensorType(Type ptrTy) {
  return tt::isTensorPointerType(ptrTy)
             ? cast<RankedTensorType>(
                   cast<tt::PointerType>(ptrTy).getPointeeType())
             : dyn_cast<RankedTensorType>(ptrTy);
}

struct CoalescePass
    : public ttgi::impl::TritonIntelGPUCoalesceBase<CoalescePass> {
private:
  void
  setCoalescedEncoding(tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                       Operation *op, int numWarps, int threadsPerWarp,
                       llvm::MapVector<Operation *, Attribute> &layoutMap) {
    Value ptr = getMemAccessPtr(op);
    LDBG("ptr: " << ptr);

    LDBG("Considering op: " << *op);
    LLVM_DEBUG({
      DBGS() << "axis info of pointer: ";
      axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
    SmallVector<unsigned> order = argSort(contiguity);
    LDBG("order=[" << triton::join(order, ", ") << "]");

    RankedTensorType refTensorType = getRankedTensorType(ptr.getType());
    auto matchesShape = [&refTensorType](const Value &val) {
      auto rttType = dyn_cast<RankedTensorType>(val.getType());
      return rttType && rttType.getShape() == refTensorType.getShape();
    };

    // The desired divisibility is the maximum divisibility among all dependent
    // pointers which have the same shape and order as `ptr`.
    llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
    memAccessesSameOrder.insert(op);
    if (ptr.getDefiningOp()) {
      for (Operation *use : mlir::multiRootGetSlice(op)) {
        Value val = getMemAccessPtr(use);
        if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use))
          continue;
        auto currOrder =
            argSort(axisInfoAnalysis.getAxisInfo(val)->getContiguity());
        if (order == currOrder) {
          LDBG("multi-root-slice: insert to memAccessesSameOrder " << *use);
          memAccessesSameOrder.insert(use);
        }
      }
    }

    auto shapePerCTA = ttg::getShapePerCTA(refTensorType);
    LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;
    unsigned perThread =
        ttgi::getNumElementsPerThread(op, order, axisInfoAnalysis);
    LDBG("perThread for op: " << perThread);

    for (Operation *opSameOrder : memAccessesSameOrder) {
      if (opSameOrder == op)
        continue;
      unsigned currPerThread =
          ttgi::getNumElementsPerThread(opSameOrder, order, axisInfoAnalysis);
      LDBG("perThread for opSameOrder: " << currPerThread);
      perThread = std::max(perThread, currPerThread);
    }

    perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
    LDBG("perThread: " << perThread);

    if (perThread <= 1)
      return;

    if (!dyn_cast<triton::LoadOp>(op)) {
      // For ops that can result in a global memory write, we should enforce
      // that each thread handles at most 128 bits, which is the widest
      // available vectorized store op; otherwise, the store will have "gaps"
      // in the memory write at the warp level, resulting in worse performance.
      // For loads, we can expect that the gaps won't matter due to the L1
      // cache.
      perThread = std::min<int>(perThread, ttgi::getNumElementsPerThread(
                                               op, order, axisInfoAnalysis));
    }
    SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
    sizePerThread[order[0]] = perThread;

    auto CTALayout = ttg::getCTALayout(refTensorType.getEncoding());
    layoutMap[op] = ttg::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  static RankedTensorType getNewType(RankedTensorType tensorType,
                                     Attribute encoding) {
    return RankedTensorType::get(tensorType.getShape(),
                                 tensorType.getElementType(), encoding);
  }

  // Find the defining makeTensorPtrOp operation of the given value.
  static std::optional<tt::MakeTensorPtrOp>
  findDefiningMakeTensorPtrOp(Value val) {
    if (auto arg = dyn_cast<BlockArgument>(val)) {
      Operation *parentOp = val.getParentBlock()->getParentOp();
      assert(isa<scf::ForOp>(parentOp) && "Expected a scf::ForOp");
      auto loopArg =
          cast<scf::ForOp>(parentOp).getInitArgs()[arg.getArgNumber() - 1];
      return findDefiningMakeTensorPtrOp(loopArg);
    }

    if (auto advanceOp = val.getDefiningOp<tt::AdvanceOp>())
      return findDefiningMakeTensorPtrOp(advanceOp.getPtr());
    if (auto makePtrOp = val.getDefiningOp<tt::MakeTensorPtrOp>())
      return makePtrOp;

    return std::nullopt;
  }

  static bool filterUser(Operation *op) {
    // Yield operations trigger updating the layout of the containing loop
    // results, so don't skip them.
    if (isa<scf::YieldOp>(op))
      return false;

    // Skip operations that don't yield a result and contain no regions.
    if (op->getNumResults() == 0 && op->getNumRegions() == 0)
      return true;

    // Operations that do not yield a block pointer aren't interesting.
    if (op->getNumRegions() == 0 &&
        llvm::none_of(op->getResultTypes(), [](Type resType) {
          return tt::isTensorPointerType(resType);
        }))
      return true;

    return false;
  }

  // Propagate the \p root block argument operation output layout along the
  // def-use chain.
  static void propagateLayout(BlockArgument arg, Attribute layout,
                              IRRewriter &rewriter) {
    llvm::errs() << "arg: " << arg << "\n";
    for (Operation *user : arg.getUsers()) {
      llvm::errs() << "user: " << *user << "\n\n";
      if (filterUser(user)) {
        llvm::errs() << "SKIP\n";
        continue;
      }

      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        // Modify and propagate the result of the enclosing loop.
        auto forOp = yieldOp->getParentOfType<scf::ForOp>();

        rewriter.modifyOpInPlace(forOp, [&]() {
          for (auto [opType, res] :
               llvm::zip(yieldOp->getOperandTypes(), forOp.getResults())) {
            if (opType == res.getType())
              continue;

            assert(tt::isTensorPointerType(res.getType()) &&
                   tt::isTensorPointerType(opType) &&
                   "Expecting blocked pointers");
            assert(cast<RankedTensorType>(
                       cast<tt::PointerType>(opType).getPointeeType())
                           .getEncoding() == layout &&
                   "Unexpected layout");

            auto resType = cast<tt::PointerType>(res.getType());
            RankedTensorType tensorType = getRankedTensorType(resType);
            res.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                             resType.getAddressSpace()));
          }
        });

        propagateLayout(forOp, layout, rewriter);
        continue;
      }

      changeAndPropagateLayout(user, layout, rewriter);
    }
  }

  static void propagateLayout(Operation *root, Attribute layout,
                              IRRewriter &rewriter) {
    assert(root && root->getNumResults() != 0 &&
           "Expecting an operation yielding a result");

    //    llvm::errs() << "root: " << *root << "\n\n";
    for (Operation *user : root->getUsers()) {
      llvm::errs() << "user: " << *user << "\n\n";
      if (filterUser(user)) {
        llvm::errs() << "SKIP\n";
        continue;
      }

      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        // Modify and propagate the result of the enclosing loop.
        auto forOp = yieldOp->getParentOfType<scf::ForOp>();

        rewriter.modifyOpInPlace(forOp, [&]() {
          for (auto [opType, res] :
               llvm::zip(yieldOp->getOperandTypes(), forOp.getResults())) {
            if (opType == res.getType())
              continue;

            assert(tt::isTensorPointerType(res.getType()) &&
                   tt::isTensorPointerType(opType) &&
                   "Expecting blocked pointers");
            assert(cast<RankedTensorType>(
                       cast<tt::PointerType>(opType).getPointeeType())
                           .getEncoding() == layout &&
                   "Unexpected layout");

            auto resType = cast<tt::PointerType>(res.getType());
            RankedTensorType tensorType = getRankedTensorType(resType);
            res.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                             resType.getAddressSpace()));
          }
        });

        propagateLayout(forOp, layout, rewriter);
        continue;
      }

      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (BlockArgument arg : forOp.getRegionIterArgs()) {
          Value loopArg = forOp.getInitArgs()[arg.getArgNumber() - 1];
          for (OpResult res : root->getResults()) {
            if (res == loopArg && tt::isTensorPointerType(res.getType())) {
              llvm::errs() << "arg: " << arg << "\n";
              llvm::errs() << "loopArg: " << loopArg << "\n";
              llvm::errs() << "arg type: " << arg.getType() << "\n";

              // Modify the layout of the loop init argument...
              tt::PointerType ptrType = cast<tt::PointerType>(arg.getType());
              auto tensorType =
                  cast<RankedTensorType>(ptrType.getPointeeType());
              arg.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                               ptrType.getAddressSpace()));

              // ... and then propagate it to the operations in the loop.
              propagateLayout(arg, layout, rewriter);
            }
          }
        }
        continue;
      }

      changeAndPropagateLayout(user, layout, rewriter);
    }
  }

  // TODO: change the implementation to handle only operation yielding one
  // result?
  // Change the \p layout of the \p op result(s) and propagate the new
  // result type to its users.
  static void changeAndPropagateLayout(Operation *op, Attribute layout,
                                       IRRewriter &rewriter) {
    assert(op && op->getNumResults() != 0 &&
           "Expecting operation yielding a result");

    rewriter.modifyOpInPlace(op, [&]() {
      for (Value res : op->getResults()) {
        if (!tt::isTensorPointerType(res.getType()))
          continue;

        auto ptrType = cast<tt::PointerType>(res.getType());
        auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
        res.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                         ptrType.getAddressSpace()));
      }
    });
    llvm::errs() << "Coalesced op: " << *op << "\n";

    propagateLayout(op, layout, rewriter);
  }

  void coalesceOp(Attribute encoding, Operation *op) {
    llvm::errs() << "Coalescing op: " << *op << "\n";

    OpBuilder builder(op);
    IRRewriter rewriter(builder);

    // Convert operands
    // Note: for load/store with a blocked pointers argument we cannot change
    // the operand type, instead we change the output type of
    // `make_tensor_ptr` and propagate the new output type along the def-use
    // chain.
    SmallVector<Value, 4> newArgs;
    for (Value operand : op->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (tensorType &&
          !isa<ttg::SharedEncodingAttr>(tensorType.getEncoding())) {
        RankedTensorType newType = getNewType(tensorType, encoding);
        newArgs.push_back(rewriter.create<ttg::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        assert(isa<tt::PointerType>(operand.getType()) &&
               "Expecting operand to have blocked pointer type");
        auto defOp = findDefiningMakeTensorPtrOp(operand);
        assert(defOp && "Expected a make_tensor_ptr operation");

        llvm::errs() << "Found make_tensor_ptr definition: " << *defOp << "\n";
        changeAndPropagateLayout(*defOp, encoding, rewriter);
        newArgs.push_back(operand);
      }
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = isa<ttg::AsyncCopyGlobalToLocalOp>(op);
      assert(!isAsync &&
             "AsyncCopyGlobalToLocalOp not supported for Intel GPU");
      newTypes.push_back(getNewType(cast<RankedTensorType>(t), encoding));
    }

    // Construct new op with the new encoding.
    Operation *newOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(), newArgs,
                        newTypes, op->getAttrs());

    // Cast the results back to the original layout.
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = rewriter.create<ttg::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();

    llvm::errs() << "newOp: " << *newOp << "\n";
    assert(succeeded(verify(newOp)) && "Operation verification failed");
  }

public:
  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    tt::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    llvm::MapVector<Operation *, Attribute> layoutMap;
    moduleOp.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr)
        return;

      RankedTensorType refTensorType = getRankedTensorType(ptr.getType());
      if (!refTensorType || !refTensorType.getEncoding())
        return;

      //      static int n = 0;
      //    if (tt::isTensorPointerType(ptr.getType()))
      //    n++;

      //      if (n != 2)
      //      return;

      int numWarps = ttg::TritonGPUDialect::getNumWarps(moduleOp);
      int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp,
                           layoutMap);
    });

    llvm::errs() << "layoutMap:\n";
    for (auto [op, encoding] : layoutMap) {
      llvm::errs() << "op: " << *op << "\n";
      llvm::errs() << "encoding: " << encoding << "\n";
    }
    llvm::errs() << "\n";

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    for (auto [op, layout] : layoutMap) {
      coalesceOp(layout, op);
      if (failed(verify(moduleOp))) {
        for (Operation &op1 : moduleOp.getOps()) {
          if (isa<tt::FuncOp>(op1)) {
            for (Operation &op2 : cast<tt::FuncOp>(op1).getOps()) {
              if (failed(verify(&op2))) {
                llvm::errs() << "op2: " << op2 << "\n";
                llvm::errs() << "Operation verification failed.\n";
              }
            }
          }
        }
        llvm::errs() << "Module verification failed.\n";
        llvm::errs() << "mod: " << moduleOp << "\n";
        assert(false);
      }
      llvm::errs() << "Module verified.\n";
    }
  }
};

} // namespace
