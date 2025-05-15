#include "intel/include/Analysis/AxisInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Utils.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
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

    const auto &contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
    SmallVector<unsigned> order = argSort(contiguity);
    LDBG("order=[" << triton::join(order, ", ") << "]");

    RankedTensorType refTensorType = ttgi::getRankedTensorType(ptr.getType());
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
    LDBG("Attempting to find `makeTensorPtrOp` defining: " << val);

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
    if (auto opRes = dyn_cast<OpResult>(val)) {
      Operation *defOp = opRes.getOwner();
      if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
        Value val = forOp.getYieldedValues()[opRes.getResultNumber()];
        return findDefiningMakeTensorPtrOp(val);
      }
      assert(false && "unhandled operation");
    }

    return std::nullopt;
  }

  static bool filterUser(Operation *op) {
    // Yield operations trigger updating the layout of the containing loop
    // results, don't skip them.
    if (isa<scf::YieldOp>(op))
      return false;

    // Skip operations that don't yield a result and contain no regions.
    if (op->getNumResults() == 0 && op->getNumRegions() == 0)
      return true;

    // Operations that do not consume a block pointer aren't interesting.
    if (llvm::none_of(op->getOperandTypes(), tt::isTensorPointerType))
      return true;

    // Operations that do not yield a block pointer aren't interesting.
    if (op->getNumRegions() == 0 &&
        llvm::none_of(op->getResultTypes(), tt::isTensorPointerType))
      return true;

    return false;
  }

  // Change the \p layout of the \p op result and propagate the new result type
  // to its users.
  void changeAndPropagateLayout(Operation *op, Attribute layout,
                                IRRewriter &rewriter) const {
    assert(op && op->getNumResults() != 0 &&
           "Expecting operation yielding results");

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
    LDBG("Coalesced op: " << *op);

    propagateLayout(op, layout, rewriter);
  }

  // Propagate the layout of the \p root operation's result to its users.
  void propagateLayout(Operation *root, Attribute layout,
                       IRRewriter &rewriter) const {
    assert(root->getNumResults() != 0 &&
           "Expecting an operation yielding a result");

    LDBG("root: " << *root);
    for (Operation *user : root->getUsers()) {
      if (filterUser(user))
        continue;

      LDBG("root's user: " << *user << "\n");
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        propagateLayoutToArgsAndBody(forOp, root, layout, rewriter);
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        if (auto forOp = yieldOp->getParentOfType<scf::ForOp>())
          propagateLayoutToLoopResults(forOp, layout, rewriter);
        continue;
      }
      changeAndPropagateLayout(user, layout, rewriter);
    }
  }

  // Propagate the layout of the \p arg block argument to its users.
  void propagateLayout(BlockArgument arg, Attribute layout,
                       IRRewriter &rewriter) const {
    LDBG("arg: " << arg);
    for (Operation *user : arg.getUsers()) {
      if (filterUser(user))
        continue;

      LDBG("arg's user: " << *user << "\n");
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        if (auto forOp = yieldOp->getParentOfType<scf::ForOp>())
          propagateLayoutToLoopResults(forOp, layout, rewriter);
        continue;
      }
      changeAndPropagateLayout(user, layout, rewriter);
    }
  }

  // Propagate the layout of the \p root operation's result to the \p forOp loop
  // init argument that uses it, and transitively to the operations in the loop
  // body that use that argument.
  void propagateLayoutToArgsAndBody(scf::ForOp forOp, Operation *root,
                                    Attribute layout,
                                    IRRewriter &rewriter) const {
    assert(llvm::any_of(root->getUsers(),
                        [&](Operation *user) { return user == forOp; }) &&
           "Expecting the loop to be a user of the root operation");

    for (BlockArgument arg : forOp.getRegionIterArgs()) {
      Value loopArg = forOp.getInitArgs()[arg.getArgNumber() - 1];
      for (OpResult res : root->getResults()) {
        if (res != loopArg || !tt::isTensorPointerType(res.getType()))
          continue;

        LDBG("loopArg: " << loopArg);

        // Modify the layout of the loop init argument...
        tt::PointerType ptrType = cast<tt::PointerType>(arg.getType());
        auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
        arg.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                         ptrType.getAddressSpace()));

        // ... and then propagate it to the operations in the loop.
        propagateLayout(arg, layout, rewriter);
      }
    }
  }

  // Modify the given loop \p forOp and propagate the result of the enclosing
  // loop.
  void propagateLayoutToLoopResults(scf::ForOp forOp, Attribute layout,
                                    IRRewriter &rewriter) const {
    Operation *yieldOp = forOp.getBody()->getTerminator();

    rewriter.modifyOpInPlace(forOp, [&]() {
      for (auto [opType, res] :
           llvm::zip(yieldOp->getOperandTypes(), forOp.getResults())) {
        if (opType == res.getType())
          continue;

        assert(tt::isTensorPointerType(res.getType()) &&
               tt::isTensorPointerType(opType) && "Expecting blocked pointers");
        assert(cast<RankedTensorType>(
                   cast<tt::PointerType>(opType).getPointeeType())
                       .getEncoding() == layout &&
               "Unexpected layout");

        auto resType = cast<tt::PointerType>(res.getType());
        RankedTensorType tensorType = ttgi::getRankedTensorType(resType);
        res.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                         resType.getAddressSpace()));
      }
    });

    propagateLayout(forOp, layout, rewriter);
  }

  void coalesceOp(Attribute encoding, Operation *op) {
    LDBG("Coalescing op: " << *op);

    OpBuilder builder(op);

    // Convert operands
    // Note: for load/store with a blocked pointers argument we cannot change
    // the operand type, instead we change the output type of
    // `make_tensor_ptr` and propagate the new output type along the def-use
    // chain.
    SmallVector<Value, 4> newArgs;
    for (Value operand : op->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (tensorType &&
          !isa<triton::gpu::SharedEncodingTrait>(tensorType.getEncoding())) {
        RankedTensorType newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<ttg::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        assert(isa<tt::PointerType>(operand.getType()) &&
               "Expecting operand to have blocked pointer type");
        auto defOp = findDefiningMakeTensorPtrOp(operand);
        assert(defOp && "Expected a make_tensor_ptr operation");
        LDBG("Found make_tensor_ptr definition: " << *defOp);
        IRRewriter rewriter(builder);
        changeAndPropagateLayout(*defOp, encoding, rewriter);
        newArgs.push_back(operand);
      }
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      assert(!isa<ttg::AsyncCopyGlobalToLocalOp>(op) &&
             "AsyncCopyGlobalToLocalOp not supported for Intel GPU");
      newTypes.push_back(getNewType(cast<RankedTensorType>(t), encoding));
    }

    // Construct new op with the new encoding.
    Operation *newOp =
        builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs,
                       newTypes, op->getAttrs());

    // Cast the results back to the original layout.
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<ttg::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }

    LDBG("Old op: " << *op);
    LDBG("newOp: " << *newOp);
    op->erase();

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

      RankedTensorType refTensorType = ttgi::getRankedTensorType(ptr.getType());
      if (!refTensorType || !refTensorType.getEncoding())
        return;

      int numWarps = ttg::lookupNumWarps(curr);
      int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp,
                           layoutMap);
    });

    LLVM_DEBUG({
      DBGS() << "layoutMap:\n";
      if (layoutMap.empty())
        DBGS() << "\t<empty>";
      for (auto [op, encoding] : layoutMap) {
        DBGS() << "\top: " << *op << "\n";
        DBGS() << "\tencoding: " << encoding << "\n";
      }
      llvm::errs() << "\n";
    });

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    for (auto [op, layout] : layoutMap) {
      coalesceOp(layout, op);
    }

    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
