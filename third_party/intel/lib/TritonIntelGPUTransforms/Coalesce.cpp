#include "intel/include/Analysis/AxisInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Utils.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "tritonintelgpu-coalesce"

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

    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "]: Considering op: " << *op << "\n";
      llvm::dbgs().indent(2) << "axis info of pointer: ";
      axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs().indent(2));
      llvm::dbgs() << "\n";
    });

    const auto &contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
    SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
    LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "order=[" << tt::join(order, ", ") << "]\n";);

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
        auto currOrder = getOrderFromContiguity(
            axisInfoAnalysis.getAxisInfo(val)->getContiguity());
        if (order == currOrder) {
          LLVM_DEBUG(llvm::dbgs().indent(2)
                     << "multi-root-slice: insert to memAccessesSameOrder "
                     << *use << "\n");
          memAccessesSameOrder.insert(use);
        }
      }
    }

    auto shapePerCTA = ttg::getShapePerCTA(refTensorType);
    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;
    unsigned perThread =
        ttgi::getNumElementsPerThread(op, order, axisInfoAnalysis);
    LLVM_DEBUG({
      llvm::dbgs().indent(2)
          << "shapePerCTA=[" << tt::join(shapePerCTA, ", ") << "]\n";
      llvm::dbgs().indent(2) << "perThread for op: " << perThread << "\n";
    });

    for (Operation *opSameOrder : memAccessesSameOrder) {
      if (opSameOrder == op)
        continue;
      unsigned currPerThread =
          ttgi::getNumElementsPerThread(opSameOrder, order, axisInfoAnalysis);
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "perThread for opSameOrder: " << currPerThread);
      perThread = std::max(perThread, currPerThread);
    }

    perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
    LLVM_DEBUG(llvm::dbgs().indent(2) << "perThread: " << perThread << "\n");

    if (!dyn_cast<tt::LoadOp>(op)) {
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

  static bool filterUser(Operation *op) {
    // Yield operations trigger updating the layout of the containing loop
    // results, don't skip them.
    if (isa<scf::YieldOp>(op))
      return false;

    // Condition operations trigger updating the layout of the 'after' region in
    // the containing while loop, don't skip them.
    if (isa<scf::ConditionOp>(op))
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

  // Change the \p layout of the \p op's result \p opRes and propagate the new
  // result type to its users.
  void changeAndPropagateLayout(Operation *op, Value opRes, Attribute layout,
                                IRRewriter &rewriter) const {
    assert(op && op->getNumResults() != 0 &&
           "Expecting operation yielding results");

    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "]: " << "ChangeAndPropagateLayout for: ";
      op->dumpPretty();
      llvm::dbgs() << "opRes: ";
      opRes.printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << "\n";
    });

    rewriter.modifyOpInPlace(op, [&]() {
      assert(tt::isTensorPointerType(opRes.getType()));
      auto ptrType = cast<tt::PointerType>(opRes.getType());
      auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
      opRes.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                         ptrType.getAddressSpace()));
    });

    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "]: Coalesced op: ";
      op->dumpPretty();
    });

    assert(op->getNumResults() == 1 &&
           "Expecting operation yielding one result");
    propagateLayout(op, op->getResult(0), layout, rewriter);
  }

  // Propagate the layout of the \p root operation's result to its users.
  void propagateLayout(Operation *op, Value opRes, Attribute layout,
                       IRRewriter &rewriter) const {
    assert(op && op->getNumResults() != 0 &&
           "Expecting an operation yielding a result");
    assert(opRes &&
           llvm::any_of(op->getResults(),
                        [&](OpResult res) { return res == opRes; }) &&
           "Expecting operation to yield 'opRes'");

    LLVM_DEBUG({
      if (!opRes.getUsers().empty()) {
        llvm::dbgs() << "[" DEBUG_TYPE "]: "
                     << "Propagate layout to operations using: " << opRes
                     << "\n";
      }
    });

    for (Operation *user : opRes.getUsers()) {
      if (filterUser(user))
        continue;

      LLVM_DEBUG({
        llvm::dbgs() << "[" DEBUG_TYPE "]: " << "user: ";
        user->dumpPretty();
      });

      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        propagateLayoutToArgsAndBody(forOp, opRes, layout, rewriter);
        continue;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
        propagateLayoutToArgsAndBody(whileOp, opRes, layout, rewriter);
        continue;
      }

      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        if (auto forOp = yieldOp->getParentOfType<scf::ForOp>())
          for (OpOperand &operand : yieldOp->getOpOperands()) {
            if (operand.get() != opRes)
              continue;
            propagateLayoutToLoopResults(forOp, operand.getOperandNumber(),
                                         layout, rewriter);
          }
        if (auto whileOp = yieldOp->getParentOfType<scf::WhileOp>())
          for (OpOperand &operand : yieldOp->getOpOperands()) {
            if (operand.get() != opRes)
              continue;
            propagateLayoutToLoopResults(whileOp, operand.getOperandNumber(),
                                         layout, rewriter);
          }
        continue;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "[" DEBUG_TYPE "]: After propagating layout:\n";
        op->getParentOfType<ModuleOp>()->dumpPretty();
      });

      changeAndPropagateLayout(user, user->getResult(0), layout, rewriter);
    }
  }

  // Propagate the layout of the \p arg block argument to its users.
  void propagateLayout(BlockArgument arg, Attribute layout,
                       IRRewriter &rewriter) const {
    LLVM_DEBUG({
      if (!arg.getUsers().empty()) {
        llvm::dbgs() << "[" DEBUG_TYPE "]: "
                     << "Propagate layout to operations using: ";
        arg.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << "\n";
      }
    });

    for (Operation *user : arg.getUsers()) {
      if (filterUser(user))
        continue;

      LLVM_DEBUG({
        llvm::dbgs() << "[" DEBUG_TYPE "]: " << "user: ";
        user->dumpPretty();
      });

      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        propagateLayoutToArgsAndBody(forOp, arg, layout, rewriter);
        continue;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
        propagateLayoutToArgsAndBody(whileOp, arg, layout, rewriter);
        continue;
      }

      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        if (auto forOp = yieldOp->getParentOfType<scf::ForOp>())
          for (OpOperand &operand : yieldOp->getOpOperands()) {
            if (operand.get() != arg)
              continue;
            propagateLayoutToLoopResults(forOp, operand.getOperandNumber(),
                                         layout, rewriter);
          }
        if (auto whileOp = yieldOp->getParentOfType<scf::WhileOp>())
          for (OpOperand &operand : yieldOp->getOpOperands()) {
            if (operand.get() != arg)
              continue;
            propagateLayoutToLoopResults(whileOp, operand.getOperandNumber(),
                                         layout, rewriter);
          }
        continue;
      }
      if (auto condOp = dyn_cast<scf::ConditionOp>(user)) {
        if (auto whileOp = condOp->getParentOfType<scf::WhileOp>()) {
          // Propagate layout to "after" region arguments.
          for (auto [condOperand, loopArg] :
               llvm::zip(condOp->getOperands().drop_front(),
                         whileOp.getAfterArguments())) {
            if (condOperand != arg ||
                !tt::isTensorPointerType(condOperand.getType()))
              continue;

            // Modify the layout of the loop argument...
            tt::PointerType ptrType = cast<tt::PointerType>(loopArg.getType());
            auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
            loopArg.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                                 ptrType.getAddressSpace()));
            LLVM_DEBUG({
              llvm::dbgs() << "[" DEBUG_TYPE "]: " << "Propagated layout to: ";
              loopArg.printAsOperand(llvm::dbgs(), {});
              llvm::dbgs() << "\n";
            });

            // ... and then propagate it to the operations in the loop.
            propagateLayout(loopArg, layout, rewriter);
          }
        }
        continue;
      }

      assert(user->getNumResults() == 1 &&
             "Expecting operation yielding one result");
      changeAndPropagateLayout(user, user->getResult(0), layout, rewriter);
    }

    LLVM_DEBUG({
      auto mod =
          arg.getParentBlock()->getParentOp()->getParentOfType<ModuleOp>();
      llvm::dbgs() << "[" DEBUG_TYPE "]: After propagating layout:\n";
      mod->dumpPretty();
    });
  }

  // Propagate the layout of the \p root operation's result to the \p loopOp
  // loop init argument that uses it, and transitively to the operations in the
  // loop body that use that argument.
  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, scf::ForOp, scf::WhileOp>::value>>
  void propagateLayoutToArgsAndBody(OpType loopOp, Value opRes,
                                    Attribute layout,
                                    IRRewriter &rewriter) const {
    for (auto [initArg, arg] :
         llvm::zip(loopOp.getInitsMutable(), loopOp.getRegionIterArgs())) {
      if (initArg.get() != opRes)
        continue;

      // Modify the layout of the loop init argument...
      auto ptrType = cast<tt::PointerType>(arg.getType());
      auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
      arg.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                       ptrType.getAddressSpace()));

      LLVM_DEBUG({
        llvm::dbgs() << "[" DEBUG_TYPE "]: " << "Propagated layout to: ";
        arg.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << "\n";
      });

      // ... and then propagate it to the operations in the loop.
      propagateLayout(arg, layout, rewriter);
    }
  }

  // Modify the \p layout to the loop's operand identified by \p resNum, and
  // propagate the modified loop results to its users.
  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, scf::ForOp, scf::WhileOp>::value>>
  void propagateLayoutToLoopResults(OpType loopOp, unsigned resNum,
                                    Attribute layout,
                                    IRRewriter &rewriter) const {
    Operation *yieldOp = nullptr;
    if constexpr (std::is_same<OpType, scf::ForOp>::value)
      yieldOp = loopOp.getBody()->getTerminator();
    if constexpr (std::is_same<OpType, scf::WhileOp>::value)
      yieldOp = loopOp.getYieldOp();

    Value loopRes = loopOp.getResult(resNum);
    rewriter.modifyOpInPlace(loopOp, [&]() {
      assert(tt::isTensorPointerType(loopRes.getType()) &&
             "Expecting blocked pointers");
      Type resType = loopRes.getType();
      auto ptrType = cast<tt::PointerType>(resType);
      RankedTensorType tensorType = ttgi::getRankedTensorType(resType);
      loopRes.setType(tt::PointerType::get(getNewType(tensorType, layout),
                                           ptrType.getAddressSpace()));
    });

    propagateLayout(loopOp, loopRes, layout, rewriter);
  }

  void coalesceOp(Attribute encoding, Operation *op) {
    LLVM_DEBUG({
      llvm::dbgs() << "[" DEBUG_TYPE "]: " << "Coalescing op: " << *op << "\n";
    });

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
          !isa<ttg::SharedEncodingTrait>(tensorType.getEncoding())) {
        RankedTensorType newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<ttg::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        assert(tt::isTensorPointerType(operand.getType()) &&
               "Expecting operand to have blocked pointer type");
        std::optional<tt::MakeTensorPtrOp> defOp =
            triton::intel::findDefiningMakeTensorPtrOp(operand);
        if (!defOp) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[" DEBUG_TYPE
                        "]: Could not find 'make_tensor_ptr' definition for "
                     << operand << "\n");
          return;
        }

        IRRewriter rewriter(builder);
        changeAndPropagateLayout(*defOp, defOp->getResult(), encoding,
                                 rewriter);
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
      llvm::dbgs() << "[" DEBUG_TYPE "]: " << "layoutMap:\n";
      if (layoutMap.empty())
        llvm::dbgs() << "[" DEBUG_TYPE "]: " << "\t<empty>";
      for (auto [op, encoding] : layoutMap) {
        llvm::dbgs() << "[" DEBUG_TYPE "]: " << "\top: " << *op << "\n";
        llvm::dbgs() << "[" DEBUG_TYPE "]: " << "\tencoding: " << encoding
                     << "\n";
      }
      llvm::dbgs() << "\n";
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
