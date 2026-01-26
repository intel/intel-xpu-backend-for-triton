#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "triton-intel-block-pointer-to-tdesc"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELBLOCKPOINTERTOTENSORDESC
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

struct TritonIntelBlockPointerToTensorDesc
    : tt::intel::impl::TritonIntelBlockPointerToTensorDescBase<
          TritonIntelBlockPointerToTensorDesc> {
public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    SmallVector<Operation *> candidates;
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isCandidate(op))
        candidates.push_back(op);
      return WalkResult::advance();
    });

    for (Operation *op : candidates) {
      if (auto loadOp = dyn_cast<tt::LoadOp>(op))
        rewriteBlockPointer<tt::LoadOp, tt::DescriptorLoadOp>(loadOp);
      else if (auto storeOp = dyn_cast<tt::StoreOp>(op))
        rewriteBlockPointer<tt::StoreOp, tt::DescriptorStoreOp>(storeOp);
      else
        llvm_unreachable("unhandled operation");
    }

    if (!cleanUp.empty())
      eraseOperations(cleanUp);

    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

private:
  bool isCandidate(Operation *op) const {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op))
      return isCandidate(loadOp);
    else if (auto storeOp = dyn_cast<tt::StoreOp>(op))
      return isCandidate(storeOp);
    return false;
  }

  template <
      typename OpTy,
      std::enable_if_t<llvm::is_one_of<OpTy, tt::LoadOp, tt::StoreOp>::value,
                       bool> = true>
  bool isCandidate(OpTy op) const {
    Value ptr = op.getPtr();
    if (!tt::isTensorPointerType(ptr.getType()))
      return false;

    ArrayRef<int32_t> boundaryCheck = op.getBoundaryCheck();
    auto tensorTy = cast<RankedTensorType>(
        cast<tt::PointerType>(ptr.getType()).getPointeeType());
    if (!llvm::equal(boundaryCheck, llvm::seq<int32_t>(0, tensorTy.getRank())))
      return false;

    auto skipAdvance = [](Value ptr) {
      while (auto advanceOp =
                 dyn_cast_or_null<tt::AdvanceOp>(ptr.getDefiningOp()))
        ptr = advanceOp.getPtr();
      return ptr;
    };

    if (auto arg = dyn_cast<BlockArgument>(ptr)) {
      Operation *parentOp = arg.getParentBlock()->getParentOp();
      // FIXME: Add support of other loop ops if needed.
      auto forOp = dyn_cast<scf::ForOp>(parentOp);
      if (!forOp)
        return false;
      Value initArg =
          forOp.getInitArgs()[arg.getArgNumber() - forOp.getNumInductionVars()];
      if (!isa_and_nonnull<tt::MakeTensorPtrOp>(
              skipAdvance(initArg).getDefiningOp()))
        return false;
      Value yieldVal = forOp.getYieldedValues()[arg.getArgNumber() -
                                                forOp.getNumInductionVars()];
      if (skipAdvance(yieldVal) != arg)
        return false;
    } else if (!isa_and_nonnull<tt::MakeTensorPtrOp>(
                   skipAdvance(ptr).getDefiningOp())) {
      return false;
    }

    return true;
  }

  arith::TruncIOp getOrCreateTruncI32Op(Value v) const {
    assert(v.getType().isInteger(64) && "Expecting i64 value");
    Location loc = v.getLoc();
    OpBuilder builder(v.getContext());
    Type targetType = builder.getIntegerType(32);

    if (Operation *defOp = v.getDefiningOp()) {
      if (auto nextOp = dyn_cast_or_null<arith::TruncIOp>(defOp->getNextNode()))
        if (nextOp.getType() == targetType && nextOp.getIn() == v)
          return nextOp;
      builder.setInsertionPointAfter(defOp);
    } else {
      builder.setInsertionPointToStart(v.getParentBlock());
    }

    return arith::TruncIOp::create(builder, loc, targetType, v);
  }

  tt::MakeTensorDescOp
  createMakeTensorDescOp(tt::MakeTensorPtrOp makeTensorPtrOp,
                         tt::PaddingOption padding) {
    OpBuilder builder(makeTensorPtrOp);
    Location loc = makeTensorPtrOp.getLoc();
    cleanUp.insert(makeTensorPtrOp);

    auto descTy = tt::TensorDescType::get(
        builder.getContext(),
        cast<RankedTensorType>(makeTensorPtrOp.getType().getPointeeType()));
    Value base = makeTensorPtrOp.getBase();
    SmallVector<Value> shape;
    for (Value s : makeTensorPtrOp.getShape())
      shape.push_back(getOrCreateTruncI32Op(s));
    SmallVector<Value> strides = makeTensorPtrOp.getStrides();
    return tt::MakeTensorDescOp::create(builder, loc, descTy, base, shape,
                                        strides, padding);
  }

  void updateYieldVals(scf::ForOp forOp,
                       SmallVectorImpl<Value> &newYieldVals) const {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->back());
    OpBuilder builder(yieldOp);
    scf::YieldOp::create(builder, yieldOp.getLoc(), newYieldVals);
    yieldOp.erase();
  }

  SmallVector<Value> calculateIndices(Value ptr) {
    // Add each dimension offsets to its corresponding dimension of indices.
    auto addIndices = [](OpBuilder &builder, Location loc,
                         Operation::operand_range offsets,
                         SmallVector<Value> &indices) {
      if (indices.empty()) {
        indices = offsets;
        return;
      }

      for (unsigned i = 0; i < indices.size(); ++i)
        indices[i] =
            builder.createOrFold<arith::AddIOp>(loc, indices[i], offsets[i]);
    };

    auto accumulateIndices = [&](Value v, SmallVector<Value> &indices) {
      Value curr = v;
      SmallVector<tt::AdvanceOp> advanceOps;
      while (auto advanceOp =
                 dyn_cast_or_null<tt::AdvanceOp>(curr.getDefiningOp())) {
        advanceOps.push_back(advanceOp);
        curr = advanceOp.getPtr();
        cleanUp.insert(advanceOp);
      }
      if (auto makeTensorPtrOp =
              dyn_cast_or_null<tt::MakeTensorPtrOp>(curr.getDefiningOp())) {
        OpBuilder builder(makeTensorPtrOp);
        addIndices(builder, makeTensorPtrOp.getLoc(),
                   makeTensorPtrOp.getOffsets(), indices);
      }
      for (tt::AdvanceOp advanceOp : llvm::reverse(advanceOps)) {
        OpBuilder builder(advanceOp);
        addIndices(builder, advanceOp.getLoc(), advanceOp.getOffsets(),
                   indices);
      }
    };

    auto updateInitArgs = [](scf::ForOp &forOp,
                             SmallVector<Value> newInitArgs) {
      OpBuilder builder(forOp);
      auto newForOp = scf::ForOp::create(
          builder, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), newInitArgs);
      newForOp->setAttrs(forOp->getAttrs());
      Block *oldBody = forOp.getBody();
      Block *newBody = newForOp.getBody();

      Block::BlockArgListType oldArgs = oldBody->getArguments();
      Block::BlockArgListType newArgs = newBody->getArguments();
      newBody->getOperations().splice(newBody->getOperations().begin(),
                                      oldBody->getOperations());
      for (auto [oldArg, newArg] :
           llvm::zip(oldArgs, newArgs.take_front(oldArgs.size())))
        oldArg.replaceAllUsesWith(newArg);

      SmallVector<Value> oldResults = forOp.getResults();
      SmallVector<Value> newResults =
          newForOp.getResults().take_front(oldResults.size());
      for (auto [oldResult, newResult] : llvm::zip(oldResults, newResults))
        oldResult.replaceAllUsesWith(newResult);

      forOp.erase();
      forOp = newForOp;
    };

    SmallVector<Value> indices;
    if (auto arg = dyn_cast<BlockArgument>(ptr)) {
      auto forOp = cast<scf::ForOp>(arg.getParentBlock()->getParentOp());
      unsigned idx = arg.getArgNumber() - forOp.getNumInductionVars();

      // Accumulate indices outside of the loop, and update the loop init args.
      Value initArg = forOp.getInitArgs()[idx];
      SmallVector<Value> newInitArgs;
      accumulateIndices(initArg, newInitArgs);
      SmallVector<Value> initArgs(forOp.getInitArgs());
      initArgs.append(newInitArgs);
      updateInitArgs(forOp, initArgs);

      // Accumulate indices inside of the loop, and update the loop yield vals.
      Value yieldVal = forOp.getYieldedValues()[idx];
      SmallVector<Value> additionalYieldVals(
          forOp.getBody()->getArguments().take_back(newInitArgs.size()));
      indices = additionalYieldVals;
      accumulateIndices(yieldVal, additionalYieldVals);
      SmallVector<Value> yieldVals(forOp.getYieldedValues());
      yieldVals.append(additionalYieldVals);
      updateYieldVals(forOp, yieldVals);
    } else {
      accumulateIndices(ptr, indices);
    }

    return indices;
  }

  template <typename OpTy, typename DescOpTy,
            std::enable_if_t<
                (llvm::is_one_of<OpTy, tt::LoadOp, tt::StoreOp>::value) &&
                    (llvm::is_one_of<DescOpTy, tt::DescriptorLoadOp,
                                     tt::DescriptorStoreOp>::value) &&
                    ((std::is_same_v<OpTy, tt::LoadOp> &&
                      std::is_same_v<DescOpTy, tt::DescriptorLoadOp>) ||
                     (std::is_same_v<OpTy, tt::StoreOp> &&
                      std::is_same_v<DescOpTy, tt::DescriptorStoreOp>)),
                bool> = true>
  DescOpTy rewriteBlockPointer(OpTy op) {
    auto makeTensorPtrOp =
        tt::intel::findDefiningMakeTensorPtrOp<tt::MakeTensorPtrOp>(op.getPtr())
            .value();
    auto padding = tt::PaddingOption::PAD_ZERO;
    if constexpr (std::is_same_v<OpTy, tt::LoadOp>)
      if (op.getPadding().has_value())
        padding = op.getPadding().value();
    tt::MakeTensorDescOp desc =
        createMakeTensorDescOp(makeTensorPtrOp, padding);
    SmallVector<Value> indices = calculateIndices(op.getPtr());

    OpBuilder builder(op);
    Location loc = op.getLoc();
    DescOpTy descOp;
    if constexpr (std::is_same_v<OpTy, tt::LoadOp>)
      descOp = tt::DescriptorLoadOp::create(builder, loc, op.getType(), desc,
                                            indices);
    else
      descOp = tt::DescriptorStoreOp::create(builder, loc, desc, op.getValue(),
                                             indices);
    op->replaceAllUsesWith(descOp);
    cleanUp.insert(op);
    return descOp;
  }

  void identifyForLoopsToClean(
      const SmallPtrSetImpl<Operation *> &ops,
      llvm::MapVector<scf::ForOp, llvm::SmallSetVector<unsigned, 4>>
          &forOpsToClean) const {
    for (Operation *op : ops) {
      auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op);
      if (!makeTensorPtrOp || !makeTensorPtrOp->hasOneUse())
        continue;

      auto forOp =
          dyn_cast<scf::ForOp>(makeTensorPtrOp->use_begin()->getOwner());
      if (!forOp)
        continue;

      // Remove loop lower and upper bounds and step (3).
      unsigned argIdx = makeTensorPtrOp->use_begin()->getOperandNumber() - 3;
      // Add induction variable (1).
      BlockArgument arg = forOp.getBody()->getArgument(argIdx + 1);
      if (!arg.hasOneUse())
        continue;

      Operation *nextOp = arg.use_begin()->getOwner();
      while (auto advanceOp = dyn_cast<tt::AdvanceOp>(nextOp)) {
        if (!advanceOp->hasOneUse() || !ops.contains(advanceOp))
          break;
        nextOp = advanceOp->use_begin()->getOwner();
      }
      if (!isa_and_nonnull<scf::YieldOp>(nextOp))
        continue;

      if (!forOp.getResult(argIdx).use_empty())
        continue;

      forOpsToClean[forOp].insert(argIdx);
    }
  }

  void
  cleanForLoop(scf::ForOp forOp,
               const llvm::SmallSetVector<unsigned, 4> &indicesToRemove) const {
    SmallVector<Value> newInitArgs;
    for (auto [i, initArg] : llvm::enumerate(forOp.getInitArgs())) {
      if (!indicesToRemove.contains(i)) {
        newInitArgs.push_back(initArg);
      }
    }

    OpBuilder builder(forOp);
    auto newForOp =
        scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), newInitArgs);
    newForOp->setAttrs(forOp->getAttrs());
    Block *oldBody = forOp.getBody();
    Block *newBody = newForOp.getBody();

    SmallVector<Value> oldArgs;
    for (auto [i, oldArg] : llvm::enumerate(oldBody->getArguments())) {
      if (i == 0 || !indicesToRemove.contains(i - 1))
        oldArgs.push_back(oldArg);
    }
    Block::BlockArgListType newArgs = newBody->getArguments();
    while (!newBody->empty())
      newBody->front().erase();
    newBody->getOperations().splice(newBody->getOperations().begin(),
                                    oldBody->getOperations());
    for (auto [oldArg, newArg] : llvm::zip(oldArgs, newArgs))
      oldArg.replaceAllUsesWith(newArg);

    SmallVector<Value> oldResults;
    for (auto [i, oldResult] : llvm::enumerate(forOp.getResults())) {
      if (!indicesToRemove.contains(i))
        oldResults.push_back(oldResult);
    }
    SmallVector<Value> newResults = newForOp.getResults();
    for (auto [oldResult, newResult] : llvm::zip(oldResults, newResults))
      oldResult.replaceAllUsesWith(newResult);

    SmallVector<Value> newYieldVals;
    for (auto [i, yieldVal] : llvm::enumerate(newForOp.getYieldedValues())) {
      if (!indicesToRemove.contains(i))
        newYieldVals.push_back(yieldVal);
    }
    updateYieldVals(newForOp, newYieldVals);
  }

  void eraseOperations(SmallPtrSetImpl<Operation *> &ops) {
    tt::intel::eraseOperations(ops);

    // Collect for loops that need cleaning and which iter args to remove.
    llvm::MapVector<scf::ForOp, llvm::SmallSetVector<unsigned, 4>>
        forOpsToClean;
    identifyForLoopsToClean(ops, forOpsToClean);

    for (auto &[forOp, indicesToRemove] : forOpsToClean) {
      // Remove loop invariant arguments.
      for (auto [i, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (indicesToRemove.contains(i))
          continue;
        Value arg = forOp.getBody()->getArgument(i + 1);
        Value yieldVal = forOp.getYieldedValues()[i];
        if (yieldVal == arg) {
          indicesToRemove.insert(i);
          arg.replaceAllUsesWith(initArg);
        }
      }

      cleanForLoop(forOp, indicesToRemove);
      tt::intel::eraseOperations(ops);
      forOp.erase();
    }

    tt::intel::eraseOperations(ops);
  }

private:
  SmallPtrSet<Operation *, 8> cleanUp;
};

} // namespace
