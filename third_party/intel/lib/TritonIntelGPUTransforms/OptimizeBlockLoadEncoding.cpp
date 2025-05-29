#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/PriorityWorklist.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/Utility.h"

#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir {
namespace triton {
namespace gpu::intel {

#define DEBUG_TYPE "tritongpu-optimize-block-encoding"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

#if 1
class BlockedToSubgroupBlockIO : public mlir::OpRewritePattern<LoadOp> {
private:
  llvm::MapVector<Operation *, Attribute> layoutMap;

public:
  BlockedToSubgroupBlockIO(mlir::MLIRContext *context,
                           llvm::MapVector<Operation *, Attribute> layoutMap,
                           int benefit)
      : OpRewritePattern<LoadOp>(context, benefit), layoutMap(layoutMap) {}

  // well, this is very stupid. instead of messing around with the block ptr
  // which we don't even care about (how does that get changed during remove
  // layout conversions anyway?) so let's just try inserting the convert layout
  // op as before and then checking the load in RemoveLayoutConversions to see
  // if the first user is a convert layout op to subgroup 2d block io. if it is
  // then, idk, maybe we can pick that layout somehow? or maybe we can intercept
  // this layout in TritonToTritonIntelGPU somehow. but that probably blows up
  // the rest of the pipeline. probably worth trying the hack in
  // RemoveLayoutConversions first
  mlir::LogicalResult
  matchAndRewrite(LoadOp loadOp,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::errs() << "LoadOp: " << loadOp << "\n";

    auto loadOpType = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!loadOpType)
      return failure();

    // traverse the def-use chain until we find the DotOpOperand layout for this
    // layout
    return failure();

#if 0
    auto oldType = dyn_cast<RankedTensorType>(loadOp.getType());
    // invalid or already visited
    if (!oldType || isa<Subgroup2DBlockEncodingAttr>(oldType.getEncoding()))
      return failure();

    if (layoutMap.find(loadOp) == layoutMap.end())
      return failure();
    auto encoding = layoutMap.lookup(loadOp);


    llvm::errs() << "processing loadop: " << loadOp << "\nwith new encoding "
                 << encoding << "\n";
     rewriter.setInsertionPointAfterValue(loadOp);
    auto newLoad = rewriter.clone(*loadOp);

    auto newType = RankedTensorType::get(oldType.getShape(), oldType.getElementType(), encoding);
    newLoad->getResult(0).setType(newType);

    // convert back to blocked
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(loadOp, oldType, newLoad->getResult(0));

    return success();
#endif
  }
};

#else
// does the load op care about the mismatch between operand and result?
class BlockedToSubgroupBlockIO
    : public mlir::OpRewritePattern<triton::MakeTensorPtrOp> {
private:
  llvm::MapVector<Operation *, Attribute> layoutMap;

  void setEncoding(ValueRange values, Attribute encoding,
                   SmallVector<Value> &changed, Operation *op) const {
    for (Value value : values) {
      bool hasChanged = false;
      auto ptrType = dyn_cast<PointerType>(value.getType());
      if (!ptrType)
        continue;
      auto tensorType = dyn_cast<RankedTensorType>(ptrType.getPointeeType());
      if (!tensorType)
        continue;

      llvm::errs() << "Considering value: " << value << "\n";
      changed.push_back(value);
    }
  }

  SmallVector<Value>
  propagateToUsers(Value value, Attribute encoding,
                   SetVector<Operation *> &opsToDelete) const {
    SmallVector<Value> changed;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        Value arg = forOp.getTiedLoopRegionIterArg(&use);
        Value result = forOp.getTiedLoopResult(&use);
        setEncoding({arg, result}, encoding, changed, user);
        continue;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
        Value arg = whileOp.getBeforeArguments()[use.getOperandNumber()];
        setEncoding({arg}, encoding, changed, user);
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        auto parent = yieldOp->getParentOp();
        SmallVector<Value> valuesToPropagate;
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
          valuesToPropagate.push_back(
              parent->getResult(use.getOperandNumber()));
        if (auto forOp = dyn_cast<scf::ForOp>(parent))
          valuesToPropagate.push_back(
              forOp.getRegionIterArg(use.getOperandNumber()));
        if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
          valuesToPropagate.push_back(
              whileOp.getBeforeArguments()[use.getOperandNumber()]);
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
          setEncoding(valuesToPropagate, encoding, changed, user);
        continue;
      }
      if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
        auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
        // Skip arg 0 as it is the condition.
        unsigned argIndex = use.getOperandNumber() - 1;
        Value afterArg = whileOp.getAfterArguments()[argIndex];
        Value result = whileOp->getResult(argIndex);
        setEncoding({afterArg, result}, encoding, changed, user);
        continue;
      }
      if (isa<LoadOp>(user)) {
        // though we need to change load op???
        setEncoding(cast<LoadOp>(user).getResult(), encoding, changed, user);
        continue;
      }

      llvm::errs() << "user: " << *user << "\n";
    }
    return changed;
  }

public:
  BlockedToSubgroupBlockIO(mlir::MLIRContext *context,
                           llvm::MapVector<Operation *, Attribute> layoutMap,
                           int benefit)
      : OpRewritePattern<triton::MakeTensorPtrOp>(context, benefit),
        layoutMap(layoutMap) {}

  mlir::LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp makeTensorPtrOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (layoutMap.find(makeTensorPtrOp) == layoutMap.end())
      return failure();
    auto encoding = layoutMap.lookup(makeTensorPtrOp);

    auto oldPtrType = cast<PointerType>(makeTensorPtrOp.getType());
    auto oldType = cast<RankedTensorType>(oldPtrType.getPointeeType());
    if (isa<Subgroup2DBlockEncodingAttr>(oldType.getEncoding())) {
      return failure();
    }

    auto newType = RankedTensorType::get(oldType.getShape(),
                                         oldType.getElementType(), encoding);
    auto newPtrType = PointerType::get(newType, oldPtrType.getAddressSpace());

    rewriter.setInsertionPointAfter(makeTensorPtrOp);
    auto newMakeTensorPtrOp = rewriter.clone(*makeTensorPtrOp);
    newMakeTensorPtrOp->getResult(0).setType(newPtrType);
    rewriter.replaceAllUsesWith(makeTensorPtrOp,
                                newMakeTensorPtrOp->getResult(0));
    rewriter.eraseOp(makeTensorPtrOp);

    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([](PointerType ty) {
      llvm::errs() << "ty: " << ty << "\n";
      return ty; // RankedTensorType::get(ty.getShape(), ty.getElementType());
    });
#if 0
    // But don't remove them from the tensors inside descriptors.
    replacer.addReplacement([](TensorDescType ty) -> std::pair<Type, WalkResult> {
      return {ty, WalkResult::skip()};
    });
#endif
    replacer.recursivelyReplaceElementsIn(*container, /*replaceAttrs=*/false,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

#if 0

    SetVector<Operation*> opsToDelete;
      opsToDelete.insert(makeTensorPtrOp);

    // now propagate the new encoding
    SmallVector<Value> queue;
    queue.push_back(makeTensorPtrOp->getResult(0));
    while (!queue.empty()) {
      Value currentValue = queue.back();
      queue.pop_back();

      // SmallVector<Value> changed = propagateToUsers(currentValue, encoding);
       for (OpOperand &use : currentValue.getUses()) {
        Operation *user = use.getOwner();
        if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        BlockArgument arg = forOp.getTiedLoopRegionIterArg(&use);
        forOp.getInitArgsMutable()[arg.getArgNumber() - 1].assign(currentValue);

        BlockArgument result = forOp.getTiedLoopResult(&use);
        queue.push_back(result);
        // clone the for loop
        auto newLoop = replaceForOpWithNewSignature(rewriter, forOp, {});
        // replace the old result with the new type


        continue;
        }
       }

      queue.insert(queue.end(), changed.begin(), changed.end());
    }


    // rewriter.replaceAllUsesWith(makeTensorPtrOp,
    //                             newMakeTensorPtrOp->getResult(0));
    // rewriter.eraseOp(makeTensorPtrOp);
    for (Operation *op : llvm::reverse(opToDelete))
      op->erase();
#endif
    return success();
  }
};
#endif

} // namespace

#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEBLOCKLOADENCODINGPASS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

// maybe it should be a struct?
class TritonIntelGPUOptimizeBlockLoadEncodingPass
    : public impl::TritonIntelGPUOptimizeBlockLoadEncodingPassBase<
          TritonIntelGPUOptimizeBlockLoadEncodingPass> {

  void getSubgroup2DBlockLayoutForOperand(
      Value operand, DpasEncodingAttr dpasLayout,
      llvm::MapVector<Operation *, Attribute> &layoutMap) {
    llvm::errs() << "analyzing operand: " << operand << "\n";
    auto isCandidateLoad = [](Value v) -> LoadOp {
      // TODO: this probably only works for loads in the loop...
      // Peel out the original cvt dot_op<..., #blocked>
      // and any other potential cvt/trans ops
      while (true) {
        if (auto cvtOp = v.getDefiningOp<ConvertLayoutOp>()) {
          v = cvtOp.getSrc();
          continue;
        }
        if (auto transOp = v.getDefiningOp<TransOp>()) {
          v = transOp.getSrc();
          continue;
        }
        break;
      }
      return isa<LoadOp>(v.getDefiningOp()) ? cast<LoadOp>(v.getDefiningOp())
                                            : nullptr;
    };

    LoadOp loadOp = isCandidateLoad(operand);
    if (!loadOp)
      return;

    auto dotOperandType = cast<RankedTensorType>(operand.getType());
    auto dotOperandEncoding =
        cast<DotOperandEncodingAttr>(dotOperandType.getEncoding());
    // layout width is determined by the DPAS operand encoding width
    const int kWidth = dotOperandEncoding.getKWidth();

    Attribute blockIOAttr =
        loadOp->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
    if (!blockIOAttr)
      return;

    llvm::errs() << "Processing load op: " << loadOp << "\n";

    // get the tensor ptr op for the load
    Value ptr = loadOp.getPtr();
    assert(isTensorPointerType(ptr.getType()) && "expecting pointer to tensor");
    auto makeTensorPtrOp = getMakeTensorPtrOp(ptr);

    assert(makeTensorPtrOp &&
           "expecting a tensor pointer parent to block io load "
           "with tensor pointer type");
    auto oldTensorPtrType = cast<PointerType>(makeTensorPtrOp.getType());
    auto oldTensorType =
        cast<RankedTensorType>(oldTensorPtrType.getPointeeType());
    auto oldLayout = cast<BlockedEncodingAttr>(
        oldTensorType.getEncoding()); // TODO: will this hold?
    llvm::errs() << "oldTensorPtrType = " << oldTensorPtrType << "\n";
    llvm::errs() << "oldTensorType = " << oldTensorType << "\n";

    llvm::errs() << "makeTensorPtrOp: " << makeTensorPtrOp << "\n";

    auto CTALayout = getCTALayout(dpasLayout);
    const unsigned elemSizeInBits =
        oldTensorType.getElementType().getIntOrFloatBitWidth();

    auto tileParams = Subgroup2DBlockEncodingAttr::getInstrShapeForLayout(
        cast<DistributedEncodingTrait>(dpasLayout), oldTensorType.getShape(),
        blockIOAttr == StringAttr::get(&getContext(), "row_major"),
        elemSizeInBits / 8, &getContext());
    SmallVector<unsigned> instrShape{tileParams[0], tileParams[1]};

    const unsigned vBlocks = tileParams[2];

    auto subgroup2DBlockEncoding = Subgroup2DBlockEncodingAttr::get(
        &getContext(), dpasLayout.getWarpsPerCTA(), CTALayout, instrShape,
        tileParams[2], oldLayout.getOrder(), kWidth,
        dpasLayout.getThreadsPerWarp());
    llvm::errs() << "subgroup2DBlockEncoding for load: "
                 << subgroup2DBlockEncoding << "\n";

    layoutMap[loadOp] = subgroup2DBlockEncoding;
  }

  void setEncoding(ValueRange values, Attribute encoding,
                   SmallVector<Value> &changed, Operation *op) {
    for (Value value : values) {
      changed.push_back(value);
    }
  }

  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = cast<RankedTensorType>(type);
    return RankedTensorType::get(tensorType.getShape(),
                                 tensorType.getElementType(), encoding);
  }

  static Type getNewPointerType(Type type, Attribute encoding) {
    assert(isa<PointerType>(type) && "expected a ptr type!");
    auto oldPointerType = cast<PointerType>(type);
    return PointerType::get(
        getNewType(oldPointerType.getPointeeType(), encoding),
        oldPointerType.getAddressSpace());
  }
#if 1

  SmallVector<Value> getTiedArgs(Operation *op, int resultIdx) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto iterArg = forOp.getRegionIterArg(resultIdx);
      auto result = forOp.getResult(resultIdx);
      auto yieldVal = forOp.getBody()->getTerminator()->getOperand(resultIdx);
      auto initVal = forOp.getInitArgs()[resultIdx];
      return {iterArg, result, yieldVal, initVal};
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      auto iterArg = whileOp.getBeforeArguments()[resultIdx];
      auto result = whileOp.getResults()[resultIdx];
      auto yieldVal =
          whileOp.getBeforeBody()->getTerminator()->getOperand(resultIdx);
      auto initVal = whileOp.getOperands()[resultIdx];
      return {iterArg, result, iterArg, initVal};
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Value> values;
      for (auto &block : ifOp.getThenRegion().getBlocks()) {
        auto terminator = block.getTerminator();
        if (isa<scf::YieldOp>(terminator))
          values.push_back(terminator->getOperands()[resultIdx]);
      }
      for (auto &block : ifOp.getElseRegion().getBlocks()) {
        auto terminator = block.getTerminator();
        if (isa<scf::YieldOp>(terminator))
          values.push_back(terminator->getOperands()[resultIdx]);
      }
      values.push_back(ifOp->getResults()[resultIdx]);
      return values;
    }
    return {};
  }

  struct EncodingInfo {
    Attribute desiredEncoding;
    bool isPtr = false;
    int addressSpace = -1;
    bool requiresConvert = false;

    bool operator==(const EncodingInfo &other) const {
      return desiredEncoding == other.desiredEncoding && isPtr == other.isPtr &&
             addressSpace == other.addressSpace;
    }
  };

  void rewriteTensorLayoutsForOp(Attribute encoding, Operation *op,
                                 mlir::MLIRContext *context) {
    auto loadOp = cast<LoadOp>(op);
    llvm::errs() << "processing load op: " << loadOp << "\n";
    auto loadPtrType = cast<PointerType>(loadOp->getOperand(0).getType());
    auto addressSpace = loadPtrType.getAddressSpace();

    llvm::MapVector<TypedValue<PointerType>, EncodingInfo> valueToEncodingInfo;
    llvm::PriorityWorklist<TypedValue<PointerType>> worklist;

    auto updateEncoding = [&](ArrayRef<Value> descValues, EncodingInfo info) {
      for (auto value : descValues) {
        llvm::errs() << "update encoding for value " << value << "\n";
        bool requiresConvert = llvm::any_of(
            value.getUsers(), [](auto user) { return isa<LoadOp>(user); });
        info.requiresConvert = requiresConvert;
        llvm::errs() << "requiresConvert? " << requiresConvert << "\n";
        auto typedVal = cast<TypedValue<PointerType>>(value);
        auto itr = valueToEncodingInfo.find(typedVal);
        if (itr == valueToEncodingInfo.end()) {
          valueToEncodingInfo[typedVal] = info;
          worklist.insert(typedVal);
        } else {
          llvm::errs() << "existing encoding: " << itr->second.desiredEncoding
                       << "\n";
          llvm::errs() << "new encoding: " << info.desiredEncoding << "\n";
          // we have already seen this value. make sure it is the right one!
          assert(itr->second == info && "already visited encoding info for "
                                        "value, expected them to be equal!");
          continue;
        }
      }
    };

    worklist.insert(cast<TypedValue<PointerType>>(loadOp->getOperand(0)));

    // Propagate encoding info
    while (!worklist.empty()) {
      auto crtValue = worklist.pop_back_val();
      llvm::errs() << "Processing " << crtValue << "\n";

      // Propagate to users
      for (OpOperand &use : crtValue.getUses()) {
        auto op = use.getOwner();
        if (isa<scf::ForOp, scf::WhileOp>(op)) {
          auto offset = 3 * isa<scf::ForOp>(op);
          auto vals = getTiedArgs(op, use.getOperandNumber() - offset);
          updateEncoding(vals, EncodingInfo{encoding, /*isPtr=*/true,
                                            /*addressSpace=*/addressSpace});
        } else if (isa<scf::YieldOp>(op)) {
          auto vals = getTiedArgs(op->getParentOp(), use.getOperandNumber());
          updateEncoding(vals, EncodingInfo{encoding, /*isPtr=*/true,
                                            /*addressSpace=*/addressSpace});
        }
      }

      // Propagate to defining ops
      if (auto opResult = dyn_cast<OpResult>(crtValue)) {
        auto definingOp = opResult.getOwner();
        if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
          auto vals = getTiedArgs(definingOp, opResult.getResultNumber());
          updateEncoding(vals, EncodingInfo{encoding, /*isPtr=*/true,
                                            /*addressSpace=*/addressSpace});
        }
      } else if (auto blockArg = dyn_cast<BlockArgument>(crtValue)) {
        auto parentOp = blockArg.getOwner()->getParentOp();
        if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
          auto offset = isa<scf::ForOp>(parentOp);
          auto vals = getTiedArgs(parentOp, blockArg.getArgNumber() - offset);
          updateEncoding(vals, EncodingInfo{encoding, /*isPtr=*/true,
                                            /*addressSpace=*/addressSpace});
        }
      }
    }

    // update types

    for (auto &[desc, einfo] : valueToEncodingInfo) {
      llvm::errs() << "update type for value " << desc << "\n";
      assert(einfo.desiredEncoding);
      Attribute newEncoding = einfo.desiredEncoding;
      PointerType oldType = desc.getType();
      auto oldTensorTy = cast<RankedTensorType>(oldType.getPointeeType());
      auto newTensorTy = RankedTensorType::get(
          oldTensorTy.getShape(), oldTensorTy.getElementType(), newEncoding);

      desc.setType(PointerType::get(newTensorTy, oldType.getAddressSpace()));
      if (einfo.requiresConvert) {
        for (auto user : desc.getUsers()) {
          if (auto loadOp = dyn_cast<LoadOp>(user)) {
            llvm::errs() << "converting load op: " << loadOp << "\n";

            OpBuilder builder(loadOp);
            auto oldLoadType = loadOp.getType();
            Value result = loadOp.getResult();

            builder.setInsertionPointAfter(loadOp);
            auto cvt = builder.create<ConvertLayoutOp>(
                loadOp.getLoc(), result.getType(), result);
            result.setType(newTensorTy);

            result.replaceAllUsesExcept(cvt.getResult(), cvt.getOperation());
          }
        }
      }
    }
  }
#else
  void rewriteTensorLayoutsForOp(Attribute encoding, Operation *op) {
    assert(isa<LoadOp>(op) && "expected load op");
    auto loadOp = cast<LoadOp>(op);
    llvm::errs() << "rewrite load op: " << loadOp << "\n";
    if (!isa<PointerType>(loadOp.getOperand(0).getType()))
      return;

    // find the tensor ptr for the load op
    Value ptr = loadOp.getPtr();
    assert(isTensorPointerType(ptr.getType()) && "expecting pointer to tensor");
    auto makeTensorPtrOp = getMakeTensorPtrOp(ptr);
    auto typeToReplace = cast<PointerType>(makeTensorPtrOp.getType());
    auto oldPointeeType =
        cast<RankedTensorType>(typeToReplace.getPointeeType());

    auto newPtrType = PointerType::get(
        RankedTensorType::get(oldPointeeType.getShape(),
                              oldPointeeType.getElementType(), encoding),
        typeToReplace.getAddressSpace());

    llvm::errs() << "starting replacement with " << makeTensorPtrOp << "\n";
    llvm::errs() << "replacing " << typeToReplace << " with " << newPtrType
                 << "\n";

    // propagate the new pointer type through the def-use chain starting with
    // the MakeTensorPtr op
    SetVector<Operation *> opsToDelete;
    SmallVector<Operation *> queue;
    queue.push_back(makeTensorPtrOp);
    while (!queue.empty()) {
      Operation *op = queue.back();
      llvm::errs() << "Processing op " << *op << "\n";
      queue.pop_back();

      auto newOp = propagatePointerType(encoding, op);
      llvm::errs() << "newOp: " << *newOp << "\n";
      for (auto use : op->getUsers()) {
        queue.push_back(use);
        llvm::errs() << "use: " << *use << "\n";
      }

      // wonder if we can just delete it here...
      op->replaceAllUsesWith(newOp);
      opsToDelete.insert(op);
    }

    for (Operation *op : llvm::reverse(opsToDelete))
      op->erase();
  }
#endif
  Operation *propagatePointerType(Attribute encoding, Operation *op) {
    OpBuilder builder(op);

    llvm::errs() << "Rewriting op " << *op << "\n\tto use new layout:\n"
                 << encoding << "\n\n";

#if 0
    // add a layout conversion from the op parent to the op
    SmallVector<Value, 4> newArgs;
    for (auto operand : op->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      // is tensor pointer type? actually maybe we just forward the operands and change the outputs
      if (tensorType &&
          !isa<triton::gpu::SharedEncodingTrait>(tensorType.getEncoding())) {
        Type newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }
#endif

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      if (isa<PointerType>(t)) {
        newTypes.push_back(getNewPointerType(t, encoding));
      } else {
        newTypes.push_back(t);
      }
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create(op->getLoc(), op->getName().getIdentifier(),
                       /*newArgs*/ op->getOperands(), newTypes, op->getAttrs());

    return newOp;
  }

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // 1. Find all candidate loads for conversion to Subgroup2D Block Encoding
    // layout. Start from the Dot Op operands and look for loads with the
    // "block_io" attribute.
    llvm::MapVector<Operation *, Attribute> layoutMap;
    m.walk([&](DotOp dotOp) {
      llvm::errs() << "dotOp: " << dotOp << "\n";
      llvm::errs() << "dotOp: " << dotOp << "\n";

      auto dotOpType = cast<RankedTensorType>(dotOp.getResult().getType());
      auto dotOpLayout = dotOpType.getEncoding();
      llvm::errs() << "dotOpLayout: " << dotOpLayout << "\n";

      auto dpasLayout = dyn_cast<DpasEncodingAttr>(dotOpLayout);
      if (!dpasLayout)
        return;

      getSubgroup2DBlockLayoutForOperand(dotOp.getA(), dpasLayout, layoutMap);
      getSubgroup2DBlockLayoutForOperand(dotOp.getB(), dpasLayout, layoutMap);
    });

    // 2. Rewrite makeTensorPtr ops and Load ops to use the new types.
    for (auto &kv : layoutMap) {
      rewriteTensorLayoutsForOp(kv.second, kv.first, context);
    }

#if 0
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    patterns.add<BlockedToSubgroupBlockIO>(context, benefitDefault);

    // triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(patterns,
    //                                                           &getContext());

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      llvm::errs() << "this pass failed.\n";
      signalPassFailure();
    }
#endif
  }
};

} // namespace gpu::intel
} // namespace triton
} // namespace mlir
