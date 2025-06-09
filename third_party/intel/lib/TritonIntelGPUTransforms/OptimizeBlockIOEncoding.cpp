#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/PriorityWorklist.h"

namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir {
namespace triton {
namespace gpu::intel {

#define DEBUG_TYPE "tritongpu-optimize-block-encoding"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

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

static Type getNewType(Type type, Attribute encoding) {
  RankedTensorType tensorType = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

static Type getNewPointerType(Type type, Attribute encoding) {
  assert(isa<PointerType>(type) && "expected a ptr type!");
  auto oldPointerType = cast<PointerType>(type);
  return PointerType::get(getNewType(oldPointerType.getPointeeType(), encoding),
                          oldPointerType.getAddressSpace());
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

void rewriteTensorLayoutsForOp(Attribute encoding, Operation *op) {
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
          auto cvt = builder.create<ConvertLayoutOp>(loadOp.getLoc(),
                                                     result.getType(), result);
          result.setType(newTensorTy);

          result.replaceAllUsesExcept(cvt.getResult(), cvt.getOperation());
        }
      }
    }
  }
}

Operation *propagatePointerType(Attribute encoding, Operation *op) {
  OpBuilder builder(op);

  llvm::errs() << "Rewriting op " << *op << "\n\tto use new layout:\n"
               << encoding << "\n\n";

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

} // namespace

#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEBLOCKIOENCODINGPASS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

class TritonIntelGPUOptimizeBlockIOEncodingPass
    : public impl::TritonIntelGPUOptimizeBlockIOEncodingPassBase<
          TritonIntelGPUOptimizeBlockIOEncodingPass> {

  void getSubgroup2DBlockLayoutForOperand(
      Value operand, DpasEncodingAttr dpasLayout,
      llvm::MapVector<Operation *, Attribute> &layoutMap) {
    auto isCandidateLoad = [](Value v) -> LoadOp {
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

    // get the MakeTensorPtr Op for the load
    Value ptr = loadOp.getPtr();
    assert(isTensorPointerType(ptr.getType()) && "expecting pointer to tensor");
    MakeTensorPtrOp makeTensorPtrOp = getMakeTensorPtrOp(ptr);
    assert(makeTensorPtrOp &&
           "expecting a tensor pointer parent to block io load "
           "with tensor pointer type");

    auto oldTensorPtrType = cast<PointerType>(makeTensorPtrOp.getType());
    auto oldTensorType =
        cast<RankedTensorType>(oldTensorPtrType.getPointeeType());
    // Note: we need the old layout to get the order for the load, but it is not
    // clear the layout will always be Blocked. Is there a better way to get
    // this info?
    auto oldLayout = cast<BlockedEncodingAttr>(oldTensorType.getEncoding());

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
    LLVM_DEBUG(DBGS() << "Generated new encoding: " << subgroup2DBlockEncoding
                      << " for op : " << loadOp << "\n");

    layoutMap[loadOp] = subgroup2DBlockEncoding;
  }

public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Step 1. Find all loads which are candidates for conversion to Subgroup 2D
    // Block Encoding. To be a candidate load, a load must be consumed by a Dot
    // Op and the load operand must be a block ptr (produced by a MakeTensorPtr
    // Op). Currently we look for loads with the "block_io" attribute but we
    // could consider moving that logic to this pass later. We place the load
    // and the candidate encoding into the layout map for propagation in step 2
    llvm::MapVector<Operation *, Attribute> layoutMap;
    m.walk([&](DotOp dotOp) {
      auto dotOpType = cast<RankedTensorType>(dotOp.getResult().getType());
      auto dpasLayout = dyn_cast<DpasEncodingAttr>(dotOpType.getEncoding());
      if (!dpasLayout)
        return;

      getSubgroup2DBlockLayoutForOperand(dotOp.getA(), dpasLayout, layoutMap);
      getSubgroup2DBlockLayoutForOperand(dotOp.getB(), dpasLayout, layoutMap);
    });

    // Step 2. Rewrite MakeTensorPtr to use the new layout and propagate the
    // change through the def-use chain, terminating at the Load Op. We add a
    // ConvertLayout Op after the Load Op to convert back to the original
    // layout. Subgroup2DBlockEncoding layouts will be chosen as anchor layouts
    // in RemoveLayoutConversions, and a subsequent run of
    // RemoveLayoutConversions after this pass cleans up intermediate layout
    // conversions and removes the original Load Op encoding.
    for (auto &kv : layoutMap) {
      rewriteTensorLayoutsForOp(kv.second, kv.first);
    }
  }
};

} // namespace gpu::intel
} // namespace triton
} // namespace mlir
