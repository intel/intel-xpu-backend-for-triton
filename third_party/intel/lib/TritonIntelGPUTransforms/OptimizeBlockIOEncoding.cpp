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

struct EncodingInfo {
  Attribute desiredEncoding;
  bool requiresConvert = false;

  bool operator==(const EncodingInfo &other) const {
    return desiredEncoding == other.desiredEncoding &&
           requiresConvert == other.requiresConvert;
  }
};

/**
 * The algorithm here takes inspiration from
 * TritonNVIDIAGPU::OptimizeDescriptorEncoding. The idea is to iterate the
 * def-use chain in both directions starting from the Load Op. We store the
 * values that need to be updated along with the new encoding in the
 * `valueToEncodingInfo` MapVector. After all value/encoding pairs have been
 * determined, we update the encoding for each value, adding a conversion to
 * the existing Load Op result layout for users of the load.
 */
void rewriteTensorLayoutsForOp(Attribute encoding, Operation *op) {
  auto loadOp = cast<LoadOp>(op);
  auto loadPtrType = cast<PointerType>(loadOp->getOperand(0).getType());
  auto addressSpace = loadPtrType.getAddressSpace();

  llvm::MapVector<TypedValue<PointerType>, EncodingInfo> valueToEncodingInfo;
  llvm::PriorityWorklist<TypedValue<PointerType>> worklist;

  auto updateEncoding = [&](ArrayRef<Value> ptrValues, EncodingInfo info) {
    for (auto value : ptrValues) {
      bool requiresConvert = llvm::any_of(
          value.getUsers(), [](auto user) { return isa<LoadOp>(user); });
      info.requiresConvert = requiresConvert;

      auto typedVal = cast<TypedValue<PointerType>>(value);
      auto itr = valueToEncodingInfo.find(typedVal);
      if (itr == valueToEncodingInfo.end()) {
        LLVM_DEBUG(DBGS() << "Add encoding " << info.desiredEncoding
                          << " for value " << typedVal << "\n");
        valueToEncodingInfo[typedVal] = info;
        worklist.insert(typedVal);
      } else {
        LLVM_DEBUG(DBGS() << "Found existing encoding info "
                          << itr->second.desiredEncoding << " for value "
                          << typedVal << ". Ensure new encoding "
                          << info.desiredEncoding << " matches.\n");
        assert(itr->second == info && "already visited encoding info for "
                                      "value, expected them to be equal!");
        continue;
      }
    }
  };

  worklist.insert(cast<TypedValue<PointerType>>(loadOp->getOperand(0)));

  // 1. Starting from the Load Op, propagate encoding info up and down the
  // def-use chain.
  while (!worklist.empty()) {
    auto crtValue = worklist.pop_back_val();

    // Propagate to users
    for (OpOperand &use : crtValue.getUses()) {
      auto op = use.getOwner();
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        auto offset = 3 * isa<scf::ForOp>(op);
        auto vals = getTiedArgs(op, use.getOperandNumber() - offset);
        updateEncoding(vals, EncodingInfo{encoding});
      } else if (isa<scf::YieldOp>(op)) {
        auto vals = getTiedArgs(op->getParentOp(), use.getOperandNumber());
        updateEncoding(vals, EncodingInfo{encoding});
      }
    }

    // Propagate to defining ops
    if (auto opResult = dyn_cast<OpResult>(crtValue)) {
      auto definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto vals = getTiedArgs(definingOp, opResult.getResultNumber());
        updateEncoding(vals, EncodingInfo{encoding});
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(crtValue)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        auto offset = isa<scf::ForOp>(parentOp);
        auto vals = getTiedArgs(parentOp, blockArg.getArgNumber() - offset);
        updateEncoding(vals, EncodingInfo{encoding});
      }
    }
  }

  // 2. Update the type for each value in-place. Add a ConvertLayout Op after
  // any loads which require conversion to the existing layout for the loaded
  // value.
  for (auto &[val, einfo] : valueToEncodingInfo) {
    Attribute newEncoding = einfo.desiredEncoding;
    LLVM_DEBUG(DBGS() << "Rewrite encoding to " << newEncoding << " for value "
                      << val << "\n");

    PointerType oldType = val.getType();
    auto oldTensorTy = cast<RankedTensorType>(oldType.getPointeeType());
    auto newTensorTy = RankedTensorType::get(
        oldTensorTy.getShape(), oldTensorTy.getElementType(), newEncoding);

    val.setType(PointerType::get(newTensorTy, oldType.getAddressSpace()));
    if (einfo.requiresConvert) {
      for (auto user : val.getUsers()) {
        if (auto loadOp = dyn_cast<LoadOp>(user)) {

          OpBuilder builder(loadOp);
          auto oldLoadType = loadOp.getType();
          Value result = loadOp.getResult();

          builder.setInsertionPointAfter(loadOp);
          auto cvt = builder.create<ConvertLayoutOp>(loadOp.getLoc(),
                                                     result.getType(), result);
          LLVM_DEBUG(DBGS() << "Added convert Op:\n"
                            << cvt << " after Load Op:\n"
                            << loadOp << "\n");
          result.setType(newTensorTy);

          result.replaceAllUsesExcept(cvt.getResult(), cvt.getOperation());
        }
      }
    }
  }
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
      return dyn_cast<LoadOp>(v.getDefiningOp());
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
    if (!isTensorPointerType(ptr.getType())) {
      // TODO: support tensor of pointer loads
      LLVM_DEBUG(DBGS() << "Ptr\n"
                        << ptr << " for Load Op:\n"
                        << loadOp
                        << "\nincompatible with Subgroup 2D Block Layout.\n");
      return;
    }
    MakeTensorPtrOp makeTensorPtrOp = getMakeTensorPtrOp(ptr);

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
        cast<DistributedEncodingTrait>(dotOperandEncoding),
        oldTensorType.getShape(),
        blockIOAttr == StringAttr::get(&getContext(), "row_major"),
        elemSizeInBits / 8, &getContext());
    SmallVector<unsigned> instrShape{tileParams[0], tileParams[1]};
    const unsigned vBlocks = tileParams[2];

    auto subgroup2DBlockEncoding = Subgroup2DBlockEncodingAttr::get(
        &getContext(), dpasLayout.getWarpsPerCTA(), CTALayout, instrShape,
        tileParams[2],
        getOrderForDotOperand(dotOperandEncoding.getOpIdx(), /*rank*/ 2,
                              /*kContig*/ true),
        kWidth, dpasLayout.getThreadsPerWarp());

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
