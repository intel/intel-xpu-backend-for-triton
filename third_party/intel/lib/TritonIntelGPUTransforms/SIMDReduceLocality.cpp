#include <Analysis/Utility.h>
#include <algorithm>
#include <triton/Analysis/Utility.h>
#include <triton/Dialect/TritonGPU/Transforms/Utility.h>

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;
using LinearLayout = mlir::triton::LinearLayout;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUSIMDREDUCELOCALITY
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

class TritonIntelGPUSIMDReduceLocalityPass
    : public mlir::triton::gpu::intel::impl::
          TritonIntelGPUSIMDReduceLocalityBase<
              TritonIntelGPUSIMDReduceLocalityPass> {
public:
  using mlir::triton::gpu::intel::impl::TritonIntelGPUSIMDReduceLocalityBase<
      TritonIntelGPUSIMDReduceLocalityPass>::
      TritonIntelGPUSIMDReduceLocalityBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<tt::ReduceOp> reduceOps;
    mod.walk([&](tt::ReduceOp op) { reduceOps.push_back(op); });
    for (auto op : reduceOps)
      optimizeReduceOp(op);
  }

  void optimizeReduceOp(tt::ReduceOp op) {
    ReduceOpHelper helper(op);
    LinearLayout inputLl = triton::gpu::toLinearLayout(helper.getSrcTy());
    MLIRContext *ctx = op->getContext();
    unsigned axis = op.getAxis();
    // llvm::outs() << "check reduce axis:" << axis << "\n";
    // llvm::outs() << "check reduce input layout:" << inputLl << "\n";
    StringAttr kRegister = StringAttr::get(ctx, "register");
    StringAttr kLane = StringAttr::get(ctx, "lane");
    StringAttr kWarp = StringAttr::get(ctx, "warp");
    StringAttr kBlock = StringAttr::get(ctx, "block");
    StringAttr kDim0 = StringAttr::get(ctx, "dim0");
    StringAttr kDim1 = StringAttr::get(ctx, "dim1");
    auto laneMapping = inputLl.sublayout(
        {kRegister, kLane}, llvm::to_vector(inputLl.getOutDimNames()));
    // llvm::outs() << "check reduce lane mapping:" << laneMapping << "\n";
    auto laneBases = laneMapping.getBases().lookup(kLane);

    unsigned shfitUpSizeLog2 = 0;
    for (size_t i = 0; i < laneMapping.getInDimSizeLog2(kLane); i++) {
      auto laneBaseOnReduceDim = laneBases[i][axis];
      if (laneBaseOnReduceDim != 0) {
        ++shfitUpSizeLog2;
      } else {
        break;
      }
    }
    // llvm::outs() << "check reduce lane shfitUpSizeLog2:" << shfitUpSizeLog2
    // << "\n";

    unsigned shiftDownSizeLog2 = 0;
    for (size_t i = laneMapping.getInDimSizeLog2(kLane); i-- > 0;) {
      auto laneBaseOnReduceDim = laneBases[i][axis];
      if (laneBaseOnReduceDim != 0) {
        ++shiftDownSizeLog2;
      } else {
        break;
      }
    }
    // llvm::outs() << "check reduce lane shiftDownSizeLog2:" <<
    // shiftDownSizeLog2
    //              << "\n";

    std::vector<unsigned> shuffleRegCandidate;
    auto regBases = laneMapping.getBases().lookup(kRegister);
    for (size_t i = 0; i < laneMapping.getInDimSizeLog2(kRegister); i++) {
      auto regBaseOnReduceDim = regBases[i][axis];
      if (!regBaseOnReduceDim) {
        shuffleRegCandidate.push_back(i);
      }
    }
    // unsigned shuffleRegIdx = 0;
    // for (unsigned regCandidate : shuffleRegCandidate) {
    //   llvm::outs() << "shuffleRegCandidate[" << shuffleRegIdx++ << "]:"
    //                << regCandidate << "\n";
    // }

    // clamp the pack size;
    bool packOrUnpack = shfitUpSizeLog2 < shiftDownSizeLog2;
    unsigned packSizeLog2 =
        std::min(std::max(shfitUpSizeLog2, shiftDownSizeLog2),
                 (unsigned)shuffleRegCandidate.size());
    shuffleRegCandidate.resize(packSizeLog2);
    if (packSizeLog2 == 0)
      return;

    // shuffleRegIdx = 0;
    // for (unsigned regCandidate : shuffleRegCandidate) {
    //   llvm::outs() << "shuffleRegCandidate[" << shuffleRegIdx++ << "]:"
    //                << regCandidate << "\n";
    // }

    std::vector<std::vector<int>> regMappingBases;

    unsigned threadsPerWarp = laneMapping.getInDimSize(kLane);
    int laneBase = packOrUnpack ? threadsPerWarp / (1 << packSizeLog2) : 1;
    for (size_t i = 0; i < laneMapping.getInDimSizeLog2(kRegister); i++) {
      if (std::find(shuffleRegCandidate.begin(), shuffleRegCandidate.end(),
                    i) != shuffleRegCandidate.end()) {
        regMappingBases.push_back({0, laneBase});
        laneBase <<= 1;
      } else {
        regMappingBases.push_back({(1 << i), 0});
      }
    }
    std::vector<std::vector<int>> laneMappingBases;
    unsigned regCandidate = 0;
    unsigned maxLaneBase = threadsPerWarp / 2;
    for (size_t i = 0; i < laneMapping.getInDimSizeLog2(kLane); i++) {
      int curLaneBase = 1 << i;
      int shiftedLaneBase = (packOrUnpack ? curLaneBase >> packSizeLog2
                                          : curLaneBase << packSizeLog2);
      // clamp to threadsPerWarp
      shiftedLaneBase = shiftedLaneBase > maxLaneBase ? 0 : shiftedLaneBase;
      if (shiftedLaneBase) {
        laneMappingBases.push_back({0, shiftedLaneBase});
      } else {
        laneMappingBases.push_back(
            {(1 << shuffleRegCandidate[regCandidate++]), 0});
      }
    }

    auto reinterPretCvtMap =
        LinearLayout({{kRegister, regMappingBases}, {kLane, laneMappingBases}},
                     {kRegister, kLane});
    // llvm::outs() << "optimize reduce lane local reinterPretCvtMap:" <<
    // reinterPretCvtMap << "\n";
    reinterPretCvtMap *=
        LinearLayout::identity1D(inputLl.getInDimSize(kWarp), kWarp, kWarp) *
        LinearLayout::identity1D(inputLl.getInDimSize(kBlock), kBlock, kBlock);

    auto newReduceLayout = reinterPretCvtMap.compose(inputLl);
    // llvm::outs() << "optimize reduce lane local composed:" << newReduceLayout
    // << "\n";

    // create the reduce op with new input layout.
    OpBuilder builder(op);

    ttg::LinearEncodingAttr newLayout =
        ttg::LinearEncodingAttr::get(ctx, newReduceLayout);
    auto tensorType = helper.getSrcTy();
    auto reinterpretedTensorType = tensorType.cloneWithEncoding(newLayout);

    if (!ttgi::cvtIsSubGroupReinterpret(tensorType, reinterpretedTensorType)) {
      // if the reinterpret cast is not supported. return.
      return;
    }

    auto newReduceOp = createReduce(builder, op, reinterpretedTensorType);
    auto convertLayoutOp =
        createConvertLayout(builder, op->getResult(0).getType(), newReduceOp);

    op.getResult().replaceAllUsesWith(convertLayoutOp);
    op.erase();
  }

  Operation *createConvertLayout(OpBuilder &builder, Type destType,
                                 Operation *newReduce) const {
    builder.setInsertionPointAfter(newReduce);
    auto newCvt = triton::gpu::ConvertLayoutOp::create(
        builder, newReduce->getLoc(), destType, newReduce->getResult(0));
    return newCvt;
  }

  Operation *createReduce(OpBuilder &builder, triton::ReduceOp reduce,
                          Type viewOpTensorType) const {
    auto dstType = cast<RankedTensorType>(viewOpTensorType);

    builder.setInsertionPointAfter(reduce);
    IRMapping mapping;
    for (auto operand : reduce.getOperands()) {
      auto converted = ttgi::ReinterpretConvertLayoutOp::create(
          builder, reduce.getLoc(), dstType, operand);
      mapping.map(operand, converted);
    }

    auto newReduce = cloneWithInferType(builder, &(*reduce), mapping);
    // auto typeInfer = dyn_cast<InferTypeOpInterface>(newReduce);
    // if (typeInfer) {
    //   SmallVector<Type, 1> newTypes;
    //   auto success = typeInfer.inferReturnTypes(
    //       newReduce->getContext(), newReduce->getLoc(),
    //       newReduce->getOperands(), newReduce->getAttrDictionary(),
    //       newReduce->getPropertiesStorage(), newReduce->getRegions(),
    //       newTypes);
    //   if (succeeded(success)) {
    //     for (size_t i = 0; i < newTypes.size(); i++)
    //       newReduce->getResult(i).setType(newTypes[i]);
    //   }
    // }
    return newReduce;
  }
};

}; // namespace
