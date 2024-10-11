#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUREDUCEDATADUPLICATION
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace ttgi = mlir::triton::gpu::intel;
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

class TritonIntelGPUReduceDataDuplicationPass
    : public intel::impl::TritonIntelGPUReduceDataDuplicationBase<
          TritonIntelGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      auto srcOrder = triton::gpu::getOrder(srcEncoding);
      auto rank = srcOrder.size();
      if (isa<triton::gpu::SharedEncodingAttr>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
      if (auto srcMmaEncoding =
              dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(srcEncoding)) {

        if (srcMmaEncoding.getVersionMajor() != 2 ||
            (srcMmaEncoding.getWarpsPerCTA()[1] == 1 &&
             dstDotOp.getParent() == srcMmaEncoding))
          return;
      }
      if (auto srcMfmaEncoding =
              dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(srcEncoding)) {

        if (srcMfmaEncoding.getWarpsPerCTA()[1] == 1 &&
            srcMfmaEncoding.getIsTransposed() &&
            dstDotOp.getParent() == srcMfmaEncoding)
          return;
      }
      if (auto srcDpasEncoding =
              dyn_cast<triton::gpu::intel::DpasEncodingAttr>(srcEncoding)) {
        unsigned opIdx = dstDotOp.getOpIdx();
        if ((opIdx == 0 /* Operand A */ &&
             dstDotOp.getParent() == srcDpasEncoding &&
             srcDpasEncoding.getWarpsPerCTA()[rank - 1] ==
                 1 /* No parallel on N dim */) ||
            (opIdx == 1 /* Operand B */ &&
             dstDotOp.getParent() == srcDpasEncoding &&
             srcDpasEncoding.getWarpsPerCTA()[rank - 2] ==
                 1 /* No parallel on M dim */))
          /* The destination dot layout has no duplication. */
          return;
      }
      SmallVector<unsigned> sharedOrder;
      if (rank == 3) {
        // add all elements except the element that is zero
        for (unsigned i = 0; i < rank; ++i)
          if (srcOrder[i] != 0)
            sharedOrder.emplace_back(srcOrder[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = std::move(srcOrder);
      }
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto tmpType = triton::MemDescType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(), sharedOrder,
              triton::gpu::getCTALayout(srcEncoding), srcType.getElementType()),
          sharedMemorySpace);
      auto tmp = builder.create<triton::gpu::LocalAllocOp>(
          cvtOp.getLoc(), tmpType, cvtOp.getSrc());
      auto newConvert = builder.create<triton::gpu::LocalLoadOp>(cvtOp.getLoc(),
                                                                 dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

} // namespace
