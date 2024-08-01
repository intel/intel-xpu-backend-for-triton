#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_INTELDECOMPOSEUNSUPPORTEDCONVERSIONS
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// pass ws related named attrs.
static void addAttrs(Operation *op, ArrayRef<NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    op->setAttr(attr.getName(), attr.getValue());
}

struct DecomposeUnsupportedConversions
    : public triton::gpu::intel::impl::IntelDecomposeUnsupportedConversionsBase<
          DecomposeUnsupportedConversions> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    /* ---------------- */
    // Convert Fp8E4B15
    /* ---------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      if (!isa<Float8E4M3B11FNUZType, Float8E4M3FNType>(
              getElementTypeOrSelf(cvtOp)))
        return;
      auto shape = cast<RankedTensorType>(cvtOp.getType()).getShape();
      auto argEncoding =
          cast<RankedTensorType>(cvtOp.getSrc().getType()).getEncoding();
      auto cvtEncoding = cast<RankedTensorType>(cvtOp.getType()).getEncoding();
      if (isa<triton::gpu::DotOperandEncodingAttr>(argEncoding) ||
          isa<triton::gpu::DotOperandEncodingAttr>(cvtEncoding))
        return;
      auto F16Ty = builder.getF16Type();

      auto newArgType = RankedTensorType::get(shape, F16Ty, argEncoding);
      auto newCvtType = RankedTensorType::get(shape, F16Ty, cvtEncoding);
      auto newArg = builder.create<triton::FpToFpOp>(cvtOp.getLoc(), newArgType,
                                                     cvtOp.getSrc());
      addAttrs(newArg, cvtOp->getAttrs());
      auto newCvt = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), newCvtType, newArg);
      addAttrs(newCvt, cvtOp->getAttrs());
      auto newRet = builder.create<triton::FpToFpOp>(
          cvtOp.getLoc(), cvtOp.getType(), newCvt.getResult());
      newRet.setRounding(
          triton::RoundingMode::RTNE); // Downcast requires rounding mode
      addAttrs(newRet, cvtOp->getAttrs());
      cvtOp.replaceAllUsesWith(newRet.getResult());
      cvtOp.erase();
    });
    /* -------------------------------- */
    // Replace `splat -> shared
    // with `splat -> blocked -> shared
    /* -------------------------------- */
    mod.walk([&](triton::SplatOp splatOp) -> void {
      auto dstType = cast<RankedTensorType>(splatOp.getType());
      auto shared =
          dyn_cast<triton::gpu::SharedEncodingAttr>(dstType.getEncoding());
      if (shared) {
        OpBuilder builder(splatOp);
        SmallVector<unsigned, 4> sizePerThread(dstType.getRank(), 1);
        auto newType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), dstType.getShape(), sizePerThread,
                getOrder(shared), numWarps, threadsPerWarp, numCTAs));
        auto newSplat = builder.create<triton::SplatOp>(
            splatOp.getLoc(), newType, splatOp.getSrc());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            splatOp.getLoc(), dstType, newSplat.getResult());
        splatOp.replaceAllUsesWith(newConvert.getResult());
        splatOp.erase();
      }
    });
    /* -------------------------------- */
    // Replace `dpas -> dot_op` with `dpas -> shared -> dot_op`
    // unless certain conditions are met
    /* -------------------------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcDpas =
          dyn_cast<triton::gpu::intel::DpasEncodingAttr>(srcType.getEncoding());
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (srcDpas && dstDotOp &&
          !triton::gpu::intel::isDpasToDotShortcut(srcType, dstType)) {
        auto sharedMemorySpace =
            triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
        auto tmpType = triton::MemDescType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                triton::gpu::getOrder(srcDpas),
                triton::gpu::getCTALayout(srcDpas), srcType.getElementType()),
            sharedMemorySpace);
        auto tmp = builder.create<triton::gpu::LocalAllocOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getSrc());
        auto newConvert = builder.create<triton::gpu::LocalLoadOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
    /* -------------------------------- */
    // Replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    /* -------------------------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcBlocked =
          dyn_cast<triton::gpu::BlockedEncodingAttr>(srcType.getEncoding());
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (srcBlocked && dstDotOp) {
        Attribute sharedMemorySpace =
            triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
        auto tmpType = triton::MemDescType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                srcBlocked.getOrder(), srcBlocked.getCTALayout(),
                srcType.getElementType()),
            sharedMemorySpace);
        auto tmp = builder.create<triton::gpu::LocalAllocOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getSrc());
        auto newConvert = builder.create<triton::gpu::LocalLoadOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }
};

} // namespace
