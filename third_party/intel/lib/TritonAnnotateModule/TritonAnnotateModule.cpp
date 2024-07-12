#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/TritonAnnotateModule/Passes.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONANNOTATEMODULE
#include "intel/include/TritonAnnotateModule/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
using namespace mlir::triton::gpu;
using DPASAnalysis = intel::DPASAnalysis;

namespace {

struct TritonAnnotateModule
    : intel::impl::TritonAnnotateModuleBase<TritonAnnotateModule> {
  using Base::Base;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    Builder builder(mod);

    mod->setAttr(intel::TritonIntelGPUDialect::getMinSGSizeAttrName(),
                 builder.getI32IntegerAttr(minSGSize));

    if (supportSG2DBlock)
      mod->setAttr(intel::TritonIntelGPUDialect::getSupportSG2DBlockAttrName(),
                   builder.getUnitAttr());

    if (supportDPAS)
      mod->setAttr(intel::TritonIntelGPUDialect::getSupportDPASAttrName(),
                   builder.getUnitAttr());

    if (supportBF16Conversion)
      mod->setAttr(
          intel::TritonIntelGPUDialect::getSupportBF16ConversionAttrName(),
          builder.getUnitAttr());

    DPASAnalysis &dpasAnalysis = getAnalysis<DPASAnalysis>();
    setThreadsPerWarp(mod, dpasAnalysis);
  }

private:
  void setThreadsPerWarp(ModuleOp &mod,
                         const DPASAnalysis &dpasAnalysis) const {
    Builder builder(mod);
    const std::string &AttrNumThreadsPerWarp =
        TritonGPUDialect::getThreadsPerWarpAttrName();

    //    mod.walk([&](FunctionOpInterface funcOp) {
    //      // FIXME: DPAS lowering only implemented for 16 threads per warp,
    //      i.e.,
    //      // DPAS is not used for devices like ATS.
    //      constexpr unsigned supportedThreadsPerWarp = 16;
    //      if (minSGSize != supportedThreadsPerWarp)
    //        return WalkResult::interrupt();
    //
    //      if (dpasAnalysis.canUseDPAS(funcOp) == DPASAnalysis::Result::Maybe)
    //      {
    //        // Set the threads per warp attribute to allow dot operation to be
    //        // lowered to DPAS instructions.
    //        mod->setAttr(AttrNumThreadsPerWarp,
    //                     builder.getI32IntegerAttr(minSGSize));
    //        assert(dpasAnalysis.canUseDPAS(funcOp) ==
    //        DPASAnalysis::Result::True &&
    //               "DPASAnalysis should report that dot operations can be "
    //               "lowered to DPAS instructions");
    //        return WalkResult::interrupt();
    //      }
    //      return WalkResult::advance();
    //    });

    // If the threads per warp attribute was not set, use the option value.
    if (!mod->hasAttr(AttrNumThreadsPerWarp))
      mod->setAttr(AttrNumThreadsPerWarp,
                   builder.getI32IntegerAttr(threadsPerWarp));
  }
};

} // namespace
