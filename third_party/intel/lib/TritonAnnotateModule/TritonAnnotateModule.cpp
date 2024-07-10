#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/TritonAnnotateModule/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONANNOTATEMODULE
#include "intel/include/TritonAnnotateModule/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
using namespace mlir::triton::gpu;

namespace {

struct TritonAnnotateModule
    : intel::impl::TritonAnnotateModuleBase<TritonAnnotateModule> {
  using Base::Base;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    Builder builder(mod);

    if (target.getValue().empty()) {
      mod.emitError("Expecting target specification");
      return signalPassFailure();
    }

    MLIRContext *ctx = &getContext();
    mod->setAttr(intel::TritonIntelGPUDialect::getTargetAttrName(),
                 builder.getStringAttr(target.getValue()));

    // FIXME: Use SYCL runtime to query supported OpenCL extensions, instead
    // of checking driver version.
    if (isLTS)
      mod->setAttr(intel::TritonIntelGPUDialect::getLTSAttrName(),
                   builder.getUnitAttr());

    std::string AttrNumThreadsPerWarp =
        TritonGPUDialect::getThreadsPerWarpAttrName();

    mod.walk([&](FunctionOpInterface funcOp) {
      using DPASAnalysis = intel::DPASAnalysis;
      DPASAnalysis dpasAnalysis(funcOp);
      DPASAnalysis::Result canUseDPAS = dpasAnalysis.canUseDPAS();

      if (canUseDPAS == DPASAnalysis::Result::Maybe) {
        // Set the threads per warp attribute to allow dot operation to be
        // lowered to DPAS instructions.
        unsigned reqThreadsPerWarp = DPASAnalysis::supportedThreadsPerWarp(
            triton::gpu::intel::getDeviceArch(mod));
        mod->setAttr(AttrNumThreadsPerWarp,
                     builder.getI32IntegerAttr(reqThreadsPerWarp));
        assert(dpasAnalysis.canUseDPAS() == DPASAnalysis::Result::True &&
               "DPASAnalysis should report that dot operations can be "
               "lowered to DPAS instructions");
        WalkResult::interrupt();
      }
    });

    // If the threads per warp attribute was not set then use the option
    // value.
    if (!mod->getAttr(AttrNumThreadsPerWarp))
      mod->setAttr(AttrNumThreadsPerWarp,
                   builder.getI32IntegerAttr(threadsPerWarp));
  }
};

} // namespace
