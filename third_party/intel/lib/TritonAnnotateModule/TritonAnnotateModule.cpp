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
    if (target.getValue().empty()) {
      mod.emitError("Expecting target specification");
      return signalPassFailure();
    }

    Builder builder(mod);
    mod->setAttr(intel::TritonIntelGPUDialect::getTargetAttrName(),
                 builder.getStringAttr(target.getValue()));

    // FIXME: Use SYCL runtime to query supported OpenCL extensions, instead
    // of checking driver version.
    if (isLTS)
      mod->setAttr(intel::TritonIntelGPUDialect::getLTSAttrName(),
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

    auto result = mod.walk([&](FunctionOpInterface funcOp) {
      if (dpasAnalysis.canUseDPAS(funcOp) == DPASAnalysis::Result::Maybe) {
        // Set the threads per warp attribute to allow dot operation to be
        // lowered to DPAS instructions.
        unsigned reqThreadsPerWarp =
            DPASAnalysis::supportedThreadsPerWarp(intel::getDeviceArch(mod));
        mod->setAttr(AttrNumThreadsPerWarp,
                     builder.getI32IntegerAttr(reqThreadsPerWarp));
        assert(dpasAnalysis.canUseDPAS(funcOp) == DPASAnalysis::Result::True &&
               "DPASAnalysis should report that dot operations can be "
               "lowered to DPAS instructions");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // If the threads per warp attribute was not set, use the option value.
    if (!result.wasInterrupted()) {
      assert(!mod->getAttr(AttrNumThreadsPerWarp) && "Unexpected attribute");
      mod->setAttr(AttrNumThreadsPerWarp,
                   builder.getI32IntegerAttr(threadsPerWarp));
    }
  }
};

} // namespace
