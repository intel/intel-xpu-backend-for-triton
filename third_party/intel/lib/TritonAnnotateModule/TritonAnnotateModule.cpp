#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/TritonAnnotateModule/Passes.h"
#include <triton/Tools/Sys/GetEnv.hpp>
#include <variant>

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONANNOTATEMODULE
#include "intel/include/TritonAnnotateModule/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace {

struct TritonAnnotateModule
    : ttgi::impl::TritonAnnotateModuleBase<TritonAnnotateModule> {
  using Base::Base;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    Builder builder(mod);

    mod->setAttr(ttgi::TritonIntelGPUDialect::getMinSGSizeAttrName(),
                 builder.getI32IntegerAttr(minSGSize));

    if (supportSG2DBlock)
      mod->setAttr(ttgi::TritonIntelGPUDialect::getSupportSG2DBlockAttrName(),
                   builder.getUnitAttr());

    if (supportDPAS)
      mod->setAttr(ttgi::TritonIntelGPUDialect::getSupportDPASAttrName(),
                   builder.getUnitAttr());

    if (supportDPASWithBF8)
      mod->setAttr(ttgi::TritonIntelGPUDialect::getSupportDPASWithBF8AttrName(),
                   builder.getUnitAttr());

    if (supportBF16Conversion)
      mod->setAttr(
          ttgi::TritonIntelGPUDialect::getSupportBF16ConversionAttrName(),
          builder.getUnitAttr());

    mod->setAttr(ttgi::TritonIntelGPUDialect::getTargetArchAttrName(),
                 builder.getStringAttr(targetArch));

    if (support16BitAtomics)
      mod->setAttr(
          ttgi::TritonIntelGPUDialect::getSupport16BitAtomicsAttrName(),
          builder.getUnitAttr());

    setThreadsPerWarp(mod);
  }

private:
  void setThreadsPerWarp(ModuleOp &mod) const {
    using DPASAnalysisFactory = ttgi::DPASAnalysisFactory;
    using DPASAnalysisResult = ttgi::DPASAnalysisResult;

    Builder builder(mod);
    bool enableWarp32 =
        tt::tools::getBoolEnv("TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32");

    if (!enableWarp32) {
      auto dpasAnalysis = DPASAnalysisFactory::createDPASAnalysis(mod);

      mod.walk([&](FunctionOpInterface funcOp) {
        // DPAS lowering only implemented for 16 threads per warp, i.e., DPAS is
        // not used for devices like ATS.
        constexpr unsigned supportedThreadsPerWarp = 16;
        if (minSGSize != supportedThreadsPerWarp)
          return WalkResult::interrupt();

        if (DPASAnalysisFactory::canUseDPAS(funcOp, dpasAnalysis) ==
            DPASAnalysisResult::Maybe) {
          // Set the threads per warp attribute to allow dot operation to be
          // lowered to DPAS instructions.
          mod->setAttr(ttg::AttrNumThreadsPerWarp,
                       builder.getI32IntegerAttr(minSGSize));
          assert(DPASAnalysisFactory::canUseDPAS(funcOp, dpasAnalysis) ==
                     DPASAnalysisResult::True &&
                 "DPASAnalysis should report that dot operations can be "
                 "lowered to DPAS instructions");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }

    // If the threads per warp attribute was not set, use the option value.
    if (!mod->hasAttr(ttg::AttrNumThreadsPerWarp))
      mod->setAttr(ttg::AttrNumThreadsPerWarp,
                   builder.getI32IntegerAttr(threadsPerWarp));
  }
};

} // namespace
