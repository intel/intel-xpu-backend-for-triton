#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/TritonAnnotateModule/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"

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

    if (support2DBlockIO)
      mod->setAttr(ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName(),
                   builder.getUnitAttr());

    if (supportDPAS)
      mod->setAttr(ttgi::TritonIntelGPUDialect::getSupportDPASAttrName(),
                   builder.getUnitAttr());

    if (supportDPASWithBF8)
      mod->setAttr(ttgi::TritonIntelGPUDialect::getSupportDPASWithBF8AttrName(),
                   builder.getUnitAttr());

    if (supportBlockScaleDPAS)
      mod->setAttr(
          ttgi::TritonIntelGPUDialect::getSupportBlockScaleDPASAttrName(),
          builder.getUnitAttr());

    if (supportBF16Conversion)
      mod->setAttr(
          ttgi::TritonIntelGPUDialect::getSupportBFloat16ConversionAttrName(),
          builder.getUnitAttr());

    if (supportF4Conversion)
      mod->setAttr(
          ttgi::TritonIntelGPUDialect::getSupportF4ConversionAttrName(),
          builder.getUnitAttr());

    if (supportF8Conversion)
      mod->setAttr(
          ttgi::TritonIntelGPUDialect::getSupportF8ConversionAttrName(),
          builder.getUnitAttr());

    mod->setAttr(ttgi::TritonIntelGPUDialect::getTargetArchAttrName(),
                 builder.getStringAttr(targetArch));

    if (support16BitAtomics)
      mod->setAttr(
          ttgi::TritonIntelGPUDialect::getSupport16BitAtomicsAttrName(),
          builder.getUnitAttr());

    if (supportPrefetch256Bytes)
      mod->setAttr(
          ttgi::TritonIntelGPUDialect::getSupportPrefetch256BAttrName(),
          builder.getUnitAttr());

    if (supportBfloat16Arithmetic)
      mod->setAttr(
          ttgi::TritonIntelGPUDialect::getSupportBFloat16ArithmeticAttrName(),
          builder.getUnitAttr());

    setThreadsPerWarp(mod);
  }

private:
  void setThreadsPerWarp(ModuleOp &mod) const {
    Builder builder(mod);

    // If the threads per warp attribute was not set, use the option value.
    if (!mod->hasAttr(ttg::AttrNumThreadsPerWarp))
      mod->setAttr(ttg::AttrNumThreadsPerWarp,
                   builder.getI32IntegerAttr(threadsPerWarp));
  }
};

} // namespace
