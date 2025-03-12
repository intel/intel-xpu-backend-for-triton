#include "intel/include/Analysis/Utility.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"

using namespace mlir;
using namespace mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_INTELDECOMPOSEUNSUPPORTEDCONVERSIONS
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

struct DecomposeUnsupportedConversions
    : public triton::gpu::intel::impl::IntelDecomposeUnsupportedConversionsBase<
          DecomposeUnsupportedConversions> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    triton::gpu::decomposeTensorCoreToDotLayoutConversion(mod,
                                                          isDpasToDotShortcut);
    triton::gpu::decomposeBlockedToDotLayoutConversion(mod);
  }
};

} // namespace
