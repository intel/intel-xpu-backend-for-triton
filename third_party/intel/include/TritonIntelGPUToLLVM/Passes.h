#ifndef TRITONGPU_CONVERSION_TRITONINTELGPUTOLLVM_PASSES_H
#define TRITONGPU_CONVERSION_TRITONINTELGPUTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton::gpu::intel {

#define GEN_PASS_DECL
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"

} // namespace mlir::triton::gpu::intel

#endif // TRITONGPU_CONVERSION_TRITONINTELGPUTOLLVM_PASSES_H
