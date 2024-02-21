#ifndef NVGPUINTEL_CONVERSION_PASSES_H
#define NVGPUINTEL_CONVERSION_PASSES_H

#include "intel/include/NVGPUIntelToLLVM/NVGPUIntelToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace triton {
namespace intel {

#define GEN_PASS_REGISTRATION
#include "intel/include/NVGPUIntelToLLVM/Passes.h.inc"

} // namespace intel
} // namespace triton
} // namespace mlir

#endif
