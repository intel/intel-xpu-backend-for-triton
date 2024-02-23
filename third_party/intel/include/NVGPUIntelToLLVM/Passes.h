#ifndef NVGPUINTEL_CONVERSION_PASSES_H
#define NVGPUINTEL_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "NVGPUIntelToLLVM/NVGPUIntelToLLVMPass.h"

namespace mlir {
namespace triton {
#define GEN_PASS_REGISTRATION
#include "NVGPUIntelToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
