#ifndef TRITONXPUGPU_CONVERSION_PASSES_H
#define TRITONXPUGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"

namespace mlir {
class Pass;

namespace triton {

std::unique_ptr<Pass> createConvertTritonXPUGPUToLLVMPass();

// #define GEN_PASS_REGISTRATION
// #include "TritonXPUGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
