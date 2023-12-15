#ifndef TRITON_CONVERSION_PASSES_H
#define TRITON_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Target/PTX/TmaMetadata.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
