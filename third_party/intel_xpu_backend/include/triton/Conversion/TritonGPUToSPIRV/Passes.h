#ifndef TRITONGPU_CONVERSION_PASSES_H
#define TRITONGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "triton/Conversion/TritonGPUToSPIRV//TritonGPUToSPIRVPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonGPUToSPIRV/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
