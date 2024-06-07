#ifndef TRITON_CONVERSION_TRITONTOTRITONGPUWARP_PASSES_H
#define TRITON_CONVERSION_TRITONTOTRITONGPUWARP_PASSES_H

#include "intel/include/TritonToTritonGPUWarp/TritonToTritonGPUWarpPass.h"

namespace mlir::triton::intel {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "intel/include/TritonToTritonGPUWarp/Passes.h.inc"

} // namespace mlir::triton::intel

#endif
