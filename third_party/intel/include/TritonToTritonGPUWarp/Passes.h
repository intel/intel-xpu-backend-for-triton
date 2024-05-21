#ifndef TRITON_CONVERSION_TRITONTOTRITONGPUWARP_PASSES_H
#define TRITON_CONVERSION_TRITONTOTRITONGPUWARP_PASSES_H

#include "intel/include/TritonToTritonGPUWarp/TritonToTritonGPUWarpPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "intel/include/TritonToTritonGPUWarp/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
