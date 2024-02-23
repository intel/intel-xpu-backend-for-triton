#ifndef TRITON_CONVERSION_NVGPUINTEL_TO_LLVM_PASS_H
#define TRITON_CONVERSION_NVGPUINTEL_TO_LLVM_PASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "intel/include/NVGPUIntelToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertNVGPUIntelToLLVMPass();

} // namespace triton

} // namespace mlir

#endif
