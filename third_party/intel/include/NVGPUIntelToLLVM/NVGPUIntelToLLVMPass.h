#ifndef TRITON_CONVERSION_NVGPUINTEL_TO_LLVM_PASS_H
#define TRITON_CONVERSION_NVGPUINTEL_TO_LLVM_PASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace intel {

#define GEN_PASS_DECL
#include "NVGPUIntelToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertNVGPUIntelToLLVMPass();

} // namespace intel
} // namespace triton

} // namespace mlir

#endif
