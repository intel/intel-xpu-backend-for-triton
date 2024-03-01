#ifndef TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PASSES_H
#define TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h.inc"

namespace gpu {
std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass();

std::unique_ptr<OperationPass<ModuleOp>> createAllocateSharedMemoryPass();

} // namespace gpu

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability);

#define GEN_PASS_REGISTRATION
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
