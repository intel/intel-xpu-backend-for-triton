#ifndef TRITONGPU_CONVERSION_TRITONINTELGPUTOLLVM_PASSES_H
#define TRITONGPU_CONVERSION_TRITONINTELGPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

#define GEN_PASS_DECL
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"

namespace gpu {
std::unique_ptr<OperationPass<ModuleOp>>
createIntelDecomposeUnsupportedConversionsPass();

std::unique_ptr<OperationPass<ModuleOp>> createIntelAllocateSharedMemoryPass();

} // namespace gpu

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonIntelGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonIntelGPUToLLVMPass(int32_t computeCapability);

#define GEN_PASS_REGISTRATION
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
