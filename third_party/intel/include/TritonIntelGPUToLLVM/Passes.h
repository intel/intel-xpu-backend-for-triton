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

enum Target { NVVM, ROCDL, GENX, Default = GENX };

#define GEN_PASS_DECL
#include "TritonIntelGPUToLLVM/Passes.h.inc"

namespace gpu {
std::unique_ptr<OperationPass<ModuleOp>>
createIntelDecomposeUnsupportedConversionsPass();

std::unique_ptr<OperationPass<ModuleOp>> createIntelAllocateSharedMemoryPass();
std::unique_ptr<OperationPass<ModuleOp>> createIntelAllocateSharedMemoryPass(
    const IntelAllocateSharedMemoryOptions &options);

} // namespace gpu

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonIntelGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonIntelGPUToLLVMPass(int32_t computeCapability, Target target);

#define GEN_PASS_REGISTRATION
#include "TritonIntelGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
