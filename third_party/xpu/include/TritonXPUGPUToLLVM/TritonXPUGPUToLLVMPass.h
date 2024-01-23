#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_PASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Target/PTX/TmaMetadata.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

enum Target { NVVM, ROCDL, GENX, Default = GENX };

#define GEN_PASS_DECL
#include "TritonXPUGPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonXPUGPUToLLVMPass();

} // namespace triton

} // namespace mlir

#endif
