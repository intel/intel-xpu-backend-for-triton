//
// Created by guangyey on 12/28/22.
//

#ifndef TRITON_TRITONGPUTOSPIRVPASS_H
#define TRITON_TRITONGPUTOSPIRVPASS_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"
#include <any>
#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToSPIRVPass(
    const std::map<std::string, std::any> &computeCapability = {});

} // namespace triton

} // namespace mlir

#endif // TRITON_TRITONGPUTOSPIRVPASS_H
