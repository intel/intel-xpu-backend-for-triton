#ifndef TRITON_CONVERSION_TRITONTOTRITONGPUWARP_TRITONTOTRITONGPUWARPPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONGPUWARP_TRITONTOTRITONGPUWARPPASS_H

#include <memory>
#include <optional>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUWarpPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUWarpPass(unsigned numWarps);

} // namespace triton
} // namespace mlir

#endif