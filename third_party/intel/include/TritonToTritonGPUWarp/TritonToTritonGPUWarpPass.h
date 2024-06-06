#ifndef TRITON_CONVERSION_TRITONTOTRITONGPUWARP_TRITONTOTRITONGPUWARPPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONGPUWARP_TRITONTOTRITONGPUWARPPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton::intel {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUWarpPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUWarpPass(unsigned numWarps);

} // namespace triton::intel
} // namespace mlir

#endif
