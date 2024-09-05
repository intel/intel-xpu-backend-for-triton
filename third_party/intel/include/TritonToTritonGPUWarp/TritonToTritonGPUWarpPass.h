#ifndef TRITON_CONVERSION_TRITONTOTRITONGPUWARP_TRITONTOTRITONGPUWARPPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONGPUWARP_TRITONTOTRITONGPUWARPPASS_H

#include <memory>

namespace mlir {

constexpr static char AttrWorkloadName[] = "triton_gpu.workload";
enum class Workload {
  // TODO: add more
  None = 0, // pattern not match any of below
  ElementWise = 1,
  Reduction = 2,
  Gemm = 3,
  Attention = 4
};

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
