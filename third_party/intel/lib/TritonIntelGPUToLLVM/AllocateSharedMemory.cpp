#include "intel/include/Analysis/Allocation.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_INTELALLOCATESHAREDMEMORY
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {
struct AllocateSharedMemory
    : public triton::gpu::intel::impl::IntelAllocateSharedMemoryBase<
          AllocateSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(
        mod, ::mlir::triton::intel::allocationAnalysisScratchSizeFn);
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};
} // namespace
