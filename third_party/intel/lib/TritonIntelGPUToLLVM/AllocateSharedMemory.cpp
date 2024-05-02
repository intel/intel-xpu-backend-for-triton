#include "intel/include/TritonIntelGPUToLLVM/Passes.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"

using namespace mlir;

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

#define GEN_PASS_DEF_INTELALLOCATESHAREDMEMORY
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir

namespace {

struct AllocateSharedMemory
    : public triton::gpu::intel::impl::IntelAllocateSharedMemoryBase<
          AllocateSharedMemory> {
  using IntelAllocateSharedMemoryBase<
      AllocateSharedMemory>::IntelAllocateSharedMemoryBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    ModuleAllocation allocation(mod);

    mod.walk([&](FunctionOpInterface funcOp) {
      if (allocation.isRoot(funcOp) && allocation.getSharedMemorySize()) {
        LLVM::LLVMPointerType ptrTy = LLVM::LLVMPointerType::get(
            ctx, triton::TritonGEN::TritonGENMemorySpace::kWorkgroup);
        funcOp.insertArgument(funcOp.getNumArguments(), ptrTy, {},
                              funcOp.getLoc());
      }
      funcOp.walk([&](Operation *op) {
        auto *funcAllocation = allocation.getFuncData(funcOp);
        auto oBufferId = funcAllocation->getBufferId(op);
        int offset = -1;
        if (oBufferId != Allocation::InvalidBufferId)
          offset = funcAllocation->getOffset(oBufferId);
        else if (op->getNumResults() == 1) {
          Value value = op->getResult(0);
          auto vBufferId = funcAllocation->getBufferId(value);
          if (vBufferId != Allocation::InvalidBufferId)
            offset = funcAllocation->getOffset(vBufferId);
        }
        if (offset == -1)
          return;
        op->setAttr("allocation.offset",
                    IntegerAttr::get(IntegerType::get(ctx, 32), offset));
      });
    });
    mod->setAttr("triton_gpu.shared",
                 IntegerAttr::get(IntegerType::get(ctx, 32),
                                  allocation.getSharedMemorySize()));
  }
};

} // namespace
