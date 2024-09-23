
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Analysis/Allocation.h"

using namespace mlir;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_INTELALLOCATESHAREDMEMORY
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

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
    int32_t initialSharedMemorySize = 0;
    if (IntegerAttr sharedAttr =
            mod->getAttrOfType<IntegerAttr>("triton_gpu.shared"))
      initialSharedMemorySize = sharedAttr.getInt();
    mod->setAttr("triton_gpu.shared",
                 IntegerAttr::get(IntegerType::get(ctx, 32),
                                  initialSharedMemorySize +
                                      allocation.getSharedMemorySize()));
  }
};

} // namespace
