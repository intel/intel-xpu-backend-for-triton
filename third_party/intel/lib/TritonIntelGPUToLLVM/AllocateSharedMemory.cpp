#include "mlir/Pass/Pass.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_INTELALLOCATESHAREDMEMORY
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct AllocateSharedMemory
    : public mlir::triton::impl::IntelAllocateSharedMemoryBase<
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
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        allocation.getSharedMemorySize()));
  }
};

} // namespace

namespace mlir {

namespace triton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>> createIntelAllocateSharedMemoryPass() {
  return std::make_unique<AllocateSharedMemory>();
}

} // namespace gpu

} // namespace triton

} // namespace mlir
