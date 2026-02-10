#include "intel/include/Analysis/Allocation.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

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
    MLIRContext *ctx = &getContext();
    ModuleAllocation allocation(
        mod, ::mlir::triton::intel::allocationAnalysisScratchSizeFn);

    mod.walk<mlir::WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
      auto *funcAllocation = allocation.getFuncData(funcOp);
      funcOp.walk([&](Operation *op) {
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
      return WalkResult::skip();
    });
    mod->setAttr("ttg.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        allocation.getSharedMemorySize()));
  }
};
} // namespace
