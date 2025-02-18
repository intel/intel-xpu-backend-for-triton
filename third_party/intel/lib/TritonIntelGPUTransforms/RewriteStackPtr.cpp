#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Visitors.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUREWRITESTACKPTR
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

struct TritonIntelGPURewriteStackPtrPass
    : public triton::gpu::intel::impl::TritonIntelGPURewriteStackPtrBase<
          TritonIntelGPURewriteStackPtrPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPURewriteStackPtrBase<
      TritonIntelGPURewriteStackPtrPass>::TritonIntelGPURewriteStackPtrBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    auto globalSmem = mod.lookupSymbol<LLVM::GlobalOp>("global_smem");
    if (!globalSmem) {
      return;
    }

    IntegerAttr sharedAttr = mod->getAttrOfType<IntegerAttr>("ttg.shared");
    if (!sharedAttr) {
      return;
    }
    bool usePoison = (sharedAttr.getInt() == 0);

    mod.walk([&](FunctionOpInterface funcOp) {
      updatestackptr(funcOp, globalSmem, usePoison);
    });
  }

private:
  void updatestackptr(FunctionOpInterface funcOp, LLVM::GlobalOp globalSmem,
                      bool usePoison) {
    MLIRContext *ctx = funcOp.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(
        ctx, TritonGEN::TritonGENMemorySpace::kWorkgroup);

    funcOp.walk([&](LLVM::AddressOfOp addrOp) {
      OpBuilder builder(addrOp);
      Value newValue;
      if (usePoison) {
        newValue = builder.create<LLVM::PoisonOp>(addrOp.getLoc(), ptrTy);
      } else {
        newValue = funcOp.getArgument(funcOp.getNumArguments() - 1);
      }
      addrOp.replaceAllUsesWith(newValue);
      addrOp.erase();
    });
  }
};

} // anonymous namespace
