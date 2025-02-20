#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

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
    bool usePoison =
        (mod->getAttrOfType<IntegerAttr>("ttg.shared").getInt() == 0);

    OpBuilder builder(&getContext());
    mod.walk([&](FunctionOpInterface funcOp) {
      funcOp.walk([&](LLVM::AddressOfOp addressOp) {
        updateStackptr(funcOp, addressOp, builder, usePoison);
      });
    });
  }

private:
  void updateStackptr(FunctionOpInterface funcOp, LLVM::AddressOfOp addressOp,
                      OpBuilder &builder, bool usePoison) {
    MLIRContext *ctx = funcOp.getContext();
    LLVM::LLVMPointerType ptrTy =
        ptr_ty(ctx, mlir::triton::TritonGEN::TritonGENMemorySpace::kWorkgroup);
    Value newValue;
    if (addressOp.getGlobalName() != "global_smem") {
      return;
    }
    if (usePoison) {
      builder.setInsertionPoint(addressOp);
      newValue = builder.create<LLVM::PoisonOp>(addressOp.getLoc(), ptrTy);
    } else {
      newValue = funcOp.getArgument(funcOp.getNumArguments() - 1);
    }
    addressOp.replaceAllUsesWith(newValue);
    addressOp.erase();
  }
};

} // anonymous namespace
