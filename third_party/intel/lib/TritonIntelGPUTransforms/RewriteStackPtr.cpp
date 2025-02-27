#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/raw_ostream.h"

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
    MLIRContext *ctx = &getContext();
    ModuleAllocation allocation(mod);
    LLVM::LLVMPointerType ptrTy =
        ptr_ty(ctx, mlir::triton::TritonGEN::TritonGENMemorySpace::kWorkgroup);

    auto globalSmem = mod.lookupSymbol<LLVM::GlobalOp>("global_smem");
    if (!globalSmem) {
      return;
    }
    bool usePoison =
        (mod->getAttrOfType<IntegerAttr>("ttg.shared").getInt() == 0);

    OpBuilder builder(ctx);
    mod.walk([&](FunctionOpInterface funcOp) {
      if (!usePoison && allocation.isRoot(funcOp)) {
        auto oldFuncType =
            dyn_cast<LLVM::LLVMFunctionType>(funcOp.getFunctionType());
        Type resultType = oldFuncType.getReturnType();
        SmallVector<Type> newArgTypes(oldFuncType.getParams().begin(),
                                      oldFuncType.getParams().end());
        newArgTypes.push_back(ptrTy);
        auto newFuncType = LLVM::LLVMFunctionType::get(resultType, newArgTypes);
        funcOp.setType(newFuncType);
        auto llvmFunc = cast<LLVM::LLVMFuncOp>(funcOp.getOperation());
        Block &entryBlock = llvmFunc.getBody().front();
        entryBlock.addArgument(ptrTy, funcOp.getLoc());
      }
      funcOp.walk([&](LLVM::AddressOfOp addressOp) {
        updateStackptr(funcOp, addressOp, builder, usePoison, ptrTy);
      });
    });
  }

private:
  void updateStackptr(FunctionOpInterface funcOp, LLVM::AddressOfOp addressOp,
                      OpBuilder &builder, bool usePoison,
                      LLVM::LLVMPointerType ptrTy) {
    Value newValue;
    if (addressOp.getGlobalName() != "global_smem") {
      return;
    }
    if (usePoison) {
      OpBuilder::InsertionGuard guard(builder);
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
