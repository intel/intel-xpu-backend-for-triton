#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <chrono>

using namespace mlir;

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
    LLVM::LLVMPointerType ptrTy =
        ptr_ty(ctx, mlir::triton::TritonGEN::TritonGENMemorySpace::kWorkgroup);

    auto globalSmem = mod.lookupSymbol<LLVM::GlobalOp>("global_smem");
    if (!globalSmem) {
      return;
    }
    CallGraph<Allocation> allocation(mod);
    bool usePoison =
        (mod->getAttrOfType<IntegerAttr>("ttg.shared").getInt() == 0);

    // 1: Process function arguments for root functions
    if (!usePoison) {
      for (auto &root : allocation.getRoots()) {
        insertFuncArguments(root, ptrTy);
      }
    }

    // 2: Collect all AddressOfOp that need updating
    SmallVector<LLVM::AddressOfOp> addressOps;
    mod.walk([&](LLVM::AddressOfOp addressOp) {
      if (addressOp.getGlobalName() != "global_smem")
        return;
      if (usePoison) {
        addressOps.push_back(addressOp);
      } else {
        auto funcOp = addressOp->getParentOfType<FunctionOpInterface>();
        if (funcOp && allocation.isRoot(funcOp)) {
          addressOps.push_back(addressOp);
        }
      }
    });

    // 3: Update collected AddressOfOp
    OpBuilder builder(ctx);
    for (LLVM::AddressOfOp addressOp : addressOps) {
      builder.setInsertionPoint(addressOp);
      Value newValue;
      if (usePoison) {
        newValue = builder.create<LLVM::PoisonOp>(addressOp.getLoc(), ptrTy);
      } else {
        auto funcOp = addressOp->getParentOfType<FunctionOpInterface>();
        assert(funcOp && "AddressOfOp must be inside a function");
        newValue = funcOp.getArgument(funcOp.getNumArguments() - 1);
      }
      addressOp.replaceAllUsesWith(newValue);
      addressOp.erase();
    }
  }

private:
  void insertFuncArguments(FunctionOpInterface funcOp,
                           LLVM::LLVMPointerType ptrTy) {
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
};

} // anonymous namespace
