#include "LLVMIntr.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

namespace mlir::triton::gpu::intel {

LLVM::CallOp createDeviceFunctionCall(
    RewriterBase &rewriter, StringRef funcName, Type retType,
    ArrayRef<Type> argTypes, ArrayRef<Value> args,
    mlir::ArrayRef<std::pair<unsigned, mlir::StringRef>> paramAttrs,
    const LLVMFuncAttributeOptions &funcAttributeOptions,
    const intel::AttributeList &passthroughAttrs, LLVM::cconv::CConv cc) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *ctx = rewriter.getContext();
  Location loc = UnknownLoc::get(ctx);
  OpBuilder b(ctx);

  LLVM::LLVMFuncOp funcOp =
      LLVM::lookupOrCreateFn(b, moduleOp, funcName, argTypes, retType).value();
  funcOp.setCConv(cc);
  funcOp.setConvergent(funcAttributeOptions.isConvergent);
  funcOp.setNoUnwind(funcAttributeOptions.isNoUnwind);
  funcOp.setWillReturn(funcAttributeOptions.isWillReturn);

  if (funcAttributeOptions.memEffectsAttr)
    funcOp.setMemoryEffectsAttr(funcAttributeOptions.memEffectsAttr);

  for (auto [idx, attrName] : paramAttrs)
    funcOp.setArgAttr(idx, attrName, rewriter.getUnitAttr());

  if (!passthroughAttrs.getFnAttributes().empty())
    funcOp->setAttrs(passthroughAttrs.getFnAttributes().getDictionary(ctx));

  auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
  callOp->setAttrs(funcOp->getAttrs());

  return callOp;
}

} // namespace mlir::triton::gpu::intel
