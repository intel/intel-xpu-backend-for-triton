#include "GenIntrinsicHelper.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

// The code convert the function attribute from the original here:
// https://github.com/llvm/llvm-project/blob/e575b7cb7a64297583d6382c16ce264d9fe45d08/mlir/lib/Target/LLVMIR/ModuleImport.cpp#L1547
// List of LLVM IR attributes that map to an explicit attribute on the MLIR
// LLVMFuncOp.
static constexpr std::array ExplicitAttributes{
    StringLiteral("aarch64_pstate_sm_enabled"),
    StringLiteral("aarch64_pstate_sm_body"),
    StringLiteral("aarch64_pstate_sm_compatible"),
    StringLiteral("aarch64_new_za"),
    StringLiteral("aarch64_preserves_za"),
    StringLiteral("aarch64_in_za"),
    StringLiteral("aarch64_out_za"),
    StringLiteral("aarch64_inout_za"),
    StringLiteral("vscale_range"),
    StringLiteral("frame-pointer"),
    StringLiteral("target-features"),
    StringLiteral("unsafe-fp-math"),
    StringLiteral("no-infs-fp-math"),
    StringLiteral("no-nans-fp-math"),
    StringLiteral("approx-func-fp-math"),
    StringLiteral("no-signed-zeros-fp-math"),
};

static void processPassthroughAttrs(llvm::Function *func,
                                    mlir::LLVM::LLVMFuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  SmallVector<Attribute> passthroughs;
  llvm::AttributeSet funcAttrs = func->getAttributes().getAttributes(
      llvm::AttributeList::AttrIndex::FunctionIndex);
  for (llvm::Attribute attr : funcAttrs) {
    // Skip the memory attribute since the LLVMFuncOp has an explicit memory
    // attribute.
    if (attr.hasAttribute(llvm::Attribute::Memory))
      continue;

    // Skip invalid type attributes.
    if (attr.isTypeAttribute()) {
      emitWarning(funcOp.getLoc(),
                  "type attributes on a function are invalid, skipping it");
      continue;
    }

    StringRef attrName;
    if (attr.isStringAttribute())
      attrName = attr.getKindAsString();
    else
      attrName = llvm::Attribute::getNameFromAttrKind(attr.getKindAsEnum());
    auto keyAttr = StringAttr::get(context, attrName);

    // Skip attributes that map to an explicit attribute on the LLVMFuncOp.
    if (llvm::is_contained(ExplicitAttributes, attrName))
      continue;

    if (attr.isStringAttribute()) {
      StringRef val = attr.getValueAsString();
      if (val.empty()) {
        passthroughs.push_back(keyAttr);
        continue;
      }
      passthroughs.push_back(
          ArrayAttr::get(context, {keyAttr, StringAttr::get(context, val)}));
      continue;
    }
    if (attr.isIntAttribute()) {
      auto val = std::to_string(attr.getValueAsInt());
      passthroughs.push_back(
          ArrayAttr::get(context, {keyAttr, StringAttr::get(context, val)}));
      continue;
    }
    if (attr.isEnumAttribute()) {
      passthroughs.push_back(keyAttr);
      continue;
    }

    llvm_unreachable("unexpected attribute kind");
  }

  if (!passthroughs.empty())
    funcOp.setPassthroughAttr(ArrayAttr::get(context, passthroughs));
}

mlir::LLVM::LLVMFuncOp
appendOrGetGenISADeclaration(OpBuilder &builder, llvm::GenISAIntrinsic::ID id,
                             ArrayRef<mlir::Type *> mlirTys) {
  auto mlirContext = builder.getContext();

  SmallVector<llvm::Type *, 4> llvmTys;
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      std::make_unique<llvm::Module>("temp", llvmContext);
  mlir::LLVM::TypeToLLVMIRTranslator llvmToMLIR(llvmContext);
  for (mlir::Type *ty : mlirTys) {
    llvmTys.push_back(llvmToMLIR.translateType(*ty));
  }
  auto llvmFunc =
      llvm::GenISAIntrinsic::getDeclaration(llvmModule.get(), id, llvmTys);

  auto genISAName = llvmFunc->getName();

  auto funcName = StringAttr::get(mlirContext, genISAName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(
      builder.getBlock()
          ->getParent()
          ->getParentOfType<mlir::LLVM::LLVMFuncOp>(),
      funcName);

  if (funcOp)
    return cast<mlir::LLVM::LLVMFuncOp>(*funcOp);

  auto llvmFuncType = llvmFunc->getFunctionType();
  LLVM::TypeFromLLVMIRTranslator mlirFromLLVM(*mlirContext);
  auto mlirFuncTy = mlirFromLLVM.translateType(llvmFuncType);
  mlir::LLVM::LLVMFunctionType funcTy =
      cast<mlir::LLVM::LLVMFunctionType>(mlirFuncTy);

  auto parent = builder.getBlock()
                    ->getParent()
                    ->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  mlir::OpBuilder b(parent);
  auto ret =
      b.create<LLVM::LLVMFuncOp>(mlir::UnknownLoc::get(mlirContext), genISAName,
                                 funcTy, LLVM::Linkage::External,
                                 /*dsoLocal*/ false, LLVM::CConv::C,
                                 /*comdat=*/SymbolRefAttr{});

  processPassthroughAttrs(llvmFunc, ret);

  return ret;
}

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
