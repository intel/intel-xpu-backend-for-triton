//===-TritonGENToLLVMIRTranslation.cpp - TritonGEN Dialect to LLVM IR -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the TritonGEN dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "Target/LLVMIR/Dialect/TritonGEN/TritonGENToLLVMIRTranslation.h"

#include "Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"

namespace {
using namespace mlir;
class TritonGENDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    // Skip the attribute if it is not a TritonGEN attribute.
    if (!attribute.getName().getValue().starts_with("triton_gen"))
      return success();

    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc =
        moduleTranslation.lookupFunction(cast<LLVM::LLVMFuncOp>(op).getName());
    if (isKernel(op))
      amendKernel(llvmContext, llvmFunc, attribute);
    return success();
  }

private:
  // Checks if the given operation is a kernel function.
  bool isKernel(Operation *op) const {
    auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
    return fn && fn.getCConv() == LLVM::CConv::SPIR_KERNEL;
  }

  // The attribute is converted into metadata and added to the function.
  void amendKernel(llvm::LLVMContext &llvmContext, llvm::Function *llvmFunc,
                   NamedAttribute attribute) const {
    StringRef name = attribute.getName().getValue();
    assert((name == triton::TritonGEN::TritonGENDialect::
                        getMaxWorkGroupSizeAttrName() ||
            name == triton::TritonGEN::TritonGENDialect::
                        getReqdWorkGroupSizeAttrName() ||
            name == triton::TritonGEN::TritonGENDialect::
                        getReqdSubGroupSizeAttrName()) &&
           "Unexpected attribute");
    SmallVector<llvm::Metadata *, 3> metadata;
    llvm::Type *i64 = llvm::IntegerType::get(llvmContext, 64);
    for (int64_t i :
         extractFromIntegerArrayAttr<int64_t>(attribute.getValue())) {
      llvm::Constant *constant = llvm::ConstantInt::get(i64, i);
      metadata.push_back(llvm::ConstantAsMetadata::get(constant));
    }
    llvm::MDNode *node = llvm::MDNode::get(llvmContext, metadata);
    llvmFunc->setMetadata(name.drop_front(11), node);
  }
};
} // namespace

namespace mlir {
void registerTritonGENDialectTranslation(DialectRegistry &registry) {
  registry.insert<triton::TritonGEN::TritonGENDialect>();
  registry.addExtension(
      +[](MLIRContext *ctx, triton::TritonGEN::TritonGENDialect *dialect) {
        dialect->addInterfaces<TritonGENDialectLLVMIRTranslationInterface>();
      });
}

void registerTritonGENDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerTritonGENDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
} // namespace mlir
