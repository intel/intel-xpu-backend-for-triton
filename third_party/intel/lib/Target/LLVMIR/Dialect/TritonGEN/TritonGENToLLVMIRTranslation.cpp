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

  constexpr static std::size_t decorationCacheControlArity = 4;
  constexpr static llvm::StringLiteral decorationCacheControlAttrName =
      "triton_gen.DecorationCacheControlINTEL";

  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    StringRef attrName = attribute.getName().getValue();
    if (attrName == decorationCacheControlAttrName) {
      assert(instructions.size() == 1 && "Expecting a single instruction");
      return handleDecorationCacheControl(instructions.front(), attribute,
                                          moduleTranslation);
    }
    if (attrName.starts_with("triton_gen"))
      return handleTritonGenAttr(op, attribute, moduleTranslation);
    return success();
  }

  LogicalResult
  convertOperation(Operation *operation, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    return TypeSwitch<Operation *, LogicalResult>(operation)
        .Case([&moduleTranslation](triton::TritonGEN::CacheControls op) {
          llvm::Value *ptr = moduleTranslation.lookupValue(op.getPtr());
          moduleTranslation.mapValue(op, ptr);
          Builder mlirBuilder(op);
          for (OpOperand &use : op->getUses())
            appendDecoration(mlirBuilder, decorationCacheControlAttrName,
                             use.getOwner(), op.getCacheControls(),
                             use.getOperandNumber());
          return success();
        })
        .Default([](Operation *op) {
          return op->emitError("unsupported TritonGEN operation: ")
                 << op->getName();
        });
  }

private:
  template <typename IntTy>
  static llvm::Metadata *getConstantIntMD(llvm::Type *type, IntTy val) {
    return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(type, val));
  }

  static void appendDecoration(Builder &mlirBuilder, llvm::StringRef mdKey,
                               Operation *op, ArrayAttr newAttrs,
                               unsigned operandNumber) {
    auto attr = op->getAttrOfType<ArrayAttr>(mdKey);
    SmallVector<Attribute> attrs = attr
                                       ? SmallVector<Attribute>{attr.getValue()}
                                       : SmallVector<Attribute>{};
    llvm::transform(
        newAttrs.getValue(), std::back_inserter(attrs),
        [&mlirBuilder, operandNumber](Attribute attr) -> Attribute {
          return TypeSwitch<Attribute, Attribute>(attr)
              .Case<triton::TritonGEN::LoadCacheControlDecorationAttr,
                    triton::TritonGEN::StoreCacheControlDecorationAttr>(
                  [&mlirBuilder, operandNumber](auto attr) {
                    constexpr int32_t loadCacheControlKey = 6442;
                    constexpr int32_t storeCacheControlKey = 6443;
                    constexpr int32_t key =
                        std::is_same_v<
                            decltype(attr),
                            triton::TritonGEN::LoadCacheControlDecorationAttr>
                            ? loadCacheControlKey
                            : storeCacheControlKey;
                    int32_t cacheLevel = attr.getCacheLevel();
                    int32_t cacheControl =
                        static_cast<int32_t>(attr.getCacheControl());
                    return mlirBuilder.getDenseI32ArrayAttr(
                        {key, cacheLevel, cacheControl,
                         static_cast<int32_t>(operandNumber)});
                  });
        });
    op->setAttr(mdKey, mlirBuilder.getArrayAttr(attrs));
  }

  static LogicalResult
  handleDecorationCacheControl(llvm::Instruction *inst,
                               NamedAttribute attribute,
                               LLVM::ModuleTranslation &moduleTranslation) {
    assert(attribute.getName() == decorationCacheControlAttrName &&
           "Expecting decoration cache key");
    auto arrayAttr = cast<ArrayAttr>(attribute.getValue());
    ArrayRef<Attribute> attrs = arrayAttr.getValue();
    SmallVector<llvm::Metadata *> decorations;
    llvm::LLVMContext &ctx = inst->getContext();
    llvm::transform(
        attrs, std::back_inserter(decorations), [&ctx](Attribute attr) {
          auto arrayAttr = cast<DenseI32ArrayAttr>(attr);
          ArrayRef<int> attrs = arrayAttr.asArrayRef();
          assert(attrs.size() == decorationCacheControlArity &&
                 "Invalid decoration cache attribute arity");
          constexpr unsigned numBits = 32;
          llvm::Type *type = llvm::IntegerType::get(ctx, numBits);
          std::array<llvm::Metadata *, decorationCacheControlArity> metadata;
          llvm::transform(attrs, metadata.begin(), [type](int val) {
            return getConstantIntMD(type, val);
          });
          return llvm::MDNode::get(ctx, metadata);
        });
    constexpr static llvm::StringLiteral decorationCacheControlMDName =
        "spirv.DecorationCacheControlINTEL";
    inst->setMetadata(decorationCacheControlMDName,
                      llvm::MDNode::get(ctx, decorations));
    return success();
  }

  LogicalResult
  handleTritonGenAttr(Operation *op, NamedAttribute attribute,
                      LLVM::ModuleTranslation &moduleTranslation) const {
    llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
    llvm::Function *llvmFunc =
        moduleTranslation.lookupFunction(cast<LLVM::LLVMFuncOp>(op).getName());
    if (isKernel(op))
      amendKernel(llvmContext, llvmFunc, attribute);
    return success();
  }

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
