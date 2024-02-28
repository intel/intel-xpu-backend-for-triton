//===- GENToLLVMPass.cpp - GEN to LLVM dialect conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/GENToLLVM/GENToLLVMPass.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/TypeSwitch.h"

#include "lib/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/GEN/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTGENTOLLVM
#include "intel/include/GENToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static LLVM::CallOp createDeviceFunctionCall(
    ConversionPatternRewriter &rewriter, StringRef funcName, Type retType,
    ArrayRef<Type> argTypes, ArrayRef<Value> args, bool convergent = false) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *context = rewriter.getContext();
  Location loc = UnknownLoc::get(context);

  auto getOrCreateFunction = [&](StringRef funcName) {
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(funcOp);

    auto funcType = LLVM::LLVMFunctionType::get(retType, argTypes);
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto func = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
    func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);

    if (convergent)
      assert(false && "Need to set the convergent attribute");

    return func;
  };

  LLVM::LLVMFuncOp funcOp = getOrCreateFunction(funcName);
  auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
  if (convergent)
    assert(false && "Need to set the convergent attribute");

  return callOp;
}

static LLVM::CallOp createSubGroupShuffle(ConversionPatternRewriter &rewriter,
                                          Value value, Value mask,
                                          GEN::ShflKind kind) {
  assert(isa<IntegerType>(mask.getType()) &&
         cast<IntegerType>(mask.getType()).isInteger(32) &&
         "Expecting mask type to be i32");

  std::string fnName = "";
  switch (kind) {
  case GEN::ShflKind::XOR:
    fnName = "_Z21sub_group_shuffle_xor";
    break;
  case GEN::ShflKind::UP:
    fnName = "_Z20sub_group_shuffle_up";
    break;
  case GEN::ShflKind::DOWN:
    fnName = "_Z22sub_group_shuffle_down";
    break;
  case GEN::ShflKind::IDX:
    fnName = "_Z17sub_group_shuffle";
    break;
  }

  TypeSwitch<Type>(value.getType())
      .Case<Float16Type>([&](auto) { fnName += "Dh"; })
      .Case<Float32Type>([&](auto) { fnName += "f"; })
      .Case<Float64Type>([&](auto) { fnName += "d"; })
      .Case<IntegerType>([&](auto ty) {
        switch (ty.getWidth()) {
        case 8:
          fnName += "c";
          break;
        case 16:
          fnName += "s";
          break;
        case 32:
          fnName += "i";
          break;
        case 64:
          fnName += "l";
          break;
        default:
          llvm_unreachable("unhandled integer type");
        }
      });

  fnName += "j";

  return createDeviceFunctionCall(rewriter, fnName, value.getType(),
                                  {value.getType(), mask.getType()},
                                  {value, mask}, true /*convergent*/);
}

namespace {

struct FuncCallLowering {
protected:
  LogicalResult rewrite(Operation *op, StringRef funcName, unsigned dim,
                        ConversionPatternRewriter &rewriter) const {
    auto retType = rewriter.getIntegerType(64);
    auto argType = rewriter.getIntegerType(32);
    auto arg = LLVM::createConstantI32(op->getLoc(), rewriter, dim);
    createDeviceFunctionCall(rewriter, funcName, retType, {argType}, {arg});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ThreadId Ops Lowerings
//===----------------------------------------------------------------------===//

struct GENThreadIdLowering : public FuncCallLowering {
protected:
  const StringRef funcName = "_Z12get_local_idj";
};

struct GENThreadIdXLowering : public ConvertOpToLLVMPattern<GEN::ThreadIdXOp>,
                              public GENThreadIdLowering {
  using ConvertOpToLLVMPattern<GEN::ThreadIdXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::ThreadIdXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENThreadIdLowering::rewrite(op, funcName, 0, rewriter);
  }
};

struct GENThreadIdYLowering : public ConvertOpToLLVMPattern<GEN::ThreadIdYOp>,
                              public GENThreadIdLowering {
  using ConvertOpToLLVMPattern<GEN::ThreadIdYOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::ThreadIdYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENThreadIdLowering::rewrite(op, funcName, 1, rewriter);
  }
};

struct GENThreadIdZLowering : public ConvertOpToLLVMPattern<GEN::ThreadIdZOp>,
                              public GENThreadIdLowering {
  using ConvertOpToLLVMPattern<GEN::ThreadIdZOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::ThreadIdZOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENThreadIdLowering::rewrite(op, funcName, 2, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// BlockId Ops Lowerings
//===----------------------------------------------------------------------===//

struct GENBlockIdLowering : public FuncCallLowering {
protected:
  const StringRef funcName = "_Z12get_group_idj";
};

struct GENBlockIdXLowering : public ConvertOpToLLVMPattern<GEN::BlockIdXOp>,
                             public GENBlockIdLowering {
  using ConvertOpToLLVMPattern<GEN::BlockIdXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::BlockIdXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENBlockIdLowering::rewrite(op, funcName, 0, rewriter);
  }
};

struct GENBlockIdYLowering : public ConvertOpToLLVMPattern<GEN::BlockIdYOp>,
                             public GENBlockIdLowering {
  using ConvertOpToLLVMPattern<GEN::BlockIdYOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::BlockIdYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENBlockIdLowering::rewrite(op, funcName, 1, rewriter);
  }
};

struct GENBlockIdZLowering : public ConvertOpToLLVMPattern<GEN::BlockIdZOp>,
                             public GENBlockIdLowering {
  using ConvertOpToLLVMPattern<GEN::BlockIdZOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::BlockIdZOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENBlockIdLowering::rewrite(op, funcName, 2, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// BlockDim Ops Lowerings
//===----------------------------------------------------------------------===//

struct GENBlockDimLowering : public FuncCallLowering {
protected:
  const StringRef funcName = "_Z14get_local_sizej";
};

struct GENBlockDimXLowering : public ConvertOpToLLVMPattern<GEN::BlockDimXOp>,
                              public GENBlockDimLowering {
  using ConvertOpToLLVMPattern<GEN::BlockDimXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::BlockDimXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENBlockDimLowering::rewrite(op, funcName, 0, rewriter);
  }
};

struct GENBlockDimYLowering : public ConvertOpToLLVMPattern<GEN::BlockDimYOp>,
                              public GENBlockDimLowering {
  using ConvertOpToLLVMPattern<GEN::BlockDimYOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::BlockDimYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENBlockDimLowering::rewrite(op, funcName, 1, rewriter);
  }
};

struct GENBlockDimZLowering : public ConvertOpToLLVMPattern<GEN::BlockDimZOp>,
                              public GENBlockDimLowering {
  using ConvertOpToLLVMPattern<GEN::BlockDimZOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::BlockDimZOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENBlockDimLowering::rewrite(op, funcName, 2, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// GridDim Ops Lowerings
//===----------------------------------------------------------------------===//

struct GENGridDimLowering : public FuncCallLowering {
protected:
  const StringRef funcName = "_Z14get_num_groupsj";
};

struct GENGridDimXLowering : public ConvertOpToLLVMPattern<GEN::GridDimXOp>,
                             public GENGridDimLowering {
  using ConvertOpToLLVMPattern<GEN::GridDimXOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::GridDimXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENGridDimLowering::rewrite(op, funcName, 0, rewriter);
  }
};

struct GENGridDimYLowering : public ConvertOpToLLVMPattern<GEN::GridDimYOp>,
                             public GENGridDimLowering {
  using ConvertOpToLLVMPattern<GEN::GridDimYOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::GridDimYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENGridDimLowering::rewrite(op, funcName, 1, rewriter);
  }
};

struct GENGridDimZLowering : public ConvertOpToLLVMPattern<GEN::GridDimZOp>,
                             public GENGridDimLowering {
  using ConvertOpToLLVMPattern<GEN::GridDimZOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::GridDimZOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return GENGridDimLowering::rewrite(op, funcName, 2, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Synchronization Ops Lowerings
//===----------------------------------------------------------------------===//

struct GENBarrierLowering : public ConvertOpToLLVMPattern<GEN::BarrierOp> {
  using ConvertOpToLLVMPattern<GEN::BarrierOp>::ConvertOpToLLVMPattern;

  enum MemFence {
    Local = 0x01,
    Global = 0x02,
  };

  LogicalResult
  matchAndRewrite(GEN::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto argType = rewriter.getIntegerType(32);
    auto arg = LLVM::createConstantI32(op->getLoc(), rewriter,
                                       MemFence::Local | MemFence::Global);
    createDeviceFunctionCall(rewriter, funcName, {}, {argType}, {arg},
                             true /*convergent*/);
    return success();
  }

private:
  const StringRef funcName = "_Z7barrierj";
};

struct SubGroupShuffleLowering
    : public ConvertOpToLLVMPattern<GEN::SubGroupShuffleOp> {
  using ConvertOpToLLVMPattern<GEN::SubGroupShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::SubGroupShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value val = op.getValue();
    Value mask = op.getMask();
    GEN::ShflKind kind = op.getKind();

    createSubGroupShuffle(rewriter, val, mask, kind);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertGENToLLVM
    : public triton::impl::ConvertGENToLLVMBase<ConvertGENToLLVM> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet pattern(context);
    LowerToLLVMOptions options(context);
    LLVMTypeConverter converter(context, options);
    LLVMConversionTarget target(*context);

    populateGENToLLVMConversionPatterns(converter, pattern);

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(pattern))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert GEN to LLVM.
struct GENToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateGENToLLVMConversionPatterns(typeConverter, patterns);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population and Registration
//===----------------------------------------------------------------------===//

void mlir::triton::populateGENToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<GENThreadIdXLowering, GENThreadIdYLowering, GENThreadIdZLowering,
               GENBlockIdXLowering, GENBlockIdYLowering, GENBlockIdZLowering,
               GENBlockDimXLowering, GENBlockDimYLowering, GENBlockDimZLowering,
               GENGridDimXLowering, GENGridDimYLowering, GENGridDimZLowering>(
      converter);
  // clang-format on
  patterns.add<GENBarrierLowering>(converter);
}

void registerConvertGENToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, GEN::GENDialect *dialect) {
    dialect->addInterfaces<GENToLLVMDialectInterface>();
  });
}
