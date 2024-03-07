//===- TritonGENToLLVMPass.cpp - TritonGEN to LLVM dialect conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/TritonGENToLLVM/TritonGENToLLVMPass.h"

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
#include "llvm/Support/ErrorHandling.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGENTOLLVM
#include "intel/include/TritonGENToLLVM/Passes.h.inc"
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
  auto convergentAttr =
      rewriter.getArrayAttr(StringAttr::get(context, "convergent"));

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
      func.setPassthroughAttr(convergentAttr);

    return func;
  };

  LLVM::LLVMFuncOp funcOp = getOrCreateFunction(funcName);
  auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
  if (convergent)
    callOp->setAttr("passthrough", convergentAttr);

  return callOp;
}

static LLVM::CallOp createSubGroupShuffle(ConversionPatternRewriter &rewriter,
                                          Value value, Value mask,
                                          TritonGEN::ShflKind kind) {
  assert(isa<IntegerType>(mask.getType()) &&
         cast<IntegerType>(mask.getType()).isInteger(32) &&
         "Expecting mask type to be i32");

  std::string fnName = "";
  switch (kind) {
  case TritonGEN::ShflKind::XOR:
    fnName = "_Z21sub_group_shuffle_xor";
    break;
  case TritonGEN::ShflKind::UP:
    fnName = "_Z20sub_group_shuffle_up";
    break;
  case TritonGEN::ShflKind::DOWN:
    fnName = "_Z22sub_group_shuffle_down";
    break;
  case TritonGEN::ShflKind::IDX:
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
  Value rewrite(Operation *op, StringRef funcName, unsigned dim,
                ConversionPatternRewriter &rewriter) const {
    auto retType = rewriter.getIntegerType(64);
    auto argType = rewriter.getIntegerType(32);
    auto arg = LLVM::createConstantI32(op->getLoc(), rewriter, dim);
    LLVM::CallOp callOp =
        createDeviceFunctionCall(rewriter, funcName, retType, {argType}, {arg});

    Type resType = op->getResult(0).getType();
    if (resType == callOp.getResult().getType())
      return callOp.getResult();

    return rewriter.create<LLVM::TruncOp>(op->getLoc(), resType,
                                          callOp.getResult());
  }
};

//===----------------------------------------------------------------------===//
// ThreadId Ops Lowerings
//===----------------------------------------------------------------------===//

template <typename SourceOp>
struct TritonGENThreadIdLowering : public ConvertOpToLLVMPattern<SourceOp>,
                                   public FuncCallLowering {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value res;
    if (isa<TritonGEN::ThreadIdXOp>(op))
      res = rewrite(op, "_Z12get_local_idj", 0, rewriter);
    else if (isa<TritonGEN::ThreadIdYOp>(op))
      res = rewrite(op, "_Z12get_local_idj", 1, rewriter);
    else if (isa<TritonGEN::ThreadIdZOp>(op))
      res = rewrite(op, "_Z12get_local_idj", 2, rewriter);
    else
      llvm_unreachable("Unexpected operation");

    rewriter.replaceOp(op, res);
    return success();
  }
};

using TritonGENThreadIdXLowering =
    TritonGENThreadIdLowering<TritonGEN::ThreadIdXOp>;
using TritonGENThreadIdYLowering =
    TritonGENThreadIdLowering<TritonGEN::ThreadIdYOp>;
using TritonGENThreadIdZLowering =
    TritonGENThreadIdLowering<TritonGEN::ThreadIdZOp>;

//===----------------------------------------------------------------------===//
// BlockId Ops Lowerings
//===----------------------------------------------------------------------===//

template <typename SourceOp>
struct TritonGENBlockIdLowering : public ConvertOpToLLVMPattern<SourceOp>,
                                  public FuncCallLowering {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value res;
    if (isa<TritonGEN::BlockIdXOp>(op))
      res = rewrite(op, "_Z12get_group_idj", 0, rewriter);
    else if (isa<TritonGEN::BlockIdYOp>(op))
      res = rewrite(op, "_Z12get_group_idj", 1, rewriter);
    else if (isa<TritonGEN::BlockIdZOp>(op))
      res = rewrite(op, "_Z12get_group_idj", 2, rewriter);
    else
      llvm_unreachable("Unexpected operation");

    rewriter.replaceOp(op, res);
    return success();
  }
};

using TritonGENBlockIdXLowering =
    TritonGENBlockIdLowering<TritonGEN::BlockIdXOp>;
using TritonGENBlockIdYLowering =
    TritonGENBlockIdLowering<TritonGEN::BlockIdYOp>;
using TritonGENBlockIdZLowering =
    TritonGENBlockIdLowering<TritonGEN::BlockIdZOp>;

//===----------------------------------------------------------------------===//
// BlockDim Ops Lowerings
//===----------------------------------------------------------------------===//

template <typename SourceOp>
struct TritonGENBlockDimLowering : public ConvertOpToLLVMPattern<SourceOp>,
                                   public FuncCallLowering {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value res;
    if (isa<TritonGEN::BlockDimXOp>(op))
      res = rewrite(op, "_Z14get_local_sizej", 0, rewriter);
    else if (isa<TritonGEN::BlockDimYOp>(op))
      res = rewrite(op, "_Z14get_local_sizej", 1, rewriter);
    else if (isa<TritonGEN::BlockDimZOp>(op))
      res = rewrite(op, "_Z14get_local_sizej", 2, rewriter);
    else
      llvm_unreachable("Unexpected operation");

    rewriter.replaceOp(op, res);
    return success();
  }
};

using TritonGENBlockDimXLowering =
    TritonGENBlockDimLowering<TritonGEN::BlockDimXOp>;
using TritonGENBlockDimYLowering =
    TritonGENBlockDimLowering<TritonGEN::BlockDimYOp>;
using TritonGENBlockDimZLowering =
    TritonGENBlockDimLowering<TritonGEN::BlockDimZOp>;

//===----------------------------------------------------------------------===//
// GridDim Ops Lowerings
//===----------------------------------------------------------------------===//

template <typename SourceOp>
struct TritonGENGridDimLowering : public ConvertOpToLLVMPattern<SourceOp>,
                                  public FuncCallLowering {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value res;
    if (isa<TritonGEN::GridDimXOp>(op))
      res = rewrite(op, "_Z14get_num_groupsj", 0, rewriter);
    else if (isa<TritonGEN::GridDimYOp>(op))
      res = rewrite(op, "_Z14get_num_groupsj", 1, rewriter);
    else if (isa<TritonGEN::GridDimZOp>(op))
      res = rewrite(op, "_Z14get_num_groupsj", 2, rewriter);
    else
      llvm_unreachable("Unexpected operation");

    rewriter.replaceOp(op, res);
    return success();
  }
};

using TritonGENGridDimXLowering =
    TritonGENGridDimLowering<TritonGEN::GridDimXOp>;
using TritonGENGridDimYLowering =
    TritonGENGridDimLowering<TritonGEN::GridDimYOp>;
using TritonGENGridDimZLowering =
    TritonGENGridDimLowering<TritonGEN::GridDimZOp>;

//===----------------------------------------------------------------------===//
// Synchronization Ops Lowerings
//===----------------------------------------------------------------------===//

struct TritonGENBarrierLowering
    : public ConvertOpToLLVMPattern<TritonGEN::BarrierOp> {
  using ConvertOpToLLVMPattern<TritonGEN::BarrierOp>::ConvertOpToLLVMPattern;

  enum MemFence {
    Local = 0x01,
    Global = 0x02,
  };

  LogicalResult
  matchAndRewrite(TritonGEN::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto retType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto argType = rewriter.getIntegerType(32);
    auto arg = LLVM::createConstantI32(op->getLoc(), rewriter, MemFence::Local);
    LLVM::CallOp callOp =
        createDeviceFunctionCall(rewriter, "_Z7barrierj", {retType}, {argType},
                                 {arg}, true /*convergent*/);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct SubGroupShuffleLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SubGroupShuffleOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::SubGroupShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SubGroupShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value val = op.getValue();
    Value mask = op.getMask();
    TritonGEN::ShflKind kind = op.getKind();
    LLVM::CallOp callOp = createSubGroupShuffle(rewriter, val, mask, kind);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertTritonGENToLLVM
    : public triton::impl::ConvertTritonGENToLLVMBase<ConvertTritonGENToLLVM> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet pattern(context);
    LowerToLLVMOptions options(context);
    LLVMTypeConverter converter(context, options);
    LLVMConversionTarget target(*context);

    populateTritonGENToLLVMConversionPatterns(converter, pattern);

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
/// Implement the interface to convert TritonGEN to LLVM.
struct TritonGENToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateTritonGENToLLVMConversionPatterns(typeConverter, patterns);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population and Registration
//===----------------------------------------------------------------------===//

void mlir::triton::populateTritonGENToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<TritonGENThreadIdXLowering, TritonGENThreadIdYLowering,
               TritonGENThreadIdZLowering, TritonGENBlockIdXLowering,
               TritonGENBlockIdYLowering, TritonGENBlockIdZLowering,
               TritonGENBlockDimXLowering, TritonGENBlockDimYLowering,
               TritonGENBlockDimZLowering, TritonGENGridDimXLowering,
               TritonGENGridDimYLowering, TritonGENGridDimZLowering>(converter);
  patterns.add<TritonGENBarrierLowering, SubGroupShuffleLowering>(converter);
}

void registerConvertTritonTritonGENToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, TritonGEN::TritonGENDialect *dialect) {
        dialect->addInterfaces<TritonGENToLLVMDialectInterface>();
      });
}
