//===- TritonGENToLLVMPass.cpp - TritonGEN to LLVM dialect conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/TritonGENToLLVM/TritonGENToLLVMPass.h"
#include "intel/include/TritonGENToLLVM/GenIntrinsics.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Intrinsics.h"
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

  LLVM::LLVMFuncOp funcOp =
      LLVM::lookupOrCreateFn(moduleOp, funcName, argTypes, retType);
  funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  if (convergent)
    funcOp.setPassthroughAttr(convergentAttr);

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

static unsigned getNumOperandsPerDword(TritonGEN::PrecisionType pTy) {
  switch (pTy) {
  case TritonGEN::PrecisionType::TF32:
    return 1;
  case TritonGEN::PrecisionType::BF16:
  case TritonGEN::PrecisionType::FP16:
    return 2;
  case TritonGEN::PrecisionType::U8:
  case TritonGEN::PrecisionType::S8:
    return 4;
  }
  llvm_unreachable("unsupported TritonGEN::PrecisionType");
}

static LLVM::CallOp createGenISADPAS(TritonGEN::MatrixDPASOp op,
                                     ConversionPatternRewriter &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *context = rewriter.getContext();
  Type resType = op->getResultTypes()[0];
  TypeRange opTypes = op->getOperandTypes();
  Location loc = op->getLoc();

  IntegerType int1Ty = rewriter.getIntegerType(1);
  IntegerType int16Ty = rewriter.getIntegerType(16);
  IntegerType int32Ty = rewriter.getIntegerType(32);

  TritonGEN::PrecisionType precisionA = op.getPa();
  Type packedAType;
  if (precisionA == TritonGEN::PrecisionType::TF32) {
    packedAType = int32Ty;
  } else {
    packedAType = int16Ty;
  }

  Value a = op.getA();
  VectorType aOrigTy = cast<VectorType>(a.getType());
  unsigned bitWidth = aOrigTy.getNumElements() *
                      aOrigTy.getElementType().getIntOrFloatBitWidth();
  VectorType aTy = VectorType::get(
      bitWidth / packedAType.getIntOrFloatBitWidth(), packedAType);
  if (aOrigTy != aTy)
    a = rewriter.create<LLVM::BitcastOp>(loc, aTy, a);

  Value b = op.getB();

  VectorType bOrigTy = cast<VectorType>(b.getType());
  bitWidth = bOrigTy.getNumElements() *
             bOrigTy.getElementType().getIntOrFloatBitWidth();
  VectorType bTy = VectorType::get(bitWidth / 32, int32Ty);
  if (bOrigTy != bTy)
    b = rewriter.create<LLVM::BitcastOp>(loc, bTy, b);

  // FIXME: Use the OpenCL API also for TF32.
  if (precisionA != TritonGEN::PrecisionType::TF32) {
    std::string fnName =
        "intel_sub_group_" + stringifyPrecisionType(precisionA).str() + "_" +
        stringifyPrecisionType(op.getPb()).str() + "_matrix_mad_k" +
        std::to_string(8 /*systolic depth*/ *
                       getNumOperandsPerDword(precisionA));
    SmallVector<Type> argTypes{aTy, bTy, opTypes[0]};
    SmallVector<Value> args{a, b, op.getC()};
    return createDeviceFunctionCall(rewriter, fnName, resType, argTypes, args,
                                    true /*convergent*/);
  }

  llvm::LLVMContext llvmContext;
  LLVM::TypeToLLVMIRTranslator typeTranslator(llvmContext);
  auto llvmResTy = typeTranslator.translateType(resType);
  auto llvmCTy = typeTranslator.translateType(opTypes[0]);
  auto llvmATy = typeTranslator.translateType(aTy);
  auto llvmBTy = typeTranslator.translateType(bTy);
  SmallVector<llvm::Type *> llvmTypes{llvmResTy, llvmCTy, llvmATy, llvmBTy};
  std::string funcName = llvm::GenISAIntrinsic::getName(
      llvm::GenISAIntrinsic::GenISA_sub_group_dpas, llvmTypes);

  SmallVector<Type> argTypes{opTypes[0], aTy,     bTy,     int32Ty,
                             int32Ty,    int32Ty, int32Ty, int1Ty};
  LLVM::LLVMFuncOp funcOp =
      LLVM::lookupOrCreateFn(moduleOp, funcName, argTypes, resType);
  funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);

  auto precA = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
                                                 static_cast<int>(op.getPa()));
  auto precB = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
                                                 static_cast<int>(op.getPa()));
  auto sysDepth =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, 8 /* systolic depth */);
  auto RC = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getRc());
  auto False = rewriter.create<LLVM::ConstantOp>(loc, int1Ty, false);
  SmallVector<Value> args{op.getC(), a, b, precA, precB, sysDepth, RC, False};
  return rewriter.create<LLVM::CallOp>(loc, funcOp, args);
}

static LLVM::CallOp
createGenISA2DBlockRead(TritonGEN::Matrix2DBlockLoadOp op,
                        ConversionPatternRewriter &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *context = rewriter.getContext();
  Type resType = op->getResultTypes()[0];
  Location loc = op->getLoc();

  Value ptr = op.getPtr();
  Value baseWidth = op.getBaseWidth();
  Value baseHeight = op.getBaseHeight();
  Value basePitch = op.getBasePitch();
  Value x = op.getX();
  Value y = op.getY();

  llvm::LLVMContext llvmContext;
  LLVM::TypeToLLVMIRTranslator typeTranslator(llvmContext);
  auto llvmResTy = typeTranslator.translateType(resType);
  SmallVector<llvm::Type *> llvmTypes{llvmResTy};
  std::string funcName = llvm::GenISAIntrinsic::getName(
      llvm::GenISAIntrinsic::GenISA_LSC2DBlockRead, llvmTypes);

  IntegerType int1Ty = rewriter.getIntegerType(1);
  IntegerType int32Ty = rewriter.getIntegerType(32);
  IntegerType int64Ty = rewriter.getIntegerType(64);

  // The IGC intrinsic requires the first argument be int64
  ptr = rewriter.create<LLVM::PtrToIntOp>(loc, int64Ty, ptr);

  SmallVector<Type> argTypes{int64Ty,
                             baseWidth.getType(),
                             baseHeight.getType(),
                             x.getType(),
                             y.getType(),
                             int32Ty,
                             int32Ty,
                             int32Ty,
                             int32Ty,
                             int1Ty,
                             int1Ty,
                             int32Ty};

  LLVM::LLVMFuncOp funcOp =
      LLVM::lookupOrCreateFn(moduleOp, funcName, argTypes, resType);
  funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);

  auto elemSize =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getElemSizeInBits());
  auto tileWidth =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getTileWidth());
  auto tileHeight =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getTileHeight());
  auto vBlocks =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getVBlocks());
  auto useTranspose =
      rewriter.create<LLVM::ConstantOp>(loc, int1Ty, op.getTranspose());
  auto vnniTransform =
      rewriter.create<LLVM::ConstantOp>(loc, int1Ty, op.getVnniTransform());
  // FIXME: Add argument to control cache.
  auto cache = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, 0);

  SmallVector<Value> args{ptr,     baseWidth,    baseHeight,    x,
                          y,       elemSize,     tileWidth,     tileHeight,
                          vBlocks, useTranspose, vnniTransform, cache};
  return rewriter.create<LLVM::CallOp>(loc, funcOp, args);
}

static LLVM::CallOp
createGenISA2DBlockWrite(TritonGEN::Matrix2DBlockStoreOp op,
                         ConversionPatternRewriter &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *context = rewriter.getContext();
  Location loc = op->getLoc();

  Value ptr = op.getPtr();
  Value baseWidth = op.getBaseWidth();
  Value baseHeight = op.getBaseHeight();
  Value basePitch = op.getBasePitch();
  Value x = op.getX();
  Value y = op.getY();
  Value storeVal = op.getStoredVal();

  llvm::LLVMContext llvmContext;
  LLVM::TypeToLLVMIRTranslator typeTranslator(llvmContext);
  auto storeTy = typeTranslator.translateType(storeVal.getType());
  SmallVector<llvm::Type *> llvmTypes{storeTy};
  std::string funcName = llvm::GenISAIntrinsic::getName(
      llvm::GenISAIntrinsic::GenISA_LSC2DBlockWrite, llvmTypes);

  IntegerType int1Ty = rewriter.getIntegerType(1);
  IntegerType int32Ty = rewriter.getIntegerType(32);
  IntegerType int64Ty = rewriter.getIntegerType(64);

  // The IGC intrinsic requires the first argument be int64
  ptr = rewriter.create<LLVM::PtrToIntOp>(loc, int64Ty, ptr);

  SmallVector<Type> argTypes{int64Ty,
                             baseWidth.getType(),
                             baseHeight.getType(),
                             x.getType(),
                             y.getType(),
                             int32Ty,
                             int32Ty,
                             int32Ty,
                             int32Ty,
                             int1Ty,
                             int1Ty,
                             int32Ty,
                             storeVal.getType()};

  LLVM::LLVMFuncOp funcOp = LLVM::lookupOrCreateFn(
      moduleOp, funcName, argTypes, LLVM::LLVMVoidType::get(context));
  funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);

  auto elemSize =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getElemSizeInBits());
  auto tileWidth =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getTileWidth());
  auto tileHeight =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getTileHeight());
  auto vBlocks =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getVBlocks());
  auto useTranspose =
      rewriter.create<LLVM::ConstantOp>(loc, int1Ty, op.getTranspose());
  auto vnniTransform =
      rewriter.create<LLVM::ConstantOp>(loc, int1Ty, op.getVnniTransform());
  // FIXME: Add argument to control cache.
  auto cache = rewriter.create<LLVM::ConstantOp>(loc, int32Ty, 0);

  SmallVector<Value> args{ptr,     baseWidth,    baseHeight,    x,
                          y,       elemSize,     tileWidth,     tileHeight,
                          vBlocks, useTranspose, vnniTransform, cache,
                          storeVal};
  return rewriter.create<LLVM::CallOp>(loc, funcOp, args);
}

static LLVM::CallOp
createGenISA2DBlockPrefetch(TritonGEN::Matrix2DBlockPrefetchOp op,
                            ConversionPatternRewriter &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *context = rewriter.getContext();
  Location loc = op->getLoc();

  Value ptr = op.getPtr();
  Value baseWidth = op.getBaseWidth();
  Value baseHeight = op.getBaseHeight();
  Value basePitch = op.getBasePitch();
  Value x = op.getX();
  Value y = op.getY();

  const StringLiteral funcName = "llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid";
  IntegerType int1Ty = rewriter.getIntegerType(1);
  IntegerType int32Ty = rewriter.getIntegerType(32);
  IntegerType int64Ty = rewriter.getIntegerType(64);

  // The IGC intrinsic requires the first argument be int64
  ptr = rewriter.create<LLVM::PtrToIntOp>(loc, int64Ty, ptr);

  SmallVector<Type> argTypes{int64Ty,
                             baseWidth.getType(),
                             baseHeight.getType(),
                             x.getType(),
                             y.getType(),
                             int32Ty,
                             int32Ty,
                             int32Ty,
                             int32Ty,
                             int1Ty,
                             int1Ty,
                             int32Ty};

  LLVM::LLVMFuncOp funcOp = LLVM::lookupOrCreateFn(
      moduleOp, funcName, argTypes, LLVM::LLVMVoidType::get(context));
  funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);

  auto elemSize =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getElemSizeInBits());
  auto tileWidth =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getTileWidth());
  auto tileHeight =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getTileHeight());
  auto vBlocks =
      rewriter.create<LLVM::ConstantOp>(loc, int32Ty, op.getVBlocks());
  auto useTranspose =
      rewriter.create<LLVM::ConstantOp>(loc, int1Ty, op.getTranspose());
  auto vnniTransform =
      rewriter.create<LLVM::ConstantOp>(loc, int1Ty, op.getVnniTransform());
  auto cache = rewriter.create<LLVM::ConstantOp>(
      loc, int32Ty, static_cast<int>(op.getCacheControl()));

  SmallVector<Value> args{ptr,     baseWidth,    baseHeight,    x,
                          y,       elemSize,     tileWidth,     tileHeight,
                          vBlocks, useTranspose, vnniTransform, cache};
  return rewriter.create<LLVM::CallOp>(loc, funcOp, args);
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
// SubgroupID Op Lowering
//===----------------------------------------------------------------------===//

struct TritonGENSubgroupIdLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SubgroupIdOp> {
  using ConvertOpToLLVMPattern<TritonGEN::SubgroupIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SubgroupIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto retType = rewriter.getIntegerType(32);
    LLVM::CallOp callOp = createDeviceFunctionCall(
        rewriter, "_Z25__spirv_BuiltInSubgroupIdv", retType, {}, {});
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

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

struct TritonSubGroupShuffleLowering
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

//===----------------------------------------------------------------------===//
// Matrix operations
//===----------------------------------------------------------------------===//

struct TritonMatrixDPASLowering
    : public ConvertOpToLLVMPattern<TritonGEN::MatrixDPASOp> {
  using ConvertOpToLLVMPattern<TritonGEN::MatrixDPASOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::MatrixDPASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::CallOp callOp = createGenISADPAS(op, rewriter);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct TritonMatrix2DBlockLoadLowering
    : public ConvertOpToLLVMPattern<TritonGEN::Matrix2DBlockLoadOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::Matrix2DBlockLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::Matrix2DBlockLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::CallOp callOp = createGenISA2DBlockRead(op, rewriter);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct TritonMatrix2DBlockStoreLowering
    : public ConvertOpToLLVMPattern<TritonGEN::Matrix2DBlockStoreOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::Matrix2DBlockStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::Matrix2DBlockStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::CallOp callOp = createGenISA2DBlockWrite(op, rewriter);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct TritonMatrix2DBlockPrefetchLowering
    : public ConvertOpToLLVMPattern<TritonGEN::Matrix2DBlockPrefetchOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::Matrix2DBlockPrefetchOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::Matrix2DBlockPrefetchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::CallOp callOp = createGenISA2DBlockPrefetch(op, rewriter);
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
  patterns
      .add<TritonGENThreadIdXLowering, TritonGENThreadIdYLowering,
           TritonGENThreadIdZLowering, TritonGENBlockIdXLowering,
           TritonGENBlockIdYLowering, TritonGENBlockIdZLowering,
           TritonGENBlockDimXLowering, TritonGENBlockDimYLowering,
           TritonGENBlockDimZLowering, TritonGENGridDimXLowering,
           TritonGENGridDimYLowering, TritonGENGridDimZLowering,
           TritonGENSubgroupIdLowering, TritonGENBarrierLowering,
           TritonSubGroupShuffleLowering, TritonMatrixDPASLowering,
           TritonMatrix2DBlockLoadLowering, TritonMatrix2DBlockStoreLowering,
           TritonMatrix2DBlockPrefetchLowering>(converter);
}

void registerConvertTritonTritonGENToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, TritonGEN::TritonGENDialect *dialect) {
        dialect->addInterfaces<TritonGENToLLVMDialectInterface>();
      });
}
