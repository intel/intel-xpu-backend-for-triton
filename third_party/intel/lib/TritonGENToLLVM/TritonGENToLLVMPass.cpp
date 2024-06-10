//===- TritonGENToLLVMPass.cpp - TritonGEN to LLVM dialect conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Attributes.h"
#include "Utils/Mangling.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ModRef.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/TritonGENToLLVM/GenIntrinsicEnum.h"
#include "intel/include/TritonGENToLLVM/GenIntrinsics.h"
#include "intel/include/TritonGENToLLVM/TritonGENToLLVMPass.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONGENTOLLVM
#include "intel/include/TritonGENToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static intel::AttributeList
getAttrList(const intel::AttrBuilder &funcAttrBuilder,
            ArrayRef<NamedAttrList> paramAttrs = {}) {
  intel::AttributeList attrs;
  attrs.addFnAttributes(funcAttrBuilder);
  if (!paramAttrs.empty())
    attrs.addParamAttributes(paramAttrs);
  return attrs;
}

static LLVM::CallOp
createDeviceFunctionCall(ConversionPatternRewriter &rewriter,
                         StringRef funcName, Type retType,
                         ArrayRef<Type> argTypes, ArrayRef<Value> args,
                         intel::AttributeList &attrs) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *ctx = rewriter.getContext();
  Location loc = UnknownLoc::get(ctx);

  LLVM::LLVMFuncOp funcOp =
      LLVM::lookupOrCreateFn(moduleOp, funcName, argTypes, retType);
  funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  funcOp->setAttrs(attrs.getFnAttributes().getDictionary(ctx));

  for (auto [idx, attrList] : llvm::enumerate(attrs.getParamAttributes())) {
    for (NamedAttribute attr : attrList)
      funcOp.setArgAttr(idx, attr.getName(), attr.getValue());
  }

  auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
  callOp->setAttrs(funcOp->getAttrs());

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
  fnName += intel::getTypeMangling(value.getType()) + "j";

  MLIRContext *ctx = rewriter.getContext();

  intel::AttrBuilder funcAttrBuilder(*ctx);
  funcAttrBuilder.addPassthroughAttribute(llvm::Attribute::Convergent);
  intel::AttributeList attrs = getAttrList(funcAttrBuilder);

  return createDeviceFunctionCall(rewriter, fnName, value.getType(),
                                  {value.getType(), mask.getType()},
                                  {value, mask}, attrs);
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
  default:
    llvm_unreachable("unsupported TritonGEN::PrecisionType");
  }
}

static LLVM::CallOp createGenISADPAS(TritonGEN::MatrixDPASOp op,
                                     ConversionPatternRewriter &rewriter) {
  Type resType = op->getResultTypes()[0];
  TypeRange opTypes = op->getOperandTypes();
  Location loc = op->getLoc();

  FloatType fp32Ty = rewriter.getF32Type();
  IntegerType int16Ty = rewriter.getIntegerType(16);
  IntegerType int32Ty = rewriter.getIntegerType(32);

  TritonGEN::PrecisionType precisionA = op.getPa();
  Type packedAType = (precisionA == TritonGEN::PrecisionType::TF32)
                         ? cast<Type>(fp32Ty)
                         : int16Ty;
  Type packedBType = (precisionA == TritonGEN::PrecisionType::TF32)
                         ? cast<Type>(fp32Ty)
                         : int32Ty;

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
  VectorType bTy = VectorType::get(
      bitWidth / packedBType.getIntOrFloatBitWidth(), packedBType);
  if (bOrigTy != bTy)
    b = rewriter.create<LLVM::BitcastOp>(loc, bTy, b);

  std::string fnName =
      "intel_sub_group_" + stringifyPrecisionType(precisionA).str() + "_" +
      stringifyPrecisionType(op.getPb()).str() + "_matrix_mad_k" +
      std::to_string(8 /*systolic depth*/ * getNumOperandsPerDword(precisionA));
  if (precisionA == TritonGEN::PrecisionType::TF32)
    fnName += "_f32";
  std::string bMangledTy = intel::getTypeMangling(bTy);
  std::string cMangledTy = intel::getTypeMangling(opTypes[0]);
  if (bMangledTy == cMangledTy)
    cMangledTy = "S0_";
  fnName = "_Z" + std::to_string(fnName.size()) + fnName +
           intel::getTypeMangling(aTy) + bMangledTy + cMangledTy;
  SmallVector<Type> argTypes{aTy, bTy, opTypes[0]};
  SmallVector<Value> args{a, b, op.getC()};

  MLIRContext *ctx = rewriter.getContext();
  intel::AttrBuilder funcAttrBuilder(*ctx);
  funcAttrBuilder.addPassthroughAttribute(llvm::Attribute::Convergent);
  intel::AttributeList attrs = getAttrList(funcAttrBuilder);

  return createDeviceFunctionCall(rewriter, fnName, resType, argTypes, args,
                                  attrs);
}

static bool isOCLBuiltinAvailable(TritonGEN::Matrix2DBlockLoadOp op) {
  // intel_sub_group_2d_block_read_32b_8r8x1c is expected to be lowered to
  // llvm.genx.GenISA.LSC2DBlockRead.v4i32, but it is incorrectly lowered to
  // llvm.genx.GenISA.LSC2DBlockRead.v8i32.
  if (op.getElemSizeInBits() == 32 && op.getTileHeight() == 8 &&
      op.getTileWidth() == 8 && op.getVBlocks() == 1)
    return false;

  // Missing intel_sub_group_2d_block_read_32b_8r16x1c and
  // intel_sub_group_2d_block_read_32b_16r16x1c.
  if (op.getElemSizeInBits() == 32 && op.getTileWidth() == 16 &&
      op.getVBlocks() == 1)
    return false;

  // Missing intel_sub_group_2d_block_read_8b_16r32x1c and
  // intel_sub_group_2d_block_read_8b_32r32x1c.
  if (op.getElemSizeInBits() == 8 && op.getTileHeight() > 8 &&
      op.getTileWidth() == 32 && op.getVBlocks() == 1)
    return false;

  if (op.getCacheControl() != TritonGEN::LoadCacheControl::DEFAULT)
    return false;

  return true;
}

static Value createGenISA2DBlockRead(TritonGEN::Matrix2DBlockLoadOp op,
                                     ConversionPatternRewriter &rewriter) {
  MLIRContext *context = rewriter.getContext();
  VectorType resType = op.getRes().getType();
  Location loc = op->getLoc();

  // FIXME: Use the OpenCL API also for all other variants.
  if (isOCLBuiltinAvailable(op)) {
    auto dest = rewriter.create<LLVM::AllocaOp>(
        loc, ptr_ty(context), resType.getElementType(),
        i32_val(resType.getNumElements()));
    std::string fnName = "intel_sub_group_2d_block_read_";
    if (op.getVnniTransform())
      fnName += "transform_";
    else if (op.getTranspose())
      fnName += "transpose_";
    fnName += std::to_string(op.getElemSizeInBits()) + "b_" +
              std::to_string(op.getTileHeight()) + "r" +
              std::to_string(op.getTileWidth()) + "x" +
              std::to_string(op.getVBlocks()) + "c";
    fnName = "_Z" + std::to_string(fnName.size()) + fnName + "PU3AS1viiiDv2_iP";
    fnName +=
        (resType.getElementType().getIntOrFloatBitWidth() == 32) ? "j" : "t";
    VectorType vecType = vec_ty(i32_ty, 2);
    Value byteCoord = insert_element(
        vecType, insert_element(vecType, undef(vecType), op.getX(), i32_val(0)),
        op.getY(), i32_val(1));
    SmallVector<Type> argTypes{
        ptr_ty(context, 1), i32_ty, i32_ty, i32_ty, vecType, ptr_ty(context)};
    SmallVector<Value> args{op.getPtr(),        op.getBaseWidth(),
                            op.getBaseHeight(), op.getBasePitch(),
                            byteCoord,          dest};

    MLIRContext *ctx = rewriter.getContext();
    intel::AttrBuilder funcAttrBuilder(*ctx);
    intel::AttrBuilder param0AttrBuilder(*ctx);
    intel::AttrBuilder param5AttrBuilder(*ctx);
    funcAttrBuilder.addPassthroughAttribute(llvm::Attribute::NoUnwind);
    param0AttrBuilder.addAttribute(llvm::Attribute::NonNull);
    param0AttrBuilder.addAttribute(llvm::Attribute::ReadOnly);
    param5AttrBuilder.addAttribute(llvm::Attribute::NonNull);
    param5AttrBuilder.addAttribute(llvm::Attribute::WriteOnly);
    std::vector<NamedAttrList> paramAttrs(argTypes.size());
    paramAttrs[0] = param0AttrBuilder.getAttributes();
    paramAttrs[5] = param5AttrBuilder.getAttributes();
    intel::AttributeList attrs = getAttrList(funcAttrBuilder, paramAttrs);

    createDeviceFunctionCall(rewriter, fnName, void_ty(context), argTypes, args,
                             attrs);
    return rewriter.create<LLVM::LoadOp>(loc, resType, dest);
  }

  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
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
                             basePitch.getType(),
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
  auto cache = rewriter.create<LLVM::ConstantOp>(
      loc, int32Ty, static_cast<int>(op.getCacheControl()));

  Value one = i32_val(1);
  SmallVector<Value> args{ptr,
                          sub(baseWidth, one),
                          sub(baseHeight, one),
                          sub(basePitch, one),
                          x,
                          y,
                          elemSize,
                          tileWidth,
                          tileHeight,
                          vBlocks,
                          useTranspose,
                          vnniTransform,
                          cache};

  auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
  callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  return callOp.getResult();
}

// FIXME: This is a temporary solution. Remove once IGC can update the address
// payload.
static LLVM::CallOp
createBlock2DReadWithAddressPayloadUpdate(TritonGEN::Matrix2DBlockLoadOp op,
                                          ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  Type resType = op->getResultTypes()[0];
  Location loc = op->getLoc();

  auto createBlock2DAddressPayload = [&](TritonGEN::Matrix2DBlockLoadOp op) {
    SmallVector<Type> argTypes{i64_ty, i32_ty, i32_ty, i32_ty, i32_ty,
                               i32_ty, i32_ty, i32_ty, i32_ty};
    Value zero = i32_val(0);
    Value one = i32_val(1);
    SmallVector<Value> args{ptrtoint(i64_ty, op.getPtr()),
                            sub(op.getBaseWidth(), one),
                            sub(op.getBaseHeight(), one),
                            sub(op.getBasePitch(), one),
                            zero,
                            zero,
                            i32_val(op.getTileWidth()),
                            i32_val(op.getTileHeight()),
                            i32_val(op.getVBlocks())};

    // Function attributes.
    intel::AttrBuilder funcAttrBuilder(*ctx);
    funcAttrBuilder.addPassthroughAttribute(llvm::Attribute::NoUnwind)
        .addPassthroughAttribute(
            llvm::Attribute::Memory,
            llvm::MemoryEffects::argMemOnly(llvm::ModRefInfo::Ref)
                .toIntValue());
    intel::AttributeList attrs = getAttrList(funcAttrBuilder);

    LLVM::CallOp callOp = createDeviceFunctionCall(
        rewriter, "__builtin_IB_subgroup_createBlock2DAddressPayload",
        ptr_ty(ctx), argTypes, args, attrs);
    return callOp.getResult();
  };

  auto setBlock2DAddressPayload = [&](Value ptr,
                                      TritonGEN::Matrix2DBlockLoadOp op) {
    assert(isa<LLVM::LLVMPointerType>(ptr.getType()) &&
           "Expecting a pointer type");
    SmallVector<Type> argTypes{ptr.getType(), i32_ty};

    // Function and parameters attributes.
    intel::AttrBuilder funcAttrBuilder(*ctx);
    intel::AttrBuilder paramAttrBuilder(*ctx);
    funcAttrBuilder.addPassthroughAttribute(llvm::Attribute::NoUnwind)
        .addPassthroughAttribute(
            llvm::Attribute::Memory,
            llvm::MemoryEffects::argMemOnly(llvm::ModRefInfo::Mod)
                .toIntValue());
    paramAttrBuilder.addAttribute(llvm::Attribute::NonNull);
    std::vector<NamedAttrList> paramAttrs(argTypes.size());
    paramAttrs[0] = paramAttrBuilder.getAttributes();
    intel::AttributeList attrs = getAttrList(funcAttrBuilder, paramAttrs);

    createDeviceFunctionCall(
        rewriter, "__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX",
        LLVM::LLVMVoidType::get(ctx), argTypes, {ptr, op.getX()}, attrs);
    createDeviceFunctionCall(
        rewriter, "__builtin_IB_subgroup_setBlock2DAddressPayloadBlockY",
        LLVM::LLVMVoidType::get(ctx), argTypes, {ptr, op.getY()}, attrs);
  };

  auto createBlock2DRead = [&](Value ptr, TritonGEN::Matrix2DBlockLoadOp op) {
    assert(isa<LLVM::LLVMPointerType>(ptr.getType()) &&
           "Expecting a pointer type");

    std::string fnName = "__builtin_IB_subgroup_block_read_ap_";
    if (op.getVnniTransform())
      fnName += "transform_";
    fnName += "u" + std::to_string(op.getElemSizeInBits()) + "_m" +
              std::to_string(op.getTileHeight()) + "k" +
              std::to_string(op.getTileWidth()) + "v" +
              std::to_string(op.getVBlocks());
    Value zero = i32_val(0);
    SmallVector<Type> argTypes{ptr.getType(), i32_ty, i32_ty, i32_ty};
    SmallVector<Value> args{ptr, zero, zero, zero};

    // Function and parameters attributes.
    intel::AttrBuilder funcAttrBuilder(*ctx);
    intel::AttrBuilder paramAttrBuilder(*ctx);
    funcAttrBuilder.addPassthroughAttribute(llvm::Attribute::NoUnwind)
        .addPassthroughAttribute(
            llvm::Attribute::Memory,
            llvm::MemoryEffects::argMemOnly(llvm::ModRefInfo::Ref)
                .toIntValue());
    paramAttrBuilder.addAttribute(llvm::Attribute::NonNull);
    SmallVector<NamedAttrList> paramAttrs(argTypes.size());
    paramAttrs[0] = paramAttrBuilder.getAttributes();
    intel::AttributeList attrs = getAttrList(funcAttrBuilder, paramAttrs);

    return createDeviceFunctionCall(rewriter, fnName, resType, argTypes, args,
                                    attrs);
  };

  Value ptr = createBlock2DAddressPayload(op);
  setBlock2DAddressPayload(ptr, op);
  return createBlock2DRead(ptr, op);
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
                             basePitch.getType(),
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
  auto cache = rewriter.create<LLVM::ConstantOp>(
      loc, int32Ty, static_cast<int>(op.getCacheControl()));

  Value one = i32_val(1);
  SmallVector<Value> args{ptr,
                          sub(baseWidth, one),
                          sub(baseHeight, one),
                          sub(basePitch, one),
                          x,
                          y,
                          elemSize,
                          tileWidth,
                          tileHeight,
                          vBlocks,
                          useTranspose,
                          vnniTransform,
                          cache,
                          storeVal};

  auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
  callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  return callOp;
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
                             basePitch.getType(),
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

  Value one = i32_val(1);
  SmallVector<Value> args{ptr,
                          sub(baseWidth, one),
                          sub(baseHeight, one),
                          sub(basePitch, one),
                          x,
                          y,
                          elemSize,
                          tileWidth,
                          tileHeight,
                          vBlocks,
                          useTranspose,
                          vnniTransform,
                          cache};

  auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
  callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  return callOp;
}

namespace {

struct FuncCallLowering {
protected:
  Value rewrite(Operation *op, StringRef funcName, unsigned dim,
                ConversionPatternRewriter &rewriter) const {
    auto retType = rewriter.getIntegerType(64);
    auto argType = rewriter.getIntegerType(32);
    auto arg = LLVM::createConstantI32(op->getLoc(), rewriter, dim);

    intel::AttributeList attrs;
    LLVM::CallOp callOp = createDeviceFunctionCall(rewriter, funcName, retType,
                                                   {argType}, {arg}, attrs);

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

    intel::AttributeList attrs;
    LLVM::CallOp callOp = createDeviceFunctionCall(
        rewriter, "_Z25__spirv_BuiltInSubgroupIdv", retType, {}, {}, attrs);
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
    MLIRContext *ctx = rewriter.getContext();
    auto retType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto argType = rewriter.getIntegerType(32);
    auto arg = LLVM::createConstantI32(op->getLoc(), rewriter, MemFence::Local);

    intel::AttrBuilder funcAttrBuilder(*ctx);
    funcAttrBuilder.addPassthroughAttribute(llvm::Attribute::Convergent);
    intel::AttributeList attrs = getAttrList(funcAttrBuilder);

    LLVM::CallOp callOp = createDeviceFunctionCall(
        rewriter, "_Z7barrierj", {retType}, {argType}, {arg}, attrs);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct TritonGENSplitBarrier {
protected:
  template <typename OpType>
  void replaceWithCall(OpType op, StringRef funcName,
                       ConversionPatternRewriter &rewriter) const {
    static_assert(
        std::is_same<OpType, TritonGEN::SplitBarrierSignalOp>::value ||
            std::is_same<OpType, TritonGEN::SplitBarrierWaitOp>::value,
        "Unexpected OpType");

    auto retType = LLVM::LLVMVoidType::get(rewriter.getContext());
    Location loc = op->getLoc();
    auto memFence = LLVM::createConstantI32(loc, rewriter,
                                            static_cast<int>(op.getMemFence()));
    auto memScope = LLVM::createConstantI32(loc, rewriter,
                                            static_cast<int>(op.getMemScope()));
    SmallVector<Value> args{memFence, memScope};
    SmallVector<Type> argTypes;
    for (auto arg : args)
      argTypes.push_back(arg.getType());

    MLIRContext *ctx = rewriter.getContext();
    intel::AttrBuilder funcAttrBuilder(*ctx);
    funcAttrBuilder.addPassthroughAttribute(llvm::Attribute::Convergent);
    intel::AttributeList attrs = getAttrList(funcAttrBuilder);

    LLVM::CallOp callOp = createDeviceFunctionCall(rewriter, funcName, retType,
                                                   argTypes, args, attrs);
    rewriter.replaceOp(op, callOp);
  }
};

struct TritonGENSplitBarrierSignalLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SplitBarrierSignalOp>,
      public TritonGENSplitBarrier {
  using ConvertOpToLLVMPattern<
      TritonGEN::SplitBarrierSignalOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TritonGEN::SplitBarrierSignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TritonGENSplitBarrier::replaceWithCall(
        op, "_Z31intel_work_group_barrier_arriveii", rewriter);
    return success();
  }
};

struct TritonGENSplitBarrierWaitLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SplitBarrierWaitOp>,
      public TritonGENSplitBarrier {
  using ConvertOpToLLVMPattern<
      TritonGEN::SplitBarrierWaitOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TritonGEN::SplitBarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TritonGENSplitBarrier::replaceWithCall(
        op, "_Z29intel_work_group_barrier_waitii", rewriter);
    return success();
  }
};

struct TritonGENNamedBarrierSignalLowering
    : public ConvertOpToLLVMPattern<TritonGEN::NamedBarrierSignalOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::NamedBarrierSignalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::NamedBarrierSignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();
    Location loc = op->getLoc();

    Value barrierId = op.getBarrierId();
    Value threadGroupCount = op.getThreadGroupCount();

    llvm::LLVMContext llvmContext;
    LLVM::TypeToLLVMIRTranslator typeTranslator(llvmContext);
    llvm::Type *barrierTy = typeTranslator.translateType(barrierId.getType());
    llvm::Type *threadGroupCountTy =
        typeTranslator.translateType(threadGroupCount.getType());
    std::string funcName = llvm::GenISAIntrinsic::getName(
        llvm::GenISAIntrinsic::GenISA_threadgroupnamedbarriers_signal,
        {barrierTy, threadGroupCountTy});

    LLVM::LLVMFuncOp funcOp = LLVM::lookupOrCreateFn(
        moduleOp, funcName, {barrierId.getType(), threadGroupCount.getType()},
        LLVM::LLVMVoidType::get(context));
    auto convergentAttr =
        rewriter.getArrayAttr(StringAttr::get(context, "convergent"));
    funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    funcOp.setPassthroughAttr(convergentAttr);

    SmallVector<Value> args{barrierId, threadGroupCount};
    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
    callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    callOp->setAttr("passthrough", convergentAttr);
    rewriter.replaceOp(op, callOp);

    return success();
  }
};

struct TritonGENNamedBarrierWaitLowering
    : public ConvertOpToLLVMPattern<TritonGEN::NamedBarrierWaitOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::NamedBarrierWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::NamedBarrierWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    MLIRContext *context = rewriter.getContext();
    Location loc = op->getLoc();

    Value barrierId = op.getBarrierId();

    llvm::LLVMContext llvmContext;
    LLVM::TypeToLLVMIRTranslator typeTranslator(llvmContext);
    llvm::Type *barrierTy = typeTranslator.translateType(barrierId.getType());
    std::string funcName = llvm::GenISAIntrinsic::getName(
        llvm::GenISAIntrinsic::GenISA_threadgroupnamedbarriers_wait,
        {barrierTy});

    LLVM::LLVMFuncOp funcOp =
        LLVM::lookupOrCreateFn(moduleOp, funcName, {barrierId.getType()},
                               LLVM::LLVMVoidType::get(context));
    auto convergentAttr =
        rewriter.getArrayAttr(StringAttr::get(context, "convergent"));
    funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    funcOp.setPassthroughAttr(convergentAttr);

    SmallVector<Value> args{barrierId};
    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
    callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    callOp->setAttr("passthrough", convergentAttr);
    rewriter.replaceOp(op, callOp);

    return success();
  }
};

struct TritonSubGroupReduceLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SubGroupReduceOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::SubGroupReduceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SubGroupReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value val = op.getValue();
    Type orig_ty = val.getType();
    if (orig_ty.isInteger() && orig_ty.getIntOrFloatBitWidth() < 8)
      val = zext(i8_ty, val);

    Type val_ty = val.getType();
    SmallVector<Type> argTypes{val_ty};
    SmallVector<Value> args{val};
    bool useCluster = (getSubgroupSize(op) != op.getSize());
    std::string fnName = "sub_group_";
    if (useCluster)
      fnName += "clustered_";
    fnName += "reduce_" + stringifyReduceKind(op.getKind()).str();
    fnName = "_Z" + std::to_string(fnName.size()) + fnName +
             intel::getTypeMangling(val_ty);
    if (useCluster) {
      fnName += "j";
      argTypes.push_back(i32_ty);
      auto size = rewriter.create<LLVM::ConstantOp>(
          loc, i32_ty, static_cast<int>(op.getSize()));
      args.push_back(size);
    }

    MLIRContext *ctx = rewriter.getContext();
    intel::AttrBuilder funcAttrBuilder(*ctx);
    funcAttrBuilder.addPassthroughAttribute(llvm::Attribute::Convergent);
    intel::AttributeList attrs = getAttrList(funcAttrBuilder);

    Value result = createDeviceFunctionCall(rewriter, fnName, val_ty, argTypes,
                                            args, attrs)
                       .getResult();
    if (orig_ty.isInteger() && orig_ty.getIntOrFloatBitWidth() < 8)
      result = trunc(orig_ty, result);
    rewriter.replaceOp(op, result);
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
    Location loc = op.getLoc();
    Value val = op.getValue();
    Value mask = op.getMask();
    TritonGEN::ShflKind kind = op.getKind();
    Type orig_type = val.getType();
    unsigned bits = orig_type.getIntOrFloatBitWidth();
    if (bits < 8) {
      if (!orig_type.isInteger())
        val = bitcast(val, int_ty(bits));
      val = zext(i8_ty, val);
    } else if (isa<BFloat16Type>(orig_type)) {
      val = bitcast(val, i16_ty);
    }
    Value result = createSubGroupShuffle(rewriter, val, mask, kind).getResult();
    if (bits < 8) {
      result = trunc(int_ty(bits), result);
      if (!orig_type.isInteger())
        result = bitcast(result, orig_type);
    } else if (isa<BFloat16Type>(orig_type)) {
      result = bitcast(result, orig_type);
    }
    rewriter.replaceOp(op, result);
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
    if (tools::getBoolEnv("TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT")) {
      LLVM::CallOp callOp =
          createBlock2DReadWithAddressPayloadUpdate(op, rewriter);
      rewriter.replaceOp(op, callOp);
      return success();
    }

    rewriter.replaceOp(op, createGenISA2DBlockRead(op, rewriter));
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
  patterns.add<
      TritonGENThreadIdXLowering, TritonGENThreadIdYLowering,
      TritonGENThreadIdZLowering, TritonGENBlockIdXLowering,
      TritonGENBlockIdYLowering, TritonGENBlockIdZLowering,
      TritonGENBlockDimXLowering, TritonGENBlockDimYLowering,
      TritonGENBlockDimZLowering, TritonGENGridDimXLowering,
      TritonGENGridDimYLowering, TritonGENGridDimZLowering,
      TritonGENSubgroupIdLowering, TritonGENBarrierLowering,
      TritonGENSplitBarrierSignalLowering, TritonGENSplitBarrierWaitLowering,
      TritonGENNamedBarrierSignalLowering, TritonGENNamedBarrierWaitLowering,
      TritonSubGroupReduceLowering, TritonSubGroupShuffleLowering,
      TritonMatrixDPASLowering, TritonMatrix2DBlockLoadLowering,
      TritonMatrix2DBlockStoreLowering, TritonMatrix2DBlockPrefetchLowering>(
      converter);
}

void registerConvertTritonTritonGENToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, TritonGEN::TritonGENDialect *dialect) {
        dialect->addInterfaces<TritonGENToLLVMDialectInterface>();
      });
}
