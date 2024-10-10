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
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ModRef.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
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

static intel::AttributeList createFunctionAttributes(
    ArrayRef<std::pair<llvm::Attribute::AttrKind, std::optional<uint64_t>>>
        attributes,
    MLIRContext *ctx) {
  intel::AttrBuilder funcAttrBuilder(*ctx);
  for (auto [kind, optValue] : attributes) {
    if (optValue)
      funcAttrBuilder.addPassthroughAttribute(kind, *optValue);
    else
      funcAttrBuilder.addPassthroughAttribute(kind);
  }

  intel::AttributeList attrs;
  attrs.addFnAttributes(funcAttrBuilder);
  return attrs;
}

struct LLVMFuncAttributeOptions {
  bool isConvergent = false;
  bool isNoUnwind = false;
  bool isWillReturn = false;
  LLVM::MemoryEffectsAttr memEffectsAttr{};
};

static constexpr LLVMFuncAttributeOptions convergentAttrs = {
    true, false, false, {}};
static constexpr LLVMFuncAttributeOptions noUnwindAttrs = {
    false, true, false, {}};
static constexpr LLVMFuncAttributeOptions noUnwindWillReturnAttrs = {
    false, true, true, {}};
static constexpr LLVMFuncAttributeOptions convergentNoUnwindWillReturnAttrs = {
    true, true, true, {}};

static LLVM::CallOp createDeviceFunctionCall(
    ConversionPatternRewriter &rewriter, StringRef funcName, Type retType,
    ArrayRef<Type> argTypes, ArrayRef<Value> args,
    mlir::ArrayRef<std::pair<unsigned, mlir::StringRef>> paramAttrs,
    const LLVMFuncAttributeOptions &funcAttributeOptions,
    const intel::AttributeList &passthroughAttrs = {}) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *ctx = rewriter.getContext();
  Location loc = UnknownLoc::get(ctx);

  LLVM::LLVMFuncOp funcOp =
      LLVM::lookupOrCreateFn(moduleOp, funcName, argTypes, retType);
  funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
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

[[maybe_unused]] static std::string getGenISATypeMangling(Type ty) {
  if (auto vecTy = dyn_cast<VectorType>(ty))
    return "v" + std::to_string(vecTy.getNumElements()) +
           getGenISATypeMangling(vecTy.getElementType());
  return (ty.isInteger() ? "i" : "f") +
         std::to_string(ty.getIntOrFloatBitWidth());
}

static SmallVector<Attribute>
loadCacheControlToDecoration(Builder &builder, uint32_t operandNum,
                             TritonGEN::LoadCacheControl orig) {
  const auto build = [&builder,
                      operandNum](TritonGEN::LoadCacheControlDecorationEnum l1,
                                  TritonGEN::LoadCacheControlDecorationEnum l3)
      -> SmallVector<Attribute> {
    return {builder.getAttr<TritonGEN::LoadCacheControlDecorationAttr>(
                0, l1, operandNum),
            builder.getAttr<TritonGEN::LoadCacheControlDecorationAttr>(
                1, l3, operandNum)};
  };

  switch (orig) {
  case TritonGEN::LoadCacheControl::DEFAULT:
    return {};
  case TritonGEN::LoadCacheControl::L1UC_L3UC:
    return build(TritonGEN::LoadCacheControlDecorationEnum::Uncached,
                 TritonGEN::LoadCacheControlDecorationEnum::Uncached);
  case TritonGEN::LoadCacheControl::L1UC_L3C:
    return build(TritonGEN::LoadCacheControlDecorationEnum::Uncached,
                 TritonGEN::LoadCacheControlDecorationEnum::Cached);
  case TritonGEN::LoadCacheControl::L1C_L3UC:
    return build(TritonGEN::LoadCacheControlDecorationEnum::Cached,
                 TritonGEN::LoadCacheControlDecorationEnum::Uncached);
  case TritonGEN::LoadCacheControl::L1C_L3C:
    return build(TritonGEN::LoadCacheControlDecorationEnum::Cached,
                 TritonGEN::LoadCacheControlDecorationEnum::Cached);
  case TritonGEN::LoadCacheControl::L1S_L3UC:
    return build(TritonGEN::LoadCacheControlDecorationEnum::Streaming,
                 TritonGEN::LoadCacheControlDecorationEnum::Uncached);
  case TritonGEN::LoadCacheControl::L1S_L3C:
    return build(TritonGEN::LoadCacheControlDecorationEnum::Streaming,
                 TritonGEN::LoadCacheControlDecorationEnum::Cached);
  case TritonGEN::LoadCacheControl::L1IAR_L3C:
    return build(TritonGEN::LoadCacheControlDecorationEnum::InvalidateAfterRead,
                 TritonGEN::LoadCacheControlDecorationEnum::Cached);
  }
  llvm_unreachable("Unhandled case");
}

static std::optional<TritonGEN::DecorationCacheControlAttr>
loadCacheControlToCacheControls(Builder &builder,
                                TritonGEN::LoadCacheControl orig,
                                uint32_t operandNum) {
  SmallVector<Attribute> decorations =
      loadCacheControlToDecoration(builder, operandNum, orig);
  if (decorations.empty())
    return {};
  return builder.getAttr<TritonGEN::DecorationCacheControlAttr>(decorations);
}

static bool isOCLBuiltinAvailable(TritonGEN::Matrix2DBlockLoadOp op) {
  VectorType resTy = op.getRes().getType();
  unsigned resElemTySize = resTy.getElementType().getIntOrFloatBitWidth();
  bool needsResElemSizeEqualTo32 =
      op.getElemSizeInBits() == 32 || op.getVnniTransform();
  assert((!needsResElemSizeEqualTo32 || resElemTySize == 32) &&
         "Expecting 32-bit element type");
  if (!needsResElemSizeEqualTo32 && resElemTySize != 16)
    return false;

  if (op.getVnniTransform())
    return true;

  if (op.getTranspose() && op.getTileHeight() != 16)
    return false;

  uint32_t tileWidth = op.getTileWidth();
  switch (op.getElemSizeInBits()) {
  case 8:
    return (tileWidth == 32);
  case 16:
    return (tileWidth == 16);
  case 32:
    return (tileWidth == 8 || tileWidth == 16);
  default:
    llvm_unreachable("unexpected element size");
  }

  return false;
}

[[maybe_unused]] static Value
createGenISA2DBlockRead(TritonGEN::Matrix2DBlockLoadOp op,
                        ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  VectorType resType = op.getRes().getType();
  Location loc = op->getLoc();

  Value ptr = op.getPtr();
  Value baseWidth = op.getBaseWidth();
  Value baseHeight = op.getBaseHeight();
  Value basePitch = op.getBasePitch();
  Value x = op.getX();
  Value y = op.getY();

  std::string funcName =
      "llvm.genx.GenISA.LSC2DBlockRead." + getGenISATypeMangling(resType);
  IntegerType int1Ty = rewriter.getIntegerType(1);
  IntegerType int32Ty = rewriter.getIntegerType(32);
  IntegerType int64Ty = rewriter.getIntegerType(64);

  // The IGC intrinsic requires the first argument be int64
  ptr = rewriter.create<LLVM::PtrToIntOp>(loc, int64Ty, ptr);
  Value one = i32_val(1);

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

  SmallVector<Value> args{ptr,
                          sub(baseWidth, one),
                          sub(baseHeight, one),
                          sub(basePitch, one),
                          x,
                          y,
                          i32_val(op.getElemSizeInBits()),
                          i32_val(op.getTileWidth()),
                          i32_val(op.getTileHeight()),
                          i32_val(op.getVBlocks()),
                          i1_val(op.getTranspose()),
                          i1_val(op.getVnniTransform()),
                          i32_val(static_cast<int>(op.getCacheControl()))};

  LLVM::CallOp call = createDeviceFunctionCall(
      rewriter, funcName, resType, argTypes, args, {}, noUnwindWillReturnAttrs);
  return call.getResult();
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

    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::Ref,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = noUnwindAttrs;
    funcAttrs.memEffectsAttr = memAttr;

    LLVM::CallOp callOp = createDeviceFunctionCall(
        rewriter, "__builtin_IB_subgroup_createBlock2DAddressPayload",
        ptr_ty(ctx), argTypes, args, {}, funcAttrs);
    return callOp.getResult();
  };

  auto setBlock2DAddressPayload = [&](Value ptr,
                                      TritonGEN::Matrix2DBlockLoadOp op) {
    assert(isa<LLVM::LLVMPointerType>(ptr.getType()) &&
           "Expecting a pointer type");
    SmallVector<Type> argTypes{ptr.getType(), i32_ty};

    std::array<std::pair<unsigned, mlir::StringRef>, 1> paramAttrs{
        std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName())};

    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::Mod,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = noUnwindWillReturnAttrs;
    funcAttrs.memEffectsAttr = memAttr;

    createDeviceFunctionCall(
        rewriter, "__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX",
        LLVM::LLVMVoidType::get(ctx), argTypes, {ptr, op.getX()}, paramAttrs,
        funcAttrs);
    createDeviceFunctionCall(
        rewriter, "__builtin_IB_subgroup_setBlock2DAddressPayloadBlockY",
        LLVM::LLVMVoidType::get(ctx), argTypes, {ptr, op.getY()}, paramAttrs,
        funcAttrs);
  };

  auto createBlock2DRead = [&](Value ptr, TritonGEN::Matrix2DBlockLoadOp op) {
    assert(isa<LLVM::LLVMPointerType>(ptr.getType()) &&
           "Expecting a pointer type");

    std::string fnName = "__builtin_IB_subgroup_block_read_ap_";
    if (op.getTranspose())
      fnName += "transpose_";
    if (op.getVnniTransform())
      fnName += "transform_";
    fnName += "u" + std::to_string(op.getElemSizeInBits()) + "_m" +
              std::to_string(op.getTileHeight()) + "k" +
              std::to_string(op.getTileWidth()) + "v" +
              std::to_string(op.getVBlocks());
    Value zero = i32_val(0);
    SmallVector<Type> argTypes{ptr.getType(), i32_ty, i32_ty, i32_ty};
    SmallVector<Value> args{ptr, zero, zero, zero};

    std::array<std::pair<unsigned, mlir::StringRef>, 1> paramAttrs{
        std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName())};

    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::Ref,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = noUnwindWillReturnAttrs;
    funcAttrs.memEffectsAttr = memAttr;

    return createDeviceFunctionCall(rewriter, fnName, resType, argTypes, args,
                                    paramAttrs, funcAttrs);
  };

  Value ptr = createBlock2DAddressPayload(op);
  setBlock2DAddressPayload(ptr, op);

  return createBlock2DRead(ptr, op);
}

static SmallVector<Attribute>
storeCacheControlToDecoration(Builder &builder, uint32_t operandNum,
                              TritonGEN::StoreCacheControl orig) {
  const auto build = [&builder,
                      operandNum](TritonGEN::StoreCacheControlDecorationEnum l1,
                                  TritonGEN::StoreCacheControlDecorationEnum l3)
      -> SmallVector<Attribute> {
    return {builder.getAttr<TritonGEN::StoreCacheControlDecorationAttr>(
                0, l1, operandNum),
            builder.getAttr<TritonGEN::StoreCacheControlDecorationAttr>(
                1, l3, operandNum)};
  };

  switch (orig) {
  case TritonGEN::StoreCacheControl::DEFAULT:
    return {};
  case TritonGEN::StoreCacheControl::L1UC_L3UC:
    return build(TritonGEN::StoreCacheControlDecorationEnum::Uncached,
                 TritonGEN::StoreCacheControlDecorationEnum::Uncached);
  case TritonGEN::StoreCacheControl::L1UC_L3WB:
    return build(TritonGEN::StoreCacheControlDecorationEnum::Uncached,
                 TritonGEN::StoreCacheControlDecorationEnum::WriteBack);
  case TritonGEN::StoreCacheControl::L1WT_L3UC:
    return build(TritonGEN::StoreCacheControlDecorationEnum::WriteThrough,
                 TritonGEN::StoreCacheControlDecorationEnum::Uncached);
  case TritonGEN::StoreCacheControl::L1WT_L3WB:
    return build(TritonGEN::StoreCacheControlDecorationEnum::WriteThrough,
                 TritonGEN::StoreCacheControlDecorationEnum::WriteBack);
  case TritonGEN::StoreCacheControl::L1S_L3UC:
    return build(TritonGEN::StoreCacheControlDecorationEnum::Streaming,
                 TritonGEN::StoreCacheControlDecorationEnum::Uncached);
  case TritonGEN::StoreCacheControl::L1S_L3WB:
    return build(TritonGEN::StoreCacheControlDecorationEnum::Streaming,
                 TritonGEN::StoreCacheControlDecorationEnum::WriteBack);
  case TritonGEN::StoreCacheControl::L1WB_L3WB:
    return build(TritonGEN::StoreCacheControlDecorationEnum::WriteBack,
                 TritonGEN::StoreCacheControlDecorationEnum::WriteBack);
  }
  llvm_unreachable("Unhandled case");
}

static std::optional<TritonGEN::DecorationCacheControlAttr>
storeCacheControlToCacheControls(Builder &builder,
                                 TritonGEN::StoreCacheControl orig,
                                 uint32_t operandNum) {
  SmallVector<Attribute> decorations =
      storeCacheControlToDecoration(builder, operandNum, orig);
  if (decorations.empty())
    return {};
  return builder.getAttr<TritonGEN::DecorationCacheControlAttr>(decorations);
}

[[maybe_unused]] static LLVM::CallOp
createGenISA2DBlockWrite(TritonGEN::Matrix2DBlockStoreOp op,
                         ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();

  // The IGC intrinsic requires the first argument be int64
  Value ptr = op.getPtr();
  ptr = rewriter.create<LLVM::PtrToIntOp>(loc, int_ty(64), ptr);
  Value baseWidth = op.getBaseWidth();
  Value baseHeight = op.getBaseHeight();
  Value basePitch = op.getBasePitch();
  Value x = op.getX();
  Value y = op.getY();
  Value storeVal = op.getStoredVal();

  VectorType storeValType = op.getStoredVal().getType();
  std::string funcName =
      "llvm.genx.GenISA.LSC2DBlockWrite." + getGenISATypeMangling(storeValType);
  Value one = i32_val(1);

  SmallVector<Type> argTypes{
      int_ty(64),          baseWidth.getType(), baseHeight.getType(),
      basePitch.getType(), x.getType(),         y.getType(),
      int_ty(32),          int_ty(32),          int_ty(32),
      int_ty(32),          int_ty(1),           int_ty(1),
      int_ty(32),          storeVal.getType()};
  SmallVector<Value> args{ptr,
                          sub(baseWidth, one),
                          sub(baseHeight, one),
                          sub(basePitch, one),
                          x,
                          y,
                          i32_val(op.getElemSizeInBits()),
                          i32_val(op.getTileWidth()),
                          i32_val(op.getTileHeight()),
                          i32_val(op.getVBlocks()),
                          i1_val(false), // transpose
                          i1_val(false), // vnniTransform
                          i32_val(static_cast<int>(op.getCacheControl())),
                          storeVal};

  LLVM::CallOp call =
      createDeviceFunctionCall(rewriter, funcName, void_ty(ctx), argTypes, args,
                               {}, noUnwindWillReturnAttrs);
  return call;
}

[[maybe_unused]] static LLVM::CallOp
createGenISA2DBlockPrefetch(TritonGEN::Matrix2DBlockPrefetchOp op,
                            ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();

  // The IGC intrinsic requires the first argument be int64
  Value ptr = op.getPtr();
  ptr = rewriter.create<LLVM::PtrToIntOp>(loc, int_ty(64), ptr);
  Value baseWidth = op.getBaseWidth();
  Value baseHeight = op.getBaseHeight();
  Value basePitch = op.getBasePitch();
  Value x = op.getX();
  Value y = op.getY();
  Value one = i32_val(1);

  SmallVector<Type> argTypes{
      int_ty(64),          baseWidth.getType(), baseHeight.getType(),
      basePitch.getType(), x.getType(),         y.getType(),
      int_ty(32),          int_ty(32),          int_ty(32),
      int_ty(32),          int_ty(1),           int_ty(1),
      int_ty(32)};
  SmallVector<Value> args{ptr,
                          sub(baseWidth, one),
                          sub(baseHeight, one),
                          sub(basePitch, one),
                          x,
                          y,
                          i32_val(op.getElemSizeInBits()),
                          i32_val(op.getTileWidth()),
                          i32_val(op.getTileHeight()),
                          i32_val(op.getVBlocks()),
                          i1_val(false), // transpose
                          i1_val(false), // vnniTransform
                          i32_val(static_cast<int>(op.getCacheControl()))};

  const StringLiteral funcName = "llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid";
  return createDeviceFunctionCall(rewriter, funcName, void_ty(ctx), {argTypes},
                                  {args}, {}, noUnwindWillReturnAttrs);
}

namespace {

//===----------------------------------------------------------------------===//
// SubgroupID Op Lowering
//===----------------------------------------------------------------------===//

struct TritonGENSubgroupIdLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SubgroupIdOp> {
  using ConvertOpToLLVMPattern<TritonGEN::SubgroupIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SubgroupIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    Type retType = rewriter.getIntegerType(32);

    auto funcAttrs = noUnwindWillReturnAttrs;
    auto memory_zero = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::NoModRef,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    funcAttrs.memEffectsAttr = memory_zero;
    intel::AttributeList passthroughAttrs = createFunctionAttributes(
        {{llvm::Attribute::NoSync, std::nullopt}}, ctx);
    LLVM::CallOp callOp =
        createDeviceFunctionCall(rewriter, "_Z16get_sub_group_idv", retType, {},
                                 {}, {}, funcAttrs, passthroughAttrs);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SubgroupLocalID Op Lowering
//===----------------------------------------------------------------------===//

struct TritonGENSubgroupLocalIdLowering
    : ConvertOpToLLVMPattern<TritonGEN::SubgroupLocalIdOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::SubgroupLocalIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SubgroupLocalIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    Type retType = rewriter.getIntegerType(32);

    LLVM::CallOp callOp =
        createDeviceFunctionCall(rewriter, "_Z22get_sub_group_local_idv",
                                 retType, {}, {}, {}, noUnwindWillReturnAttrs);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Synchronization Ops Lowerings
//===----------------------------------------------------------------------===//

struct TritonGENSplitBarrier {
protected:
  template <typename OpType>
  void replaceWithCall(OpType op, StringRef funcName,
                       ConversionPatternRewriter &rewriter) const {
    static_assert(
        std::is_same<OpType, TritonGEN::SplitBarrierSignalOp>::value ||
            std::is_same<OpType, TritonGEN::SplitBarrierWaitOp>::value,
        "Unexpected OpType");

    MLIRContext *ctx = rewriter.getContext();
    Location loc = op->getLoc();
    Type retType = void_ty(ctx);
    Value memFence = i32_val(static_cast<int>(op.getMemFence()));
    Value memScope = i32_val(static_cast<int>(op.getMemScope()));
    SmallVector<Value> args{memFence, memScope};
    SmallVector<Type> argTypes;
    for (auto arg : args)
      argTypes.push_back(arg.getType());

    LLVM::CallOp callOp =
        createDeviceFunctionCall(rewriter, funcName, retType, argTypes, args,
                                 {}, convergentNoUnwindWillReturnAttrs);
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

struct TritonSubGroupBase {
protected:
  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, TritonGEN::SubGroupReduceOp,
                                 TritonGEN::SubGroupScanOp>::value>>
  static Value extend(OpType op, Value val, Type type,
                      ConversionPatternRewriter &rewriter) {
    Location loc = op.getLoc();
    unsigned bitWidth = type.getIntOrFloatBitWidth();

    if constexpr (llvm::is_one_of<OpType, TritonGEN::SubGroupReduceOp,
                                  TritonGEN::SubGroupScanOp>::value) {
      if (type.isInteger() && bitWidth < 8)
        val = zext(i8_ty, val);
    }

    return val;
  }

  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, TritonGEN::SubGroupReduceOp,
                                 TritonGEN::SubGroupScanOp>::value>>
  static Value truncate(OpType op, Value val, Type type,
                        ConversionPatternRewriter &rewriter) {
    Location loc = op.getLoc();
    unsigned bitWidth = type.getIntOrFloatBitWidth();

    if constexpr (llvm::is_one_of<OpType, TritonGEN::SubGroupReduceOp,
                                  TritonGEN::SubGroupScanOp>::value) {
      if (type.isInteger() && bitWidth < 8)
        val = trunc(type, val);
      return val;
    }

    return val;
  }
};

struct TritonSubGroupReduceLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SubGroupReduceOp>,
      public TritonSubGroupBase {
  using ConvertOpToLLVMPattern<
      TritonGEN::SubGroupReduceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SubGroupReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value val = op.getValue();
    Type origTy = val.getType();
    val = TritonSubGroupBase::extend(op, val, origTy, rewriter);
    Type valTy = val.getType();
    SmallVector<Type> argTypes{valTy};
    SmallVector<bool> argIsUnsigned{false};
    SmallVector<Value> args{val};
    bool useCluster = (getSubgroupSize(op) != op.getSize());

    std::string fnName = "sub_group_";
    fnName += useCluster ? "clustered_" : "non_uniform_";
    fnName += "reduce_" + stringifyReduceKind(op.getKind()).str();
    LLVMFuncAttributeOptions funcAttrs{};
    if (useCluster) {
      argTypes.push_back(i32_ty);
      argIsUnsigned.push_back(true);
      auto size = rewriter.create<LLVM::ConstantOp>(
          loc, i32_ty, static_cast<int>(op.getSize()));
      args.push_back(size);
      MLIRContext *ctx = rewriter.getContext();
      funcAttrs = convergentNoUnwindWillReturnAttrs;
    }
    fnName = intel::mangle(fnName, argTypes, argIsUnsigned);

    Value result = createDeviceFunctionCall(rewriter, fnName, valTy, argTypes,
                                            args, {}, funcAttrs)
                       .getResult();
    result = TritonSubGroupBase::truncate(op, result, origTy, rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct TritonSubGroupScanLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SubGroupScanOp>,
      public TritonSubGroupBase {
  using ConvertOpToLLVMPattern<
      TritonGEN::SubGroupScanOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SubGroupScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value val = op.getValue();
    Type origTy = val.getType();
    val = TritonSubGroupBase::extend(op, op.getValue(), origTy, rewriter);
    Type valTy = val.getType();
    SmallVector<Type> argTypes{valTy};
    SmallVector<Value> args{val};

    std::string fnName = "sub_group_non_uniform_scan_";
    switch (op.getScanKind()) {
    case TritonGEN::ScanKind::EXCLUSIVE:
      fnName += "exclusive_";
      break;
    case TritonGEN::ScanKind::INCLUSIVE:
      fnName += "inclusive_";
      break;
    default:
      llvm_unreachable("unhandled scan kind");
    };

    fnName += stringifyReduceKind(op.getReduceKind()).str();
    fnName = intel::mangle(fnName, valTy);

    Value result =
        createDeviceFunctionCall(rewriter, fnName, valTy, argTypes, args, {},
                                 convergentNoUnwindWillReturnAttrs)
            .getResult();
    result = TritonSubGroupBase::truncate(op, result, origTy, rewriter);
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
    Location loc = op->getLoc();

    FloatType fp32Ty = f32_ty;
    IntegerType int16Ty = int_ty(16);
    IntegerType int32Ty = int_ty(32);

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

    Value c = op.getC();
    VectorType cOrigTy = cast<VectorType>(c.getType());
    assert(cOrigTy == op->getResultTypes()[0] &&
           "Accumulator and result type mismatch");
    // OCL builtins encode bfloat16 as int16
    VectorType cTy = cOrigTy.getElementType().isBF16()
                         ? VectorType::get(cOrigTy.getShape(), int16Ty)
                         : cOrigTy;
    if (cOrigTy != cTy)
      c = rewriter.create<LLVM::BitcastOp>(loc, cTy, c);

    std::string fnName =
        "intel_sub_group_" + stringifyPrecisionType(precisionA).str() + "_" +
        stringifyPrecisionType(op.getPb()).str() + "_matrix_mad_k" +
        std::to_string(8 /*systolic depth*/ *
                       getNumOperandsPerDword(precisionA));

    SmallVector<Type> argTypes{aTy, bTy, cTy};
    fnName = intel::mangle(fnName, argTypes);

    SmallVector<Value> args{a, b, c};
    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::NoModRef,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = convergentNoUnwindWillReturnAttrs;
    funcAttrs.memEffectsAttr = memAttr;

    Value result = createDeviceFunctionCall(rewriter, fnName, cTy, argTypes,
                                            args, {}, funcAttrs)
                       ->getResult(0);
    if (cOrigTy != cTy)
      result = rewriter.create<LLVM::BitcastOp>(loc, cOrigTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
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

    if (!isOCLBuiltinAvailable(op)) {
      op.emitWarning() << "OpenCL API not available for this operation. Got "
                       << *op;
      rewriter.replaceOp(op, createGenISA2DBlockRead(op, rewriter));
      return success();
    }

    MLIRContext *ctx = rewriter.getContext();
    Location loc = op->getLoc();
    VectorType resType = op.getRes().getType();

    auto dest = rewriter.create<LLVM::AllocaOp>(
        loc, ptr_ty(ctx), resType.getElementType(),
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
        intel::getTypeMangling(resType.getElementType(), /*isUnsigned=*/true);
    VectorType vecType = vec_ty(i32_ty, 2);
    Value byteCoord = insert_element(
        vecType, insert_element(vecType, undef(vecType), op.getX(), i32_val(0)),
        op.getY(), i32_val(1));
    SmallVector<Type> argTypes{ptr_ty(ctx, 1), i32_ty,  i32_ty,
                               i32_ty,         vecType, ptr_ty(ctx)};
    SmallVector<Value> args{op.getPtr(),        op.getBaseWidth(),
                            op.getBaseHeight(), op.getBasePitch(),
                            byteCoord,          dest};

    std::array<std::pair<unsigned, mlir::StringRef>, 4> paramAttrs{
        std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
        std::make_pair(0, LLVM::LLVMDialect::getReadonlyAttrName()),
        std::make_pair(5, LLVM::LLVMDialect::getNonNullAttrName()),
        std::make_pair(5, LLVM::LLVMDialect::getWriteOnlyAttrName()),
    };

    LLVM::CallOp call =
        createDeviceFunctionCall(rewriter, fnName, void_ty(ctx), argTypes, args,
                                 paramAttrs, noUnwindWillReturnAttrs);
    constexpr uint32_t ptrOperandIndex = 0;
    if (std::optional<TritonGEN::DecorationCacheControlAttr> optCacheControls =
            loadCacheControlToCacheControls(rewriter, op.getCacheControl(),
                                            ptrOperandIndex)) {
      call->setAttr(TritonGEN::TritonGENDialect::getCacheControlsAttrName(),
                    *optCacheControls);
    }

    rewriter.replaceOp(op, rewriter.create<LLVM::LoadOp>(loc, resType, dest));
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
    MLIRContext *ctx = rewriter.getContext();
    Location loc = op->getLoc();

    VectorType storeValType = op.getStoredVal().getType();
    auto storeValPtr = rewriter.create<LLVM::AllocaOp>(
        loc, ptr_ty(ctx), storeValType.getElementType(),
        i32_val(storeValType.getNumElements()));
    rewriter.create<LLVM::StoreOp>(loc, op.getStoredVal(), storeValPtr);

    std::string fnName = "intel_sub_group_2d_block_write_";
    fnName += std::to_string(op.getElemSizeInBits()) + "b_" +
              std::to_string(op.getTileHeight()) + "r" +
              std::to_string(op.getTileWidth()) + "x" +
              std::to_string(op.getVBlocks()) + "c";
    fnName = "_Z" + std::to_string(fnName.size()) + fnName + "PU3AS1viiiDv2_iP";
    unsigned storeValBitWidth =
        storeValType.getElementType().getIntOrFloatBitWidth();
    fnName += (storeValBitWidth == 32)   ? "j"
              : (storeValBitWidth == 16) ? "t"
                                         : "h";

    VectorType vecType = vec_ty(i32_ty, 2);
    Value byteCoord = insert_element(
        vecType, insert_element(vecType, undef(vecType), op.getX(), i32_val(0)),
        op.getY(), i32_val(1));
    SmallVector<Type> argTypes{ptr_ty(ctx, 1), i32_ty,  i32_ty,
                               i32_ty,         vecType, ptr_ty(ctx)};
    SmallVector<Value> args{op.getPtr(),        op.getBaseWidth(),
                            op.getBaseHeight(), op.getBasePitch(),
                            byteCoord,          storeValPtr};

    std::array<std::pair<unsigned, mlir::StringRef>, 4> paramAttrs{
        std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
        std::make_pair(0, LLVM::LLVMDialect::getWriteOnlyAttrName()),
        std::make_pair(5, LLVM::LLVMDialect::getNonNullAttrName()),
        std::make_pair(5, LLVM::LLVMDialect::getReadonlyAttrName()),
    };

    LLVM::CallOp call =
        createDeviceFunctionCall(rewriter, fnName, void_ty(ctx), argTypes, args,
                                 paramAttrs, noUnwindWillReturnAttrs);
    constexpr uint32_t ptrOperandIndex = 0;
    if (std::optional<TritonGEN::DecorationCacheControlAttr> optCacheControls =
            storeCacheControlToCacheControls(rewriter, op.getCacheControl(),
                                             ptrOperandIndex)) {
      call->setAttr(TritonGEN::TritonGENDialect::getCacheControlsAttrName(),
                    *optCacheControls);
    }

    rewriter.replaceOp(op, call);
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
    MLIRContext *ctx = rewriter.getContext();
    Location loc = op->getLoc();
    std::string fnName = "intel_sub_group_2d_block_prefetch_";
    fnName += std::to_string(op.getElemSizeInBits()) + "b_" +
              std::to_string(op.getTileHeight()) + "r" +
              std::to_string(op.getTileWidth()) + "x" +
              std::to_string(op.getVBlocks()) + "c";
    fnName = "_Z" + std::to_string(fnName.size()) + fnName + "PU3AS1viiiDv2_i";
    VectorType vecType = vec_ty(i32_ty, 2);
    Value byteCoord = insert_element(
        vecType, insert_element(vecType, undef(vecType), op.getX(), i32_val(0)),
        op.getY(), i32_val(1));
    SmallVector<Type> argTypes{ptr_ty(ctx, 1), i32_ty, i32_ty, i32_ty, vecType};
    SmallVector<Value> args{op.getPtr(), op.getBaseWidth(), op.getBaseHeight(),
                            op.getBasePitch(), byteCoord};

    std::array<std::pair<unsigned, mlir::StringRef>, 1> paramAttrs{
        std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
    };

    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::Ref,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = noUnwindAttrs;
    funcAttrs.memEffectsAttr = memAttr;

    LLVM::CallOp call = createDeviceFunctionCall(
        rewriter, fnName, void_ty(ctx), argTypes, args, paramAttrs, funcAttrs);
    constexpr uint32_t ptrOperandIndex = 0;
    if (std::optional<TritonGEN::DecorationCacheControlAttr> optCacheControls =
            loadCacheControlToCacheControls(rewriter, op.getCacheControl(),
                                            ptrOperandIndex)) {
      call->setAttr(TritonGEN::TritonGENDialect::getCacheControlsAttrName(),
                    *optCacheControls);
    }

    rewriter.replaceOp(op, call);
    return success();
  }
};

template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                               OpType, TritonGEN::SIMDBlockReadOp,
                               TritonGEN::SIMDBlockWriteOp>::value>>
static std::string getSIMDBlockManglingName(OpType op, VectorType vecTy) {
  constexpr bool isWrite =
      std::is_same<OpType, TritonGEN::SIMDBlockWriteOp>::value;
  const LLVM::LLVMPointerType ptrTy = op.getPtr().getType();
  const unsigned numElems = vecTy.getNumElements();
  // Note: OCL builtin name here differs from regular mangling.
  std::string funcName = "intel_sub_group_block_";
  if constexpr (isWrite)
    funcName += "write";
  else
    funcName += "read";
  funcName += "_u" + intel::getTypeMangling(vecTy.getElementType()) +
              (numElems == 1 ? "" : std::to_string(numElems));
  funcName =
      "_Z" + std::to_string(funcName.size()) + funcName + "PU3AS" +
      std::to_string(ptrTy.getAddressSpace()) +
      intel::getTypeMangling(vecTy.getElementType(), /*isUnsigned=*/true);
  if constexpr (isWrite)
    funcName += intel::getTypeMangling(vecTy, /*isUnsigned=*/true);
  return funcName;
}

struct TritonSIMDBlockReadLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SIMDBlockReadOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::SIMDBlockReadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SIMDBlockReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::LLVMPointerType ptrTy = op.getPtr().getType();
    VectorType vecTy = op.getRes().getType();

    std::string funcName = getSIMDBlockManglingName(op, vecTy);
    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::Ref,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = noUnwindWillReturnAttrs;
    funcAttrs.memEffectsAttr = memAttr;
    LLVM::CallOp call = createDeviceFunctionCall(
        rewriter, funcName, vecTy, {ptrTy}, {op.getPtr()}, {}, funcAttrs, {});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

struct TritonSIMDBlockWriteLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SIMDBlockWriteOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::SIMDBlockWriteOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SIMDBlockWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    LLVM::LLVMPointerType ptrTy = op.getPtr().getType();
    VectorType vecTy = op.getVal().getType();

    std::string funcName = getSIMDBlockManglingName(op, vecTy);

    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::ModRef,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = noUnwindWillReturnAttrs;
    funcAttrs.memEffectsAttr = memAttr;
    LLVM::CallOp call = createDeviceFunctionCall(
        rewriter, funcName, void_ty(ctx), {ptrTy, vecTy},
        {op.getPtr(), op.getVal()}, {}, funcAttrs);

    rewriter.replaceOp(op, call);
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
    MLIRContext *ctx = &getContext();
    RewritePatternSet pattern(ctx);
    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter converter(ctx, options);
    LLVMConversionTarget target(*ctx);

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
  void loadDependentDialects(MLIRContext *ctx) const final {
    ctx->loadDialect<LLVM::LLVMDialect>();
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
      TritonGENSubgroupIdLowering, TritonGENSubgroupLocalIdLowering,
      TritonGENSplitBarrierSignalLowering, TritonGENSplitBarrierWaitLowering,
      TritonSubGroupReduceLowering, TritonSubGroupScanLowering,
      TritonMatrixDPASLowering, TritonMatrix2DBlockLoadLowering,
      TritonMatrix2DBlockStoreLowering, TritonMatrix2DBlockPrefetchLowering,
      TritonSIMDBlockReadLowering, TritonSIMDBlockWriteLowering>(converter);
}

void registerConvertTritonTritonGENToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, TritonGEN::TritonGENDialect *dialect) {
        dialect->addInterfaces<TritonGENToLLVMDialectInterface>();
      });
}
