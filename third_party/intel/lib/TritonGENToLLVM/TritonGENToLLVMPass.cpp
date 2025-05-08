//===- TritonGENToLLVMPass.cpp - TritonGEN to LLVM dialect conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Attributes.h"
#include "Utils/LLVMIntr.h"
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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/identity.h"
#include "llvm/Support/ErrorHandling.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

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

// HW requires base address to be 64-byte aligned. Compensate the non-64-byte
// alignment base address by adjusting the base width and x-coordinate offset.
template <
    typename OpTy,
    std::enable_if_t<llvm::is_one_of<OpTy, TritonGEN::Matrix2DBlockLoadOp,
                                     TritonGEN::Matrix2DBlockStoreOp,
                                     TritonGEN::Matrix2DBlockPrefetchOp>::value,
                     bool> = true>
static std::pair<Value, Value>
computeAlignedBaseWidthAndOffset(OpTy op, ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value baseAddr =
      rewriter.create<LLVM::PtrToIntOp>(loc, int_ty(64), op.getPtr());
  // A mask for 64-byte alignment (0x3f = 63).
  constexpr int64_t ALIGNMENT_MASK = 0x3f;
  // Calculate the byte offset of the base address from a 64-byte alignment.
  Value offsetInBytes =
      b.trunc(i32_ty, b.and_(baseAddr, b.i64_val(ALIGNMENT_MASK)));
  // Adjust the base width to account for the byte offset.
  Value adjustedBaseWidth = b.add(op.getBaseWidth(), offsetInBytes);
  // Adjust the x-coordinate offset based on the number of scalar elements.
  Value elemSizeInBytes = b.i32_val(op.getElemSizeInBits() / 8);
  Value adjustedXOffset =
      b.add(op.getX(), b.udiv(offsetInBytes, elemSizeInBytes));
  return {adjustedBaseWidth, adjustedXOffset};
}

[[maybe_unused]] static Value
createGenISA2DBlockRead(TritonGEN::Matrix2DBlockLoadOp op,
                        ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  VectorType resType = op.getRes().getType();
  Location loc = op->getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  Value ptr = op.getPtr();
  Value baseHeight = op.getBaseHeight();
  Value basePitch = op.getBasePitch();
  Value y = op.getY();

  std::string funcName =
      "llvm.genx.GenISA.LSC2DBlockRead." + getGenISATypeMangling(resType);
  IntegerType int1Ty = rewriter.getIntegerType(1);
  IntegerType int32Ty = rewriter.getIntegerType(32);
  IntegerType int64Ty = rewriter.getIntegerType(64);

  // The IGC intrinsic requires the first argument be int64
  ptr = rewriter.create<LLVM::PtrToIntOp>(loc, int64Ty, ptr);

  Value one = b.i32_val(1);
  auto [baseWidth, x] = computeAlignedBaseWidthAndOffset(op, rewriter);

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
                          b.sub(baseWidth, one),
                          b.sub(baseHeight, one),
                          b.sub(basePitch, one),
                          x,
                          y,
                          b.i32_val(op.getElemSizeInBits()),
                          b.i32_val(op.getTileWidth()),
                          b.i32_val(op.getTileHeight()),
                          b.i32_val(op.getVBlocks()),
                          b.i1_val(op.getTranspose()),
                          b.i1_val(op.getVnniTransform()),
                          b.i32_val(static_cast<int>(op.getCacheControl()))};

  LLVM::CallOp call =
      intel::createDeviceFunctionCall(rewriter, funcName, resType, argTypes,
                                      args, {}, intel::noUnwindWillReturnAttrs);
  return call.getResult();
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
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // The IGC intrinsic requires the first argument be int64
  Value ptr = op.getPtr();
  ptr = rewriter.create<LLVM::PtrToIntOp>(loc, int_ty(64), ptr);
  Value baseHeight = op.getBaseHeight();
  Value basePitch = op.getBasePitch();
  Value y = op.getY();
  Value storeVal = op.getStoredVal();

  VectorType storeValType = op.getStoredVal().getType();
  std::string funcName =
      "llvm.genx.GenISA.LSC2DBlockWrite." + getGenISATypeMangling(storeValType);
  Value one = b.i32_val(1);
  auto [baseWidth, x] = computeAlignedBaseWidthAndOffset(op, rewriter);

  SmallVector<Type> argTypes{
      int_ty(64),          baseWidth.getType(), baseHeight.getType(),
      basePitch.getType(), x.getType(),         y.getType(),
      int_ty(32),          int_ty(32),          int_ty(32),
      int_ty(32),          int_ty(1),           int_ty(1),
      int_ty(32),          storeVal.getType()};
  SmallVector<Value> args{ptr,
                          b.sub(baseWidth, one),
                          b.sub(baseHeight, one),
                          b.sub(basePitch, one),
                          x,
                          y,
                          b.i32_val(op.getElemSizeInBits()),
                          b.i32_val(op.getTileWidth()),
                          b.i32_val(op.getTileHeight()),
                          b.i32_val(op.getVBlocks()),
                          b.i1_val(false), // transpose
                          b.i1_val(false), // vnniTransform
                          b.i32_val(static_cast<int>(op.getCacheControl())),
                          storeVal};

  LLVM::CallOp call = intel::createDeviceFunctionCall(
      rewriter, funcName, void_ty(ctx), argTypes, args, {},
      intel::noUnwindWillReturnAttrs);
  return call;
}

[[maybe_unused]] static LLVM::CallOp
createGenISA2DBlockPrefetch(TritonGEN::Matrix2DBlockPrefetchOp op,
                            ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op->getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // The IGC intrinsic requires the first argument be int64
  Value ptr = op.getPtr();
  ptr = rewriter.create<LLVM::PtrToIntOp>(loc, int_ty(64), ptr);
  Value baseHeight = op.getBaseHeight();
  Value basePitch = op.getBasePitch();
  Value y = op.getY();
  Value one = b.i32_val(1);
  auto [baseWidth, x] = computeAlignedBaseWidthAndOffset(op, rewriter);

  SmallVector<Type> argTypes{
      int_ty(64),          baseWidth.getType(), baseHeight.getType(),
      basePitch.getType(), x.getType(),         y.getType(),
      int_ty(32),          int_ty(32),          int_ty(32),
      int_ty(32),          int_ty(1),           int_ty(1),
      int_ty(32)};
  SmallVector<Value> args{ptr,
                          b.sub(baseWidth, one),
                          b.sub(baseHeight, one),
                          b.sub(basePitch, one),
                          x,
                          y,
                          b.i32_val(op.getElemSizeInBits()),
                          b.i32_val(op.getTileWidth()),
                          b.i32_val(op.getTileHeight()),
                          b.i32_val(op.getVBlocks()),
                          b.i1_val(false), // transpose
                          b.i1_val(false), // vnniTransform
                          b.i32_val(static_cast<int>(op.getCacheControl()))};

  const StringLiteral funcName = "llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid";
  return intel::createDeviceFunctionCall(rewriter, funcName, void_ty(ctx),
                                         {argTypes}, {args}, {},
                                         intel::noUnwindWillReturnAttrs);
}

namespace {

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
    auto funcAttrs = intel::convergentNoUnwindWillReturnAttrs;
    funcAttrs.memEffectsAttr = memAttr;

    Value result = intel::createDeviceFunctionCall(
                       rewriter, fnName, cTy, argTypes, args, {}, funcAttrs)
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
    if (op.getElemSizeInBits() == 8 && op.getTileWidth() == 16 &&
        op.getVBlocks() != 4 && !op.getVnniTransform()) {
      // TODO: add ocl builtin/spirv intrinsics for 8b 16 column 1 vBlock & 2
      // vBlock reads
      rewriter.replaceOp(op, createGenISA2DBlockRead(op, rewriter));
      return success();
    }

    MLIRContext *ctx = rewriter.getContext();
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    VectorType resType = op.getRes().getType();

    auto dest = rewriter.create<LLVM::AllocaOp>(
        loc, ptr_ty(ctx), resType.getElementType(),
        b.i32_val(resType.getNumElements()));
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

    auto [baseWidth, offsetX] = computeAlignedBaseWidthAndOffset(op, rewriter);

    VectorType vecType = vec_ty(i32_ty, 2);
    Value byteCoord = b.insert_element(
        vecType,
        b.insert_element(vecType, b.undef(vecType), offsetX, b.i32_val(0)),
        op.getY(), b.i32_val(1));
    SmallVector<Type> argTypes{ptr_ty(ctx, 1), i32_ty,  i32_ty,
                               i32_ty,         vecType, ptr_ty(ctx)};

    SmallVector<Value> args{op.getPtr(),       baseWidth, op.getBaseHeight(),
                            op.getBasePitch(), byteCoord, dest};

    std::array<std::pair<unsigned, mlir::StringRef>, 4> paramAttrs{
        std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
        std::make_pair(0, LLVM::LLVMDialect::getReadonlyAttrName()),
        std::make_pair(5, LLVM::LLVMDialect::getNonNullAttrName()),
        std::make_pair(5, LLVM::LLVMDialect::getWriteOnlyAttrName()),
    };

    LLVM::CallOp call = intel::createDeviceFunctionCall(
        rewriter, fnName, void_ty(ctx), argTypes, args, paramAttrs,
        intel::noUnwindWillReturnAttrs);
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    VectorType storeValType = op.getStoredVal().getType();
    auto storeValPtr = rewriter.create<LLVM::AllocaOp>(
        loc, ptr_ty(ctx), storeValType.getElementType(),
        b.i32_val(storeValType.getNumElements()));
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

    auto [baseWidth, offsetX] = computeAlignedBaseWidthAndOffset(op, rewriter);

    VectorType vecType = vec_ty(i32_ty, 2);
    Value byteCoord = b.insert_element(
        vecType,
        b.insert_element(vecType, b.undef(vecType), offsetX, b.i32_val(0)),
        op.getY(), b.i32_val(1));
    SmallVector<Type> argTypes{ptr_ty(ctx, 1), i32_ty,  i32_ty,
                               i32_ty,         vecType, ptr_ty(ctx)};
    SmallVector<Value> args{op.getPtr(),       baseWidth, op.getBaseHeight(),
                            op.getBasePitch(), byteCoord, storeValPtr};

    std::array<std::pair<unsigned, mlir::StringRef>, 4> paramAttrs{
        std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
        std::make_pair(0, LLVM::LLVMDialect::getWriteOnlyAttrName()),
        std::make_pair(5, LLVM::LLVMDialect::getNonNullAttrName()),
        std::make_pair(5, LLVM::LLVMDialect::getReadonlyAttrName()),
    };

    LLVM::CallOp call = intel::createDeviceFunctionCall(
        rewriter, fnName, void_ty(ctx), argTypes, args, paramAttrs,
        intel::noUnwindWillReturnAttrs);
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    std::string fnName = "intel_sub_group_2d_block_prefetch_";
    fnName += std::to_string(op.getElemSizeInBits()) + "b_" +
              std::to_string(op.getTileHeight()) + "r" +
              std::to_string(op.getTileWidth()) + "x" +
              std::to_string(op.getVBlocks()) + "c";
    fnName = "_Z" + std::to_string(fnName.size()) + fnName + "PU3AS1viiiDv2_i";
    auto [baseWidth, offsetX] = computeAlignedBaseWidthAndOffset(op, rewriter);
    VectorType vecType = vec_ty(i32_ty, 2);
    Value byteCoord = b.insert_element(
        vecType,
        b.insert_element(vecType, b.undef(vecType), offsetX, b.i32_val(0)),
        op.getY(), b.i32_val(1));
    SmallVector<Type> argTypes{ptr_ty(ctx, 1), i32_ty, i32_ty, i32_ty, vecType};
    SmallVector<Value> args{op.getPtr(), baseWidth, op.getBaseHeight(),
                            op.getBasePitch(), byteCoord};

    std::array<std::pair<unsigned, mlir::StringRef>, 1> paramAttrs{
        std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
    };

    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::Ref,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = intel::noUnwindAttrs;
    funcAttrs.memEffectsAttr = memAttr;

    LLVM::CallOp call = intel::createDeviceFunctionCall(
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
                               OpType, TritonGEN::SubGroupBlockReadOp,
                               TritonGEN::SubGroupBlockWriteOp>::value>>
static std::string getSubGroupBlockManglingName(OpType op, Type type) {
  constexpr bool isWrite =
      std::is_same<OpType, TritonGEN::SubGroupBlockWriteOp>::value;
  const LLVM::LLVMPointerType ptrTy = op.getPtr().getType();
  // Note: OCL builtin name here differs from regular mangling.
  std::string funcName = "intel_sub_group_block_";
  if constexpr (isWrite)
    funcName += "write";
  else
    funcName += "read";
  Type elementType =
      TypeSwitch<Type, Type>(type)
          .Case([](VectorType vecType) { return vecType.getElementType(); })
          // Scalar case
          .Default(llvm::identity<Type>());
  const unsigned numElems =
      TypeSwitch<Type, unsigned>(type)
          .Case([](VectorType vecType) { return vecType.getNumElements(); })
          // Scalar case
          .Default(0u);
  funcName += "_u" + intel::getTypeMangling(elementType) +
              (numElems ? std::to_string(numElems) : "");
  funcName = "_Z" + std::to_string(funcName.size()) + funcName + "PU3AS" +
             std::to_string(ptrTy.getAddressSpace()) +
             intel::getTypeMangling(elementType, /*isUnsigned=*/true);
  if constexpr (isWrite)
    funcName += intel::getTypeMangling(type, /*isUnsigned=*/true);
  return funcName;
}

struct TritonSubGroupBlockReadLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SubGroupBlockReadOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::SubGroupBlockReadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SubGroupBlockReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::LLVMPointerType ptrTy = op.getPtr().getType();
    Type type = op.getRes().getType();

    std::string funcName = getSubGroupBlockManglingName(op, type);
    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::Ref,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = intel::noUnwindWillReturnAttrs;
    funcAttrs.memEffectsAttr = memAttr;
    LLVM::CallOp call = intel::createDeviceFunctionCall(
        rewriter, funcName, type, {ptrTy}, {op.getPtr()}, {}, funcAttrs, {});

    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

struct TritonSubGroupBlockWriteLowering
    : public ConvertOpToLLVMPattern<TritonGEN::SubGroupBlockWriteOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::SubGroupBlockWriteOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::SubGroupBlockWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    LLVM::LLVMPointerType ptrTy = op.getPtr().getType();
    Type type = op.getVal().getType();

    std::string funcName = getSubGroupBlockManglingName(op, type);

    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::ModRef,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = intel::noUnwindWillReturnAttrs;
    funcAttrs.memEffectsAttr = memAttr;
    LLVM::CallOp call = intel::createDeviceFunctionCall(
        rewriter, funcName, void_ty(ctx), {ptrTy, type},
        {op.getPtr(), op.getVal()}, {}, funcAttrs);

    rewriter.replaceOp(op, call);
    return success();
  }
};

struct TritonFToTf32OpLowering
    : public ConvertOpToLLVMPattern<TritonGEN::FToTf32Op> {
  using ConvertOpToLLVMPattern<TritonGEN::FToTf32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::FToTf32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    Value value = op->getOperand(0);
    SmallVector<Type> argTypes{f32_ty};
    SmallVector<Value> args{value};

    const StringLiteral funcName = "_Z25__spirv_RoundFToTF32INTELf";
    auto retType = f32_ty;
    auto callOp = intel::createDeviceFunctionCall(
        rewriter, funcName, retType, {argTypes}, {args}, {},
        intel::noUnwindWillReturnAttrs);
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
  patterns
      .add<TritonMatrixDPASLowering, TritonMatrix2DBlockLoadLowering,
           TritonMatrix2DBlockStoreLowering,
           TritonMatrix2DBlockPrefetchLowering, TritonSubGroupBlockReadLowering,
           TritonSubGroupBlockWriteLowering, TritonFToTf32OpLowering>(
          converter);
}

void registerConvertTritonTritonGENToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, TritonGEN::TritonGENDialect *dialect) {
        dialect->addInterfaces<TritonGENToLLVMDialectInterface>();
      });
}
