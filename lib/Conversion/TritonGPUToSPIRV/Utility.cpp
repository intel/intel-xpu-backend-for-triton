#include "Utility.h"
#include "TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace mlir {

namespace spirv {
using namespace mlir::triton;

Value createConstantI32(Location loc, PatternRewriter &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<spirv::ConstantOp>(loc, i32ty,
                                            IntegerAttr::get(i32ty, v));
}

Value createConstantF32(Location loc, PatternRewriter &rewriter, float v) {
  auto type = type::f32Ty(rewriter.getContext());
  return rewriter.create<spirv::ConstantOp>(loc, type,
                                            rewriter.getF32FloatAttr(v));
}

Value createConstantF64(Location loc, PatternRewriter &rewriter, float v) {
  auto type = type::f64Ty(rewriter.getContext());
  return rewriter.create<spirv::ConstantOp>(loc, type,
                                            rewriter.getF64FloatAttr(v));
}

// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  return builder.create<spirv::ConstantOp>(loc, ty,
                                           builder.getIntegerAttr(ty, value));
}

// Create an integer constant of \param width bits.
Value createSPIRVIntegerConstant(OpBuilder &builder, Location loc, short width,
                                 int64_t value) {
  Type ty = builder.getIntegerType(width);
  return builder.create<spirv::ConstantOp>(loc, ty,
                                           builder.getIntegerAttr(ty, value));
}

bool checkOpSupported(std::map<std::string, int> computeCapability,
                      std::string dtype) {
  // TODO: For now, we define the computeCapability with {dtype, int}.
  // If it is >= 1, then the special op is supported.
  if (computeCapability.find(dtype) != computeCapability.end() &&
      computeCapability.at(dtype) >= 1) {
    return true;
  }
  return false;
}

SharedMemoryObject
getSharedMemoryObjectFromStruct(Location loc, Value spirvStruct,
                                ConversionPatternRewriter &rewriter) {
  auto types =
      spirvStruct.getType().cast<spirv::StructType>().getElementTypes();
  SmallVector<Value> elems(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    elems[i] = extract_val(type, spirvStruct, rewriter.getI32ArrayAttr(i));
  }

  auto rank = (elems.size() - 1) / 2;
  return {/*base=*/elems[0],
          /*strides=*/{elems.begin() + 1, elems.begin() + 1 + rank},
          /*offsets=*/{elems.begin() + 1 + rank, elems.end()}};
}

SmallVector<Value>
getStridesFromShapeAndOrder(ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
                            Location loc, ConversionPatternRewriter &rewriter) {
  auto rank = shape.size();
  SmallVector<Value> strides(rank);
  int64_t stride = 1;
  for (auto idx : order) {
    strides[idx] = i32_val(stride);
    stride *= shape[idx];
  }
  return strides;
}

void storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Value val, Value pred) {
  // scalar store
  // Create block structure for the masked load.
  auto *preheader = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  auto *tailblock = rewriter.splitBlock(preheader, opPosition);
  auto *condblock = rewriter.createBlock(tailblock);

  // Test the mask
  rewriter.setInsertionPoint(preheader, preheader->end());
  rewriter.create<mlir::cf::CondBranchOp>(loc, pred, condblock, tailblock);

  // Do the Store
  rewriter.setInsertionPoint(condblock, condblock->end());

  rewriter.create<spirv::StoreOp>(loc, ptr, val);
  rewriter.create<mlir::cf::BranchOp>(loc, tailblock);

  rewriter.setInsertionPoint(tailblock, tailblock->begin());
}

Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
               int i) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = shflSync(loc, rewriter, val0, i);
    val1 = shflSync(loc, rewriter, val1, i);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }

  auto scope = rewriter.getAttr<spirv::ScopeAttr>(spirv::Scope::Subgroup);
  return rewriter.create<spirv::GroupNonUniformShuffleXorOp>(loc, scope, val,
                                                             i32_val(i));
}

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content) {
  auto funcOp =
      rewriter.getBlock()->getParent()->getParentOfType<spirv::FuncOp>();
  assert(funcOp);
  auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallVector<Attribute, 8> contentStr;
  for (auto c : content) {
    auto cAttr = rewriter.getI8IntegerAttr(c);
    contentStr.push_back(cAttr);
  }
  size_t contentSize = contentStr.size();
  auto globalType = spirv::ArrayType::get(i8_ty, contentSize);

  spirv::ConstantOp globalString;
  spirv::GlobalVariableOp globalVar;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    VectorType _vec_type = VectorType::get({(int64_t)contentSize}, i8_ty);
    DenseElementsAttr dstElementsAttr =
        DenseElementsAttr::get(_vec_type, contentStr);
    globalString = rewriter.create<spirv::ConstantOp>(
        UnknownLoc::get(ctx), globalType, dstElementsAttr);

    globalVar = rewriter.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(ctx),
        ptr_ty(globalType, spirv::StorageClass::CrossWorkgroup),
        stringConstName,
        //        FlatSymbolRefAttr::get(globalString));
        nullptr);
  }

  Value zero = i32_val(0);
  Value globalPtr = rewriter.create<spirv::AddressOfOp>(
      UnknownLoc::get(rewriter.getContext()), globalVar);
  Value stringStart = rewriter.create<spirv::PtrAccessChainOp>(
      UnknownLoc::get(ctx), ptr_ty(i8_ty, spirv::StorageClass::CrossWorkgroup),
      globalPtr, zero, zero);
  Value genericStringStart =
      bitcast(stringStart, ptr_ty(i8_ty, spirv::StorageClass::Generic));
  return genericStringStart;
}

Value convertFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        const Value &v, bool use_INTELConvertFToBF16Op) {
  if (use_INTELConvertFToBF16Op) {
    // If support, then convert to bf16 using the INTELConvertFToBF16Op
    return rewriter.create<spirv::INTELConvertFToBF16Op>(loc, bf16_ty, v);
  } else {
    // Otherwise, Convert the FP32 value by RNE(Rounding to Nearest Even).
    // Algorithm is as follows:
    //   STEP1: U32_VAL = BITCAST(F32_VAL)
    //   STEP2: U32_VAL_TMP = U32_VAL >> 16
    //   STEP3: U32_VAL_TMP = U32_VAL_TMP & 1
    //   STEP4: ROUNDING_BIAS = U32_VAL_TMP + UINT32(0x7FFF)
    //   STEP5: U32_VAL_TMP = U32_VAL + ROUNDING_BIAS
    //   STEP6: BF16_VAL = static_cast<UINT16>(U32_VAL_TMP >> 16)
    Value val = v;
    auto mask = fcmp_oeq(val, val);
    // STEP1
    auto fp32_i32_value = bitcast(v, i32_ty);
    // STEP2
    val = lshr(fp32_i32_value, i32_val(16));
    // val = rewriter.create<arith::TruncIOp>(loc, i16_ty, val);
    val = itrunc(i16_ty, val);
    // STEP3
    val = and_(val, int_val(16, 1));
    // STEP4
    auto rounding_bias = int_val(16, 0x7FF);
    val = add(val, rounding_bias);
    val = zext(i32_ty, val);
    // Step 5
    val = add(val, fp32_i32_value);
    // Step6
    val = lshr(val, int_val(32, 16));
    // val = rewriter.create<arith::TruncIOp>(loc, i16_ty, val);
    val = itrunc(i16_ty, val);
    val = bitcast(val, i16_ty);
    // If the value is NaN, return BF16 NaN.
    val = select(mask, val, int_val(16, 0xFFFF));
    return val;
  }
}

Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                        const Value &v, bool use_INTELConvertFToBF16Op) {
  if (use_INTELConvertFToBF16Op) {
    return rewriter.create<spirv::INTELConvertBF16ToFOp>(loc, f32_ty, v);
  } else {
    Value val = v;
    val = zext(i32_ty, val);
    val = shl(val, i32_val(16));
    val = bitcast(val, f32_ty);
    return val;
  }
}

} // namespace spirv
} // namespace mlir
