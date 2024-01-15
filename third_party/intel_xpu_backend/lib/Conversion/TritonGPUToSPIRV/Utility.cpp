#include "Utility.h"
#include "TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace mlir {

namespace spirv {
using namespace mlir::triton;

/// Returns true if the given `type` is a boolean scalar or vector type.
bool isBoolScalarOrVector(Type type) {
  assert(type && "Not a valid type");
  if (type.isInteger(1))
    return true;

  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType().isInteger(1);

  return false;
}

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

bool checkOpSupported(const std::map<std::string, std::any> &computeCapability,
                      std::string key) {
  auto option = computeCapability.find(key);
  if (option != computeCapability.end()) {
    return std::any_cast<bool>(option->second);
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

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(ConversionPatternRewriter &rewriter,
                               Location loc, Value linear,
                               ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = reorder(shape, order);
  SmallVector<Value> reorderedMultiDim(rank);
  if (auto constantOp = linear.getDefiningOp<arith::ConstantOp>()) {
    unsigned intVal =
        constantOp.getValue().cast<IntegerAttr>().getValue().getSExtValue();
    reorderedMultiDim = delinearize(rewriter, loc, intVal, reordered);
  } else {
    reorderedMultiDim = delinearize(rewriter, loc, linear, reordered);
  }
  SmallVector<Value> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

SmallVector<Value> delinearize(ConversionPatternRewriter &rewriter,
                               Location loc, unsigned linear,
                               ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  unsigned remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    unsigned dimSize = en.value();
    multiDim[en.index()] = i32_val(remained % dimSize);
    remained = remained / dimSize;
  }
  return multiDim;
}

SmallVector<Value> delinearize(ConversionPatternRewriter &rewriter,
                               Location loc, Value linear,
                               ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  Value remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    Value dimSize = i32_val(en.value());
    multiDim[en.index()] = urem(remained, dimSize);
    remained = udiv(remained, dimSize);
  }
  return multiDim;
}

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape,
                ArrayRef<unsigned> order) {
  return linearize(rewriter, loc, reorder<Value>(multiDim, order),
                   reorder<unsigned>(shape, order));
}

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape) {
  auto rank = multiDim.size();
  Value linear = i32_val(0);
  if (rank > 0) {
    linear = multiDim.back();
    for (auto [dim, dimShape] :
         llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
      Value dimSize = i32_val(dimShape);
      linear = add(mul(linear, dimSize), dim);
    }
  }
  return linear;
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

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Value pred) {
  auto ptrTy = ptr.getType().cast<spirv::PointerType>();
  auto retTy = ptrTy.getPointeeType();

  // scalar load
  // Create block structure for the masked load.
  auto *preheader = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  auto *tailblock = rewriter.splitBlock(preheader, opPosition);
  tailblock->addArgument(retTy, loc);
  auto *condblock = rewriter.createBlock(tailblock);

  // Test the mask
  rewriter.setInsertionPoint(preheader, preheader->end());

  // Prediction false to use the other value.
  Value other_ = undef(retTy);

  rewriter.create<mlir::cf::CondBranchOp>(loc, pred, condblock, tailblock,
                                          ValueRange{other_});

  // Do the load
  rewriter.setInsertionPoint(condblock, condblock->end());

  Value ret = rewriter.create<spirv::LoadOp>(loc, ptr);
  rewriter.create<mlir::cf::BranchOp>(loc, tailblock, ValueRange{ret});

  rewriter.setInsertionPoint(tailblock, tailblock->begin());

  ret = *tailblock->args_begin();
  return ret;
}

static Value commonShflSync(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, Value i,
                            const std::string &shuffleType) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = commonShflSync(loc, rewriter, val0, i, shuffleType);
    val1 = commonShflSync(loc, rewriter, val1, i, shuffleType);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }

  auto scope = rewriter.getAttr<spirv::ScopeAttr>(spirv::Scope::Subgroup);
  if (shuffleType == "up") {
    return rewriter.create<spirv::GroupNonUniformShuffleUpOp>(loc, scope, val,
                                                              i);
  } else if (shuffleType == "down") {
    return rewriter.create<spirv::GroupNonUniformShuffleDownOp>(loc, scope, val,
                                                                i);
  } else if (shuffleType == "xor") {
    return rewriter.create<spirv::GroupNonUniformShuffleXorOp>(loc, scope, val,
                                                               i);
  } else if (shuffleType == "idx") {
    return rewriter.create<spirv::GroupNonUniformShuffleOp>(loc, scope, val, i);
  } else {
    llvm_unreachable("Unknown shuffle type");
  }
}

Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
               int i) {
  return commonShflSync(loc, rewriter, val, i32_val(i), "xor");
}

Value shflUpSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i) {
  return commonShflSync(loc, rewriter, val, i32_val(i), "up");
}

Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  int i) {
  return commonShflSync(loc, rewriter, val, i32_val(i), "idx");
}

Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  Value i) {
  return commonShflSync(loc, rewriter, val, i, "idx");
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

  spirv::ConstantOp globalString;
  spirv::GlobalVariableOp globalVar;
  spirv::SpecConstantCompositeOp specCstComposite;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    llvm::SmallVector<Attribute, 8> contentStr;
    SmallString<25> specCstName;
    SmallString<15> contentWithNull(content);
    // must ends with null
    contentWithNull.push_back('\0');
    unsigned specConstIdx = 0;
    for (auto c : contentWithNull) {
      auto cAttr = rewriter.getI8IntegerAttr(c);
      (llvm::Twine(key) + "_speccst" + llvm::Twine(specConstIdx++))
          .toStringRef(specCstName);
      auto sc = rewriter.create<mlir::spirv::SpecConstantOp>(
          loc, rewriter.getStringAttr(specCstName), cAttr);
      contentStr.push_back(mlir::SymbolRefAttr::get(sc));
      specCstName.clear();
    }

    size_t contentSize = contentStr.size();
    auto globalType = spirv::ArrayType::get(i8_ty, contentSize);
    mlir::SmallString<20> specCstCompositeName;
    (llvm::Twine(key) + "_speccstcomp").toStringRef(specCstCompositeName);
    specCstComposite = rewriter.create<mlir::spirv::SpecConstantCompositeOp>(
        loc, mlir::TypeAttr::get(globalType),
        rewriter.getStringAttr(specCstCompositeName),
        rewriter.getArrayAttr(contentStr));

    globalVar = rewriter.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(ctx),
        ptr_ty(globalType, spirv::StorageClass::CrossWorkgroup),
        stringConstName, FlatSymbolRefAttr::get(specCstComposite));
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
    auto mask = fis_nan(val);
    // STEP1
    auto fp32_i32_value = bitcast(v, i32_ty);
    // STEP2
    val = lshr(fp32_i32_value, i32_val(16));
    // STEP3
    val = and_(val, int_val(32, 1));
    // STEP4
    auto rounding_bias = int_val(32, 0x7FFF);
    val = add(val, rounding_bias);
    // Step 5
    val = add(val, fp32_i32_value);
    // Step6
    val = lshr(val, int_val(32, 16));
    // val = rewriter.create<arith::TruncIOp>(loc, i16_ty, val);
    val = itrunc(i16_ty, val);
    val = bitcast(val, i16_ty);
    // If the value is NaN, return BF16 NaN.
    val = select(mask, int_val(16, 0xFFFF), val);
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

spirv::FuncOp getNearestFuncOp(Operation *from) {
  assert(from && "expected valid operation");
  if (isa<spirv::FuncOp>(from))
    return dyn_cast<spirv::FuncOp>(from);

  while (from->getParentOp()) {
    from = from->getParentOp();

    if (isa<spirv::FuncOp>(from))
      return dyn_cast<spirv::FuncOp>(from);
  }
  return nullptr;
}

spirv::FuncOp appendOrGetFuncOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                StringRef libName, StringRef funcName,
                                mlir::FunctionType funcType,
                                const NamedAttrList &extraAttrs) {
  spirv::FuncOp func =
      getNearestFuncOp(rewriter.getBlock()->getParent()->getParentOp());
  assert(func && "cannot find func op");
  auto funcAttr = StringAttr::get(rewriter.getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(func, funcAttr);
  if (funcOp)
    return cast<spirv::FuncOp>(*funcOp);

  mlir::OpBuilder b(func);
  NamedAttrList attributes(extraAttrs);
  attributes.set("libname", StringAttr::get(rewriter.getContext(), libName));
  attributes.set("libpath", StringAttr::get(rewriter.getContext(), ""));
  attributes.set(
      "linkage_attributes",
      ArrayAttr::get(rewriter.getContext(),
                     {
                         StringAttr::get(rewriter.getContext(), funcName),
                         StringAttr::get(rewriter.getContext(), "Import"),
                     }));
  auto ret = b.create<spirv::FuncOp>(
      loc, funcName, funcType, spirv::FunctionControl::Inline, attributes);
  return ret;
}

} // namespace spirv
} // namespace mlir
