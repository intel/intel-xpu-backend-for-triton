#include "ElementwiseOpToSPIRV.h"
#include "llvm/ADT/StringMap.h"
#include <string>

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::getTotalElemsPerThread;

static SmallVector<Value> reorderValues(const SmallVector<Value> &values,
                                        Type inType, Type ouType) {
  auto inTensorTy = inType.dyn_cast<RankedTensorType>();
  auto ouTensorTy = ouType.dyn_cast<RankedTensorType>();
  if (!inTensorTy || !ouTensorTy)
    return values;
  auto inEncoding =
      dyn_cast<triton::gpu::DotOperandEncodingAttr>(inTensorTy.getEncoding());
  auto ouEncoding =
      dyn_cast<triton::gpu::DotOperandEncodingAttr>(ouTensorTy.getEncoding());
  assert(inEncoding == ouEncoding);
  if (!inEncoding)
    return values;
  size_t inBitWidth = inTensorTy.getElementType().getIntOrFloatBitWidth();
  size_t ouBitWidth = ouTensorTy.getElementType().getIntOrFloatBitWidth();
  auto ouEltTy = ouTensorTy.getElementType();
  if (inBitWidth == ouBitWidth)
    return values;
  if (inBitWidth == 16 && ouBitWidth == 32) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 8) {
      ret.push_back(values[i]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
    }
    return ret;
  }
  if (inBitWidth == 8 && ouBitWidth == 16) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 16) {
      ret.push_back(values[i + 0]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 8]);
      ret.push_back(values[i + 9]);
      ret.push_back(values[i + 10]);
      ret.push_back(values[i + 11]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
      ret.push_back(values[i + 12]);
      ret.push_back(values[i + 13]);
      ret.push_back(values[i + 14]);
      ret.push_back(values[i + 15]);
    }
    return ret;
    // for (unsigned i = 0; i < values.size(); i += 16) {
    //   ret.push_back(values[i]);
    //   ret.push_back(values[i + 1]);
    //   ret.push_back(values[i + 4]);
    //   ret.push_back(values[i + 5]);
    //   ret.push_back(values[i + 8]);
    //   ret.push_back(values[i + 9]);
    //   ret.push_back(values[i + 12]);
    //   ret.push_back(values[i + 13]);

    //   ret.push_back(values[i + 2]);
    //   ret.push_back(values[i + 3]);
    //   ret.push_back(values[i + 6]);
    //   ret.push_back(values[i + 7]);
    //   ret.push_back(values[i + 10]);
    //   ret.push_back(values[i + 11]);
    //   ret.push_back(values[i + 14]);
    //   ret.push_back(values[i + 15]);
    // }
    // return values;
  }
  llvm_unreachable("unimplemented code path");
}

inline SmallVector<Value> unpackI32(const SmallVector<Value> &inValues,
                                    Type srcTy,
                                    ConversionPatternRewriter &rewriter,
                                    Location loc,
                                    TypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding =
      tensorTy.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<MmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  for (auto& v : inValues) {
    // cast i32 to appropriate eltType vector and extract elements
    auto eltType = typeConverter->convertType(tensorTy.getElementType());
    auto vecType = vec_ty(eltType, 32 / eltType.getIntOrFloatBitWidth());
    auto vec = bitcast(v, vecType);
    for (int i = 0; i < 32 / eltType.getIntOrFloatBitWidth(); i++) {
      outValues.push_back(extract_element(vec, i32_val(i)));
    }
  }
  return outValues;
}

inline SmallVector<Value> packI32(const SmallVector<Value> &inValues,
                                  Type srcTy,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc, TypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding =
      tensorTy.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<MmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  auto eltType = typeConverter->convertType(tensorTy.getElementType());
  int vecWidth = 32 / eltType.getIntOrFloatBitWidth();
  auto vecType = vec_ty(eltType, vecWidth);
  for (int i = 0; i < inValues.size(); i += vecWidth) {
    Value vec = undef(vecType);
    for (int j = 0; j < vecWidth; j++) {
      vec = insert_element(vec, inValues[i + j], i32_val(j));
    }
    outValues.push_back(bitcast(vec, i32_ty));
  }
  return outValues;
}

struct FpToFpOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::FpToFpOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::FpToFpOp>::ConvertTritonGPUOpToSPIRVPattern;

  typedef std::function<SmallVector<Value>(
      Location, ConversionPatternRewriter &, const Value &, const Value &,
      const Value &, const Value &)>
      ConvertorT;
  /* ------------------ */
  // FP8 -> FP16
  /* ------------------ */

  static SmallVector<Value>
  convertFp8x4ToFp16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const Value &v0, const Value &v1, const Value &v2,
                       const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static SmallVector<Value>
  convertFp8E4M3x4ToFp16x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static SmallVector<Value>
  convertFp8E5M2x4ToFp16x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  /* ------------------ */
  // FP8 -> BF16
  /* ------------------ */
  static SmallVector<Value>
  convertFp8x4ToBf16x4(Location loc, ConversionPatternRewriter &rewriter,
                       const char *ptxAsm, const Value &v0, const Value &v1,
                       const Value &v2, const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static SmallVector<Value>
  convertFp8E4M3x4ToBf16x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  };

  static SmallVector<Value>
  convertFp8E5M2x4ToBf16x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  };

  /* ------------------ */
  // FP16 -> FP8
  /* ------------------ */

  static SmallVector<Value>
  convertFp16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const char *ptxAsm, const Value &v0, const Value &v1,
                       const Value &v2, const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static SmallVector<Value>
  convertFp16x4ToFp8E4M3x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static SmallVector<Value>
  convertFp16x4ToFp8E5M2x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  /* ------------------ */
  // FP32 -> FP8
  /* ------------------ */

  static SmallVector<Value>
  convertFp32x4ToFp8E4M3x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static SmallVector<Value>
  convertFp32x4ToFp8E5M2x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  /* ------------------ */
  // BF16 -> FP8
  /* ------------------ */

  static SmallVector<Value>
  convertBf16x4ToFp8x4(Location loc, ConversionPatternRewriter &rewriter,
                       const char *ptxAsm, const Value &v0, const Value &v1,
                       const Value &v2, const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static SmallVector<Value>
  convertBf16x4ToFp8E4M3x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  };

  static SmallVector<Value>
  convertBf16x4ToFp8E5M2x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  /* ------------------ */
  // FP8 -> FP32
  /* ------------------ */

  static SmallVector<Value>
  convertFp8E4M3x4ToFp32x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static SmallVector<Value>
  convertFp8E5M2x4ToFp32x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  //

  static SmallVector<Value>
  convertFp8E4M3x4ToFp64x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static SmallVector<Value>
  convertFp64x4ToFp8E4M3x4(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v0, const Value &v1, const Value &v2,
                           const Value &v3) {
    llvm::report_fatal_error("SPIRV doesn't support FP8 yet");
  }

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return rewriter.create<spirv::FConvertOp>(loc, f32_ty, v);
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return rewriter.create<spirv::FConvertOp>(loc, f16_ty, v);
  }

  ConvertorT getConversionFunc(Type srcTy, Type dstTy) const {
    auto F8E4M3TyID = TypeID::get<mlir::Float8E4M3FNType>();
    auto F8E5M2TyID = TypeID::get<mlir::Float8E5M2Type>();
    auto F16TyID = TypeID::get<mlir::Float16Type>();
    auto BF16TyID = TypeID::get<mlir::BFloat16Type>();
    auto F32TyID = TypeID::get<mlir::Float32Type>();
    auto F64TyID = TypeID::get<mlir::Float64Type>();
    static DenseMap<std::pair<TypeID, TypeID>, ConvertorT> convertorMap = {
        // F8 -> F16
        {{F8E4M3TyID, F16TyID}, convertFp8E4M3x4ToFp16x4},
        {{F8E5M2TyID, F16TyID}, convertFp8E5M2x4ToFp16x4},
        // F16 -> F8
        {{F16TyID, F8E4M3TyID}, convertFp16x4ToFp8E4M3x4},
        {{F16TyID, F8E5M2TyID}, convertFp16x4ToFp8E5M2x4},
        // F8 -> BF16
        {{F8E4M3TyID, BF16TyID}, convertFp8E4M3x4ToBf16x4},
        {{F8E5M2TyID, BF16TyID}, convertFp8E5M2x4ToBf16x4},
        // BF16 -> F8
        {{BF16TyID, F8E4M3TyID}, convertBf16x4ToFp8E4M3x4},
        {{BF16TyID, F8E5M2TyID}, convertBf16x4ToFp8E5M2x4},
        // F8 -> F32
        {{F8E4M3TyID, F32TyID}, convertFp8E4M3x4ToFp32x4},
        {{F8E5M2TyID, F32TyID}, convertFp8E5M2x4ToFp32x4},
        // F32 -> F8
        {{F32TyID, F8E4M3TyID}, convertFp32x4ToFp8E4M3x4},
        {{F32TyID, F8E5M2TyID}, convertFp32x4ToFp8E5M2x4},
    };

    std::pair<TypeID, TypeID> key = {srcTy.getTypeID(), dstTy.getTypeID()};
    if (convertorMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to " << dstTy
                   << "\n";
      llvm_unreachable("");
    }
    return convertorMap.lookup(key);
  }

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // llvm::outs() << 0 << "\n";
    auto srcTensorType = op.getFrom().getType().cast<mlir::RankedTensorType>();
    auto dstTensorType =
        op.getResult().getType().cast<mlir::RankedTensorType>();
    auto loc = op->getLoc();
    // check that the number of elements is divisible by 4
    // Get convertor
    auto cvtFunc = getConversionFunc(srcTensorType.getElementType(),
                                     dstTensorType.getElementType());
    // Unpack value
    auto inVals = getTypeConverter()->unpackLLElements(loc, adaptor.getFrom(),
                                                       rewriter, srcTensorType);
    inVals =
        unpackI32(inVals, srcTensorType, rewriter, loc, getTypeConverter());
    // Cast
    SmallVector<Value> outVals;
    auto elems = inVals.size();
    assert(elems % 4 == 0 &&
           "FP8 casting only support tensors with 4-aligned sizes");
    for (size_t i = 0; i < elems; i += 4)
      outVals.append(cvtFunc(loc, rewriter, inVals[i], inVals[i + 1],
                             inVals[i + 2], inVals[i + 3]));
    // Pack values
    assert(outVals.size() == elems);
    outVals = reorderValues(outVals, srcTensorType, dstTensorType);
    outVals =
        packI32(outVals, dstTensorType, rewriter, loc, getTypeConverter());
    auto result = getTypeConverter()->packLLElements(loc, outVals, rewriter,
                                                     dstTensorType);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static spirv::FuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter,
                                         const Value &v, StringRef libName,
                                         StringRef funcName,
                                         mlir::FunctionType funcType,
                                         const NamedAttrList &extraAttrs = {}) {
    auto funcAttr = StringAttr::get(v.getContext(), funcName);
    Operation *funcOp =
        SymbolTable::lookupNearestSymbolFrom(v.getDefiningOp(), funcAttr);
    if (funcOp)
      return cast<spirv::FuncOp>(*funcOp);

    mlir::OpBuilder b(v.getDefiningOp()->getParentOfType<spirv::FuncOp>());
    NamedAttrList attributes(extraAttrs);
    attributes.set("libname", StringAttr::get(v.getContext(), libName));
    attributes.set("libpath", StringAttr::get(v.getContext(), ""));
    attributes.set("linkage_attributes",
                   ArrayAttr::get(v.getContext(),
                                  {
                                      StringAttr::get(v.getContext(), funcName),
                                      StringAttr::get(v.getContext(), "Import"),
                                  }));
    auto ret =
        b.create<spirv::FuncOp>(v.getLoc(), funcName, funcType,
                                spirv::FunctionControl::Inline, attributes);
    return ret;
  }
};

template <typename SourceOp, typename ConcreteT>
class ElementwiseOpSPIRVConversionBase
    : public ConvertTritonGPUOpToSPIRVPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpSPIRVConversionBase(
      TritonGPUToSPIRVTypeConverter &converter, MLIRContext *context,
      PatternBenefit benefit = 1, bool use_INTELConvertFToBF16Op = false)
      : ConvertTritonGPUOpToSPIRVPattern<SourceOp>(converter, context, benefit),
        use_INTELConvertFToBF16Op(use_INTELConvertFToBF16Op) {}

  bool use_INTELConvertFToBF16Op = false;
  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();
    // element type
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<Value> resultVals;
    //
    SmallVector<SmallVector<Value>> allOperands;
    auto operands = adaptor.getOperands();
    for (const auto& operand : operands) {
      auto argTy = op->getOperand(0).getType();
      auto sub_operands = this->getTypeConverter()->unpackLLElements(
          loc, operand, rewriter, argTy);
      sub_operands = unpackI32(sub_operands, argTy, rewriter, loc,
                               this->getTypeConverter());
      allOperands.resize(sub_operands.size());
      auto vs = llvm::enumerate(sub_operands);
      for (const auto& v : vs)
        allOperands[v.index()].push_back(v.value());
    }
    if (allOperands.size() == 0)
      allOperands.push_back({});
    for (const SmallVector<Value> &operands : allOperands) {
      Value curr =
          ((ConcreteT *)(this))
              ->createDestOp(op, adaptor, rewriter, elemTy, operands, loc);
      if (!bool(curr))
        return failure();
      resultVals.push_back(curr);
    }
    if (op->getNumOperands() > 0) {
      auto argTy = op->getOperand(0).getType();
      resultVals = reorderValues(resultVals, argTy, resultTy);
    }
    resultVals =
        packI32(resultVals, resultTy, rewriter, loc, this->getTypeConverter());
    Value view = this->getTypeConverter()->packLLElements(loc, resultVals,
                                                          rewriter, resultTy);
    rewriter.replaceOp(op, view);

    return success();
  }
};

template <typename SourceOp, typename DestOp>
struct ElementwiseOpSPIRVConversion
    : public ElementwiseOpSPIRVConversionBase<
          SourceOp, ElementwiseOpSPIRVConversion<SourceOp, DestOp>> {
  using Base = ElementwiseOpSPIRVConversionBase<
      SourceOp, ElementwiseOpSPIRVConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  DestOp createDestOp(SourceOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter, Type elemTy,
                      ValueRange operands, Location loc) const {
    return rewriter.create<DestOp>(loc, elemTy, operands,
                                   adaptor.getAttributes().getValue());
  }
};

/// Returns true if the given `type` is a boolean scalar or vector type.
static bool isBoolScalarOrVector(Type type) {
  assert(type && "Not a valid type");
  if (type.isInteger(1))
    return true;

  if (auto vecType = type.dyn_cast<VectorType>())
    return vecType.getElementType().isInteger(1);

  return false;
}

struct CmpIOpSPIRVConversion
    : public ElementwiseOpSPIRVConversionBase<triton::gpu::CmpIOp,
                                              CmpIOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<triton::gpu::CmpIOp,
                                                CmpIOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  Value createDestOp(triton::gpu::CmpIOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {

    Type oprandType =
        this->getTypeConverter()->convertType(operands[0].getType());
    switch (op.getPredicate()) {

#define DISPATCH_WITH_LOGICAL(cmpPredicate, spirvOp, spirvLogicOp)             \
  case cmpPredicate:                                                           \
    if (isBoolScalarOrVector(oprandType)) {                                    \
      return rewriter.create<spirvLogicOp>(loc, operands[0], operands[1]);     \
    } else {                                                                   \
      return rewriter.create<spirvOp>(loc, operands[0], operands[1]);          \
    }

      DISPATCH_WITH_LOGICAL(arith::CmpIPredicate::eq, spirv::IEqualOp,
                            spirv::LogicalEqualOp);
      DISPATCH_WITH_LOGICAL(arith::CmpIPredicate::ne, spirv::INotEqualOp,
                            spirv::LogicalNotEqualOp);
#undef DISPATCH_WITH_LOGICAL

#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    return rewriter.create<spirvOp>(loc, operands[0], operands[1]);

      DISPATCH(arith::CmpIPredicate::slt, spirv::SLessThanOp);
      DISPATCH(arith::CmpIPredicate::sle, spirv::SLessThanEqualOp);
      DISPATCH(arith::CmpIPredicate::sgt, spirv::SGreaterThanOp);
      DISPATCH(arith::CmpIPredicate::sge, spirv::SGreaterThanEqualOp);
      DISPATCH(arith::CmpIPredicate::ult, spirv::ULessThanOp);
      DISPATCH(arith::CmpIPredicate::ule, spirv::ULessThanEqualOp);
      DISPATCH(arith::CmpIPredicate::ugt, spirv::UGreaterThanOp);
      DISPATCH(arith::CmpIPredicate::uge, spirv::UGreaterThanEqualOp);

#undef DISPATCH

    default:
      break;
    }
    return nullptr;
  }
};

struct CmpFOpSPIRVConversion
    : public ElementwiseOpSPIRVConversionBase<triton::gpu::CmpFOp,
                                              CmpFOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<triton::gpu::CmpFOp,
                                                CmpFOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  Value createDestOp(triton::gpu::CmpFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    switch (op.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    return rewriter.create<spirvOp>(loc, operands[0], operands[1]);

      // Ordered.
      DISPATCH(arith::CmpFPredicate::OEQ, spirv::FOrdEqualOp);
      DISPATCH(arith::CmpFPredicate::OGT, spirv::FOrdGreaterThanOp);
      DISPATCH(arith::CmpFPredicate::OGE, spirv::FOrdGreaterThanEqualOp);
      DISPATCH(arith::CmpFPredicate::OLT, spirv::FOrdLessThanOp);
      DISPATCH(arith::CmpFPredicate::OLE, spirv::FOrdLessThanEqualOp);
      DISPATCH(arith::CmpFPredicate::ONE, spirv::FOrdNotEqualOp);
      // Unordered.
      DISPATCH(arith::CmpFPredicate::UEQ, spirv::FUnordEqualOp);
      DISPATCH(arith::CmpFPredicate::UGT, spirv::FUnordGreaterThanOp);
      DISPATCH(arith::CmpFPredicate::UGE, spirv::FUnordGreaterThanEqualOp);
      DISPATCH(arith::CmpFPredicate::ULT, spirv::FUnordLessThanOp);
      DISPATCH(arith::CmpFPredicate::ULE, spirv::FUnordLessThanEqualOp);
      DISPATCH(arith::CmpFPredicate::UNE, spirv::FUnordNotEqualOp);

#undef DISPATCH

    default:
      break;
    }
    return nullptr;
  }
};

template <class T>
struct ExternElementwiseSPIRVConversion
    : public ElementwiseOpSPIRVConversionBase<
          T, ExternElementwiseSPIRVConversion<T>> {
  using Base =
      ElementwiseOpSPIRVConversionBase<T, ExternElementwiseSPIRVConversion<T>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  llvm::StringMap<std::string> imfMapping{{"isinfd", "isinf"},
                                          {"isnand", "isnan"},
                                          {"powif", "pownf"},
                                          {"powi", "pown"}};

  Value createDestOp(T op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    StringRef symbol = op.getSymbol();
    if (symbol.empty())
      llvm::errs() << "ExternElementwiseOpConversion";

    // TODO: move the prefix changing to a bridge lib.
    std::string funcName;
    if (symbol.consume_front("__nv_")) {
      if (imfMapping.contains(symbol.str())) {
        funcName = "__devicelib_imf_" + imfMapping.at(symbol.str());
      } else {
        funcName = "__devicelib_imf_" + symbol.str();
      }
    } else {
      funcName = symbol.str();
    }

    mlir::FunctionType funcType = getFunctionType(elemTy, operands);

    spirv::FuncOp funcOp = appendOrGetFuncOp(rewriter, op, funcName, funcType);

    return rewriter
        .create<spirv::FunctionCallOp>(loc, elemTy, funcName, operands)
        .getResult(0);
  }

private:
  mlir::FunctionType getFunctionType(Type resultType,
                                     ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return mlir::FunctionType::get(this->getContext(), operandTypes,
                                   resultType);
  }

  spirv::FuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter, T op,
                                  StringRef funcName,
                                  mlir::FunctionType funcType) const {
    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    if (funcOp)
      return cast<spirv::FuncOp>(*funcOp);

    auto parent = ((Operation *)op)->getParentOfType<spirv::FuncOp>();
    mlir::OpBuilder b(parent);
    auto ret = b.create<spirv::FuncOp>(op->getLoc(), funcName, funcType);
    ret.getOperation()->setAttr(
        "libname", StringAttr::get(op->getContext(), op.getLibname()));
    ret.getOperation()->setAttr(
        "libpath", StringAttr::get(op->getContext(), op.getLibpath()));
    ret.getOperation()->setAttr(
        "linkage_attributes",
        ArrayAttr::get(op->getContext(),
                       {
                           StringAttr::get(op->getContext(), funcName),
                           StringAttr::get(op->getContext(), "Import"),
                       }));
    return ret;
  }
};

struct BitcastOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<triton::BitcastOp,
                                       BitcastOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<triton::BitcastOp,
                                                BitcastOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(triton::BitcastOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    // a safety bitcast that checks the input type and the output type follows
    // the SPIRV dialect rule.
    return bitcast(operands[0], elemTy);
  }
};

template <typename SourceOp>
struct ZeroFillShiftOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<
          SourceOp, ZeroFillShiftOpSPIRVConversion<SourceOp>> {
  using Base = ElementwiseOpSPIRVConversionBase<
      SourceOp, ZeroFillShiftOpSPIRVConversion<SourceOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(SourceOp op, Adaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    // we need to align PTX semantic for shift op, which has clamps.
    auto base = operands[0];
    auto shift = operands[1];
    auto bw =
        int_val(elemTy.getIntOrFloatBitWidth(), elemTy.getIntOrFloatBitWidth());
    auto zero = int_val(elemTy.getIntOrFloatBitWidth(), 0);
    auto shiftVal = rewriter.create<SourceOp>(loc, elemTy, base, shift);
    return select(icmp_ult(shift, bw), shiftVal, zero);
  }
};

template <typename SourceOp>
struct SignFillShiftOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<
          SourceOp, SignFillShiftOpSPIRVConversion<SourceOp>> {
  using Base = ElementwiseOpSPIRVConversionBase<
      SourceOp, SignFillShiftOpSPIRVConversion<SourceOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(SourceOp op, Adaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    // we need to align PTX semantic for shift op, which has clamps.
    auto base = operands[0];
    auto shift = operands[1];
    auto bw =
        int_val(elemTy.getIntOrFloatBitWidth(), elemTy.getIntOrFloatBitWidth());
    shift = select(icmp_ult(shift, bw), shift,
                   int_val(elemTy.getIntOrFloatBitWidth(),
                           elemTy.getIntOrFloatBitWidth() - 1));
    return rewriter.create<SourceOp>(loc, elemTy, base, shift);
  }
};

struct FDivOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::DivFOp,
                                       FDivOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<mlir::arith::DivFOp,
                                                FDivOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::DivFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      auto lhs = mlir::spirv::convertBf16ToFp32(
          loc, rewriter, operands[0], this->use_INTELConvertFToBF16Op);
      auto rhs = mlir::spirv::convertBf16ToFp32(
          loc, rewriter, operands[1], this->use_INTELConvertFToBF16Op);
      auto f32_result = rewriter.create<spirv::FDivOp>(loc, lhs, rhs);
      return mlir::spirv::convertFp32ToBf16(loc, rewriter, f32_result,
                                            this->use_INTELConvertFToBF16Op);
    } else {
      return rewriter.create<spirv::FDivOp>(loc, elemTy, operands[0],
                                            operands[1]);
    }
  }
};

struct FMulOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::MulFOp,
                                       FMulOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<mlir::arith::MulFOp,
                                                FMulOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::MulFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      auto lhs = mlir::spirv::convertBf16ToFp32(
          loc, rewriter, operands[0], this->use_INTELConvertFToBF16Op);
      auto rhs = mlir::spirv::convertBf16ToFp32(
          loc, rewriter, operands[1], this->use_INTELConvertFToBF16Op);
      auto f32_result = rewriter.create<spirv::FMulOp>(loc, lhs, rhs);
      return mlir::spirv::convertFp32ToBf16(loc, rewriter, f32_result,
                                            this->use_INTELConvertFToBF16Op);
    } else {
      return rewriter.create<spirv::FMulOp>(loc, elemTy, operands[0],
                                            operands[1]);
    }
  }
};

struct FAddOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::AddFOp,
                                       FAddOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<mlir::arith::AddFOp,
                                                FAddOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::AddFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      auto lhs = mlir::spirv::convertBf16ToFp32(
          loc, rewriter, operands[0], this->use_INTELConvertFToBF16Op);
      auto rhs = mlir::spirv::convertBf16ToFp32(
          loc, rewriter, operands[1], this->use_INTELConvertFToBF16Op);
      auto f32_result = rewriter.create<spirv::FAddOp>(loc, lhs, rhs);
      return mlir::spirv::convertFp32ToBf16(loc, rewriter, f32_result,
                                            this->use_INTELConvertFToBF16Op);
    } else {
      return rewriter.create<spirv::FAddOp>(loc, elemTy, operands[0],
                                            operands[1]);
    }
  }
};

struct FSubOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::SubFOp,
                                       FSubOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<mlir::arith::SubFOp,
                                                FSubOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::SubFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      auto lhs = mlir::spirv::convertBf16ToFp32(
          loc, rewriter, operands[0], this->use_INTELConvertFToBF16Op);
      auto rhs = mlir::spirv::convertBf16ToFp32(
          loc, rewriter, operands[1], this->use_INTELConvertFToBF16Op);
      auto f32_result = rewriter.create<spirv::FSubOp>(loc, lhs, rhs);
      return mlir::spirv::convertFp32ToBf16(loc, rewriter, f32_result,
                                            this->use_INTELConvertFToBF16Op);
    } else {
      return rewriter.create<spirv::FSubOp>(loc, elemTy, operands[0],
                                            operands[1]);
    }
  }
};

struct SIToFPOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::SIToFPOp,
                                       SIToFPOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<mlir::arith::SIToFPOp,
                                                SIToFPOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::SIToFPOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto value = rewriter.create<arith::SIToFPOp>(loc, f32_ty, operands[0]);
      return mlir::spirv::convertFp32ToBf16(loc, rewriter, value,
                                            this->use_INTELConvertFToBF16Op);
    } else {
      return rewriter.create<arith::SIToFPOp>(loc, elemTy, operands[0]);
    }
  }
};

struct FPToSIOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::FPToSIOp,
                                       FPToSIOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<mlir::arith::FPToSIOp,
                                                FPToSIOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::FPToSIOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value = mlir::spirv::convertBf16ToFp32(
          loc, rewriter, operands[0], this->use_INTELConvertFToBF16Op);
      return rewriter.create<arith::FPToSIOp>(loc, elemTy, value);
    } else {
      return rewriter.create<arith::FPToSIOp>(loc, elemTy, operands[0]);
    }
  }
};

struct ExtFOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::ExtFOp,
                                       ExtFOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<mlir::arith::ExtFOp,
                                                ExtFOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::ExtFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return mlir::spirv::convertBf16ToFp32(loc, rewriter, operands[0],
                                            this->use_INTELConvertFToBF16Op);
    } else {
      return rewriter.create<arith::ExtFOp>(loc, elemTy, operands[0]);
    }
  }
};

struct TruncFOpSPIRVConversion
    : ElementwiseOpSPIRVConversionBase<mlir::arith::TruncFOp,
                                       TruncFOpSPIRVConversion> {
  using Base = ElementwiseOpSPIRVConversionBase<mlir::arith::TruncFOp,
                                                TruncFOpSPIRVConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::arith::TruncFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto inElemTy = getElementType(op.getIn());
      assert(inElemTy.isF32() && "unsupported conversion");
      return mlir::spirv::convertFp32ToBf16(loc, rewriter, operands[0],
                                            this->use_INTELConvertFToBF16Op);
    } else {
      return rewriter.create<arith::TruncFOp>(loc, elemTy, operands[0]);
    }
  }
};

struct ExpOpSPIRVConversionApprox
    : ElementwiseOpSPIRVConversionBase<mlir::math::ExpOp,
                                       ExpOpSPIRVConversionApprox> {
  using Base = ElementwiseOpSPIRVConversionBase<mlir::math::ExpOp,
                                                ExpOpSPIRVConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::math::ExpOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {
    // Use spirv.Cl.exp to calculate exponentials.
    return rewriter.create<spirv::CLExpOp>(loc, elemTy, operands[0]);
  }
};

struct AbsFOpConversion
    : ElementwiseOpSPIRVConversionBase<mlir::math::AbsFOp, AbsFOpConversion> {
  using Base =
      ElementwiseOpSPIRVConversionBase<mlir::math::AbsFOp, AbsFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  Value createDestOp(mlir::math::AbsFOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter, Type elemTy,
                     ValueRange operands, Location loc) const {

    if (llvm::isa<IntegerType>(elemTy)) {
      // Mask out the sign bit
      auto num_bits =
          getElementTypeOrSelf(op.getType()).getIntOrFloatBitWidth();
      assert(num_bits <= 16);
      auto mask = (1u << (num_bits - 1u)) - 1u;
      auto maskAttr = rewriter.getIntegerAttr(elemTy, mask);
      auto maskConst =
          rewriter.create<spirv::ConstantOp>(loc, elemTy, maskAttr);
      return and_(operands[0], maskConst);
    }

    return rewriter.create<mlir::math::AbsFOp>(loc, elemTy, operands[0]);
  }
};

void populateElementwiseOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation *allocation,
    Value smem, PatternBenefit benefit,
    std::map<std::string, int> &computeCapability) {

#define POPULATE_TERNARY_OP(SRC_OP, DST_OP)                                    \
  patterns.add<ElementwiseOpSPIRVConversion<SRC_OP, DST_OP>>(                  \
      typeConverter, context, benefit);
  POPULATE_TERNARY_OP(triton::gpu::SelectOp, spirv::SelectOp)
#undef POPULATE_TERNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpSPIRVConversion<SRC_OP, DST_OP>>(                  \
      typeConverter, context, benefit);
  POPULATE_BINARY_OP(arith::SubIOp, spirv::ISubOp) // -
  POPULATE_BINARY_OP(arith::AddIOp, spirv::IAddOp) // +
  POPULATE_BINARY_OP(arith::MulIOp, spirv::IMulOp) // *
  POPULATE_BINARY_OP(arith::DivSIOp, spirv::SDivOp)
  POPULATE_BINARY_OP(arith::DivUIOp, spirv::UDivOp)
  POPULATE_BINARY_OP(arith::RemFOp, spirv::FRemOp) // %
  POPULATE_BINARY_OP(arith::RemSIOp, spirv::SRemOp)
  POPULATE_BINARY_OP(arith::RemUIOp, spirv::UModOp)
  POPULATE_BINARY_OP(arith::AndIOp, arith::AndIOp) // &
  POPULATE_BINARY_OP(arith::OrIOp, arith::OrIOp)   // |
  POPULATE_BINARY_OP(arith::XOrIOp, arith::XOrIOp) // ^
#undef POPULATE_BINARY_OP

#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpSPIRVConversion<SRC_OP, DST_OP>>(                  \
      typeConverter, context, benefit);
  POPULATE_UNARY_OP(arith::TruncIOp, arith::TruncIOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, arith::ExtSIOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, arith::ExtUIOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, arith::FPToUIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, arith::UIToFPOp)
  POPULATE_UNARY_OP(math::AbsIOp, math::AbsIOp)
  POPULATE_UNARY_OP(math::LogOp, math::LogOp)
  POPULATE_UNARY_OP(math::CosOp, math::CosOp)
  POPULATE_UNARY_OP(math::SinOp, math::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, math::SqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)
  POPULATE_UNARY_OP(triton::BitcastOp, spirv::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, spirv::BitcastOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, spirv::BitcastOp)
#undef POPULATE_UNARY_OP

  patterns.add<CmpIOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<CmpFOpSPIRVConversion>(typeConverter, context, benefit);

  patterns.add<ZeroFillShiftOpSPIRVConversion<arith::ShLIOp>>(typeConverter,
                                                              context, benefit);
  patterns.add<ZeroFillShiftOpSPIRVConversion<arith::ShRUIOp>>(
      typeConverter, context, benefit);
  patterns.add<SignFillShiftOpSPIRVConversion<arith::ShRSIOp>>(
      typeConverter, context, benefit);

  patterns.add<BitcastOpSPIRVConversion>(typeConverter, context, benefit);

  patterns.add<AbsFOpConversion>(typeConverter, context, benefit);
  patterns.add<FDivOpSPIRVConversion>(
      typeConverter, context, benefit,
      mlir::spirv::checkOpSupported(computeCapability,
                                    "INTELConvertFToBF16Op"));
  patterns.add<FSubOpSPIRVConversion>(
      typeConverter, context, benefit,
      mlir::spirv::checkOpSupported(computeCapability,
                                    "INTELConvertFToBF16Op"));
  patterns.add<FAddOpSPIRVConversion>(
      typeConverter, context, benefit,
      mlir::spirv::checkOpSupported(computeCapability,
                                    "INTELConvertFToBF16Op"));
  patterns.add<FMulOpSPIRVConversion>(
      typeConverter, context, benefit,
      mlir::spirv::checkOpSupported(computeCapability,
                                    "INTELConvertFToBF16Op"));

  patterns.add<ExtFOpSPIRVConversion>(
      typeConverter, context, benefit,
      mlir::spirv::checkOpSupported(computeCapability,
                                    "INTELConvertFToBF16Op"));
  patterns.add<TruncFOpSPIRVConversion>(
      typeConverter, context, benefit,
      mlir::spirv::checkOpSupported(computeCapability,
                                    "INTELConvertFToBF16Op"));
  patterns.add<FPToSIOpSPIRVConversion>(
      typeConverter, context, benefit,
      mlir::spirv::checkOpSupported(computeCapability,
                                    "INTELConvertFToBF16Op"));
  patterns.add<SIToFPOpSPIRVConversion>(
      typeConverter, context, benefit,
      mlir::spirv::checkOpSupported(computeCapability,
                                    "INTELConvertFToBF16Op"));

  patterns.add<FpToFpOpSPIRVConversion>(typeConverter, context, benefit);

  patterns
      .add<ExternElementwiseSPIRVConversion<triton::PureExternElementwiseOp>>(
          typeConverter, context, benefit);
  patterns
      .add<ExternElementwiseSPIRVConversion<triton::ImpureExternElementwiseOp>>(
          typeConverter, context, benefit);
  // ExpOpSPIRVConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpSPIRVConversionApprox will return failure
  // and ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will
  // call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpSPIRVConversionApprox>(typeConverter, context, benefit);
}
