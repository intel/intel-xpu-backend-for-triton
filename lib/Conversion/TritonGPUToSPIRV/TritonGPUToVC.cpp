#include "TritonGPUToVC.h"
#include "Utility.h"
// #include "llvm/Support/Debug.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

VectorType convertAndKeepShape(RankedTensorType type) {
  return VectorType::get(type.getShape(), type.getElementType());
}

/// @brief encodeVectorType(xxx, 8x8x2xf16, false) returns ["v64i32", 64xi32]
std::pair<std::string, VectorType>
encodeVectorType(ConversionPatternRewriter &rewriter, VectorType type,
                 bool use64bitData = false, bool enforceInteger = false) {
  auto elemType = type.getElementType();
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  int size = type.getNumElements() * bitWidth / 32;
  if (use64bitData) {
    size /= 2;
  }
  std::string str;
  switch (size) {
  case 16:
    str += "v16";
    break;
  case 32:
    str += "v32";
    break;
  case 64:
    str += "v64";
    break;
  case 128:
    str += "v128";
    break;
  case 256:
    str += "v256";
    break;
  default:
    assert(0 && "add more support");
    break;
  }
  if (use64bitData) {
    str += "i64";
    elemType = rewriter.getI64Type();
  } else if (enforceInteger) {
    str += "i32";
    elemType = rewriter.getI32Type();
  } else if (elemType == rewriter.getF32Type())
    str += "f32";
  else if (elemType == rewriter.getF16Type()) {
    str += "i32";
    elemType = rewriter.getI32Type();
  } else
    assert(0 && "add more support");
  auto newType = VectorType::get(size, elemType);
  return std::make_pair(str, newType);
}
unsigned encodeDataum(Type type) {
  switch (type.getIntOrFloatBitWidth()) {
  case 8:
    return 1;
  case 16:
    return 2;
  case 32:
    return 3;
  case 64:
    return 4;
  default:
    assert(0 && "add more support");
    return 0;
  }
}

enum class CacheReadHint : uint32_t {
  UNCACHED = 0,
  CACHED = 1,
  STREAMING = 2,
  READ_INVALIDATE = 3,
};

enum class CacheWriteHint : uint32_t {
  UNCACHED = 0,
  WRITE_THROUGH = 1,
  WRITE_BACK = 2,
  STREAMING = 3,
};

template <typename OpType> unsigned encodeCacheHint(OpType op) {
  auto l1hint = op.getCache();
  auto l3hint = 0;
  constexpr bool isWrite = std::is_same_v<OpType, StoreOp>;
  unsigned cacheHint = 1;
  if constexpr (!isWrite) {
    auto l1CacheValue = l1hint == CacheModifier::NONE ? CacheReadHint::UNCACHED
                                                      : CacheReadHint::CACHED;
    auto l3CacheValue = CacheReadHint::UNCACHED;
    // auto l3CacheValue = CacheReadHint::CACHED;
    if (l1CacheValue == CacheReadHint::UNCACHED) {
      if (l3CacheValue == CacheReadHint::UNCACHED)
        cacheHint = 1;
      else if (l3CacheValue == CacheReadHint::CACHED)
        cacheHint = 2;
    } else if (l1CacheValue == CacheReadHint::CACHED) {
      if (l3CacheValue == CacheReadHint::UNCACHED)
        cacheHint = 3;
      else if (l3CacheValue == CacheReadHint::CACHED)
        cacheHint = 4;
    } else if (l1CacheValue == CacheReadHint::STREAMING) {
      if (l3CacheValue == CacheReadHint::UNCACHED)
        cacheHint = 5;
      else if (l3CacheValue == CacheReadHint::CACHED)
        cacheHint = 6;
    } else if (l1CacheValue == CacheReadHint::READ_INVALIDATE) {
      if (l3CacheValue == CacheReadHint::CACHED)
        cacheHint = 7;
    }
  } else {
    auto l1CacheValue = l1hint == CacheModifier::NONE
                            ? CacheWriteHint::UNCACHED
                            : CacheWriteHint::WRITE_BACK;
    // auto l3CacheValue = CacheWriteHint::WRITE_BACK;
    auto l3CacheValue = CacheWriteHint::UNCACHED;
    if (l1CacheValue == CacheWriteHint::UNCACHED) {
      if (l3CacheValue == CacheWriteHint::UNCACHED)
        cacheHint = 1;
      else if (l3CacheValue == CacheWriteHint::WRITE_BACK)
        cacheHint = 2;
    } else if (l1CacheValue == CacheWriteHint::WRITE_THROUGH) {
      if (l3CacheValue == CacheWriteHint::UNCACHED)
        cacheHint = 3;
      else if (l3CacheValue == CacheWriteHint::WRITE_BACK)
        cacheHint = 4;
    } else if (l1CacheValue == CacheWriteHint::STREAMING) {
      if (l3CacheValue == CacheWriteHint::UNCACHED)
        cacheHint = 5;
      else if (l3CacheValue == CacheWriteHint::WRITE_BACK)
        cacheHint = 6;
    } else if (l1CacheValue == CacheWriteHint::WRITE_BACK) {
      if (l3CacheValue == CacheWriteHint::WRITE_BACK)
        cacheHint = 7;
    }
  }
  return cacheHint;
}

void lookupOrInsertIntrinsic(ConversionPatternRewriter &rewriter, Operation *op,
                             std::string name, FunctionType funcType,
                             bool isVC = true) {
  auto funcAttr = StringAttr::get(rewriter.getContext(), name);
  Operation *found = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (!found) {
    OpBuilder::InsertionGuard guard(rewriter);
    auto kernel = op->getParentOfType<spirv::FuncOp>();
    rewriter.setInsertionPoint(kernel);
    auto func = rewriter.create<spirv::FuncOp>(kernel.getLoc(), name, funcType);
    auto linkageTypeAttr =
        rewriter.getAttr<spirv::LinkageTypeAttr>(spirv::LinkageType::Import);
    std::replace(name.begin(), name.end(), '_', '.');
    auto nameAttr = StringAttr::get(rewriter.getContext(), name);
    auto linkage = spirv::LinkageAttributesAttr::get(rewriter.getContext(),
                                                     name, linkageTypeAttr);
    //  nameAttr, linkageTypeAttr);
    func.setLinkageAttributesAttr(linkage);
    if (isVC)
      func->setAttr("VectorComputeFunctionINTEL", rewriter.getUnitAttr());
  }
}

/// @brief
/// assemble the tensor descriptor payload[8xi32] which is of the format
/// -> [base pointer, surface width, surface height, surface pitch,
///     offsetX, offsetY, blockInfo] for 2D tensor desc
/// -> [base pointer, unused] for 1D and scattered tensor desc
/// only base pointer is i64, others are i32
class MakeTensorPtrToVC : public OpConversionPattern<MakeTensorPtrOp> {
public:
  using OpConversionPattern<MakeTensorPtrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();
    // payload
    auto v8i32 = VectorType::get(8, i32Type);
    auto v4i64 = VectorType::get(4, i64Type);
    Value payLoad = rewriter.create<spirv::UndefOp>(loc, v4i64);
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };
    auto base = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64Type,
                                                        adaptor.getBase());
    auto idx0 = createIntConstant(i32Type, 0);
    payLoad =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad, base, idx0);
    payLoad = rewriter.create<spirv::BitcastOp>(loc, v8i32, payLoad);
    auto ptrType = cast<PointerType>(op.getResult().getType());
    auto tType = cast<RankedTensorType>(ptrType.getPointeeType());
    auto rank = tType.getRank();
    if (rank == 2) {
      auto idx2 = createIntConstant(i32Type, 2);
      auto idx3 = createIntConstant(i32Type, 3);
      auto idx4 = createIntConstant(i32Type, 4);
      auto idx5 = createIntConstant(i32Type, 5);
      auto idx6 = createIntConstant(i32Type, 6);
      auto idx7 = createIntConstant(i32Type, 7);
      auto blockWidth = tType.getShape()[1];
      auto blockHeight = tType.getShape()[0];

      /*
      // fixme: support memref for now
      auto memType = cast<MemRefType>(op.getSource().getType());
      unsigned bitWidth = memType.getElementType().getIntOrFloatBitWidth();
      auto surfaceWidth = memType.getShape()[1] * (bitWidth / 8) - 1;
      auto surfaceHeight = memType.getShape()[0] - 1;
      rewriter.create<addIop>;
      auto createOffset = [&](unsigned idx) -> Value {
        Value val;
        if (ShapedType::isDynamic(op.getStaticOffsets()[idx])) {
          val = op.getOffsets()[idx];
          val = rewriter.create<arith::TruncIOp>(loc, i32Type, val);
        } else {
          val = createIntConstant(i32Type, op.getStaticOffsets()[idx]);
        }
        return val;
      };
      */

      auto bytes = createIntConstant(
          i32Type, tType.getElementType().getIntOrFloatBitWidth() / 8);
      auto one = createIntConstant(i32Type, 1);
      Value surfaceW =
          rewriter.create<arith::TruncIOp>(loc, i32Type, op.getShape()[1]);
      surfaceW = rewriter.create<arith::MulIOp>(loc, surfaceW, bytes);
      surfaceW = rewriter.create<arith::SubIOp>(loc, surfaceW, one);
      Value surfaceH =
          rewriter.create<arith::TruncIOp>(loc, i32Type, op.getShape()[0]);
      surfaceH = rewriter.create<arith::SubIOp>(loc, surfaceH, one);
      Value surfaceP =
          rewriter.create<arith::TruncIOp>(loc, i32Type, op.getStrides()[0]);
      surfaceP = rewriter.create<arith::MulIOp>(loc, surfaceP, bytes);
      surfaceP = rewriter.create<arith::SubIOp>(loc, surfaceP, one);
      auto offsetX = op.getOffsets()[1];
      auto offsetY = op.getOffsets()[0];
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceW, idx2);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceH, idx3);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              surfaceP, idx4);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetX, idx5);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              offsetY, idx6);
      unsigned blockVal = ((blockHeight - 1) << 8) | (blockWidth - 1);
      auto blockInfo = createIntConstant(i32Type, blockVal);
      payLoad = rewriter.create<spirv::VectorInsertDynamicOp>(loc, payLoad,
                                                              blockInfo, idx7);
    }
    rewriter.replaceOp(op, payLoad);
    return success();
  }
};

class AdvanceToVC : public OpConversionPattern<AdvanceOp> {
public:
  using OpConversionPattern<AdvanceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i32Type = rewriter.getI32Type();
    auto offsets = adaptor.getOffsets();
    auto desc = adaptor.getPtr();
    for (size_t i = 0; i < offsets.size(); i++) {
      auto offset = offsets[i];
      if (auto cst = dyn_cast<spirv::ConstantOp>(offset.getDefiningOp()))
        if (auto attr = dyn_cast<mlir::IntegerAttr>(cst.getValue());
            attr && attr.getInt() == 0)
          continue;
      auto idx5 = rewriter.create<spirv::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 5));
      auto idx6 = rewriter.create<spirv::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, 6));
      Value idx = i == 0 ? idx6 : idx5;
      auto oldOffset =
          rewriter.create<spirv::VectorExtractDynamicOp>(loc, desc, idx);
      // offset = rewriter.create<arith::TruncIOp>(loc, i32Type, offset);
      auto newOffset =
          rewriter.create<spirv::IAddOp>(loc, i32Type, oldOffset, offset);
      desc = rewriter.create<spirv::VectorInsertDynamicOp>(loc, desc, newOffset,
                                                           idx);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

template <typename OpType>
class LoadStorePrefetchToRawSend : public OpConversionPattern<OpType> {
public:
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = cast<PointerType>(op.getPtr().getType());
    auto tType = cast<RankedTensorType>(ptrType.getPointeeType());
    auto rank = tType.getRank();
    assert(rank <= 2 && "only support 1d/2d load/store/prefetch for now");
    auto loc = op->getLoc();
    constexpr bool isLoad = std::is_same_v<OpType, LoadOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, PrefetchOp>;
    auto createIntConstant = [&](Type type, unsigned value) {
      auto attr = rewriter.getIntegerAttr(type, value);
      return rewriter.create<spirv::ConstantOp>(loc, type, attr);
    };

    /// collect common info
    auto i1Type = rewriter.getI1Type();
    auto i8Type = rewriter.getI8Type();
    auto i32Type = rewriter.getI32Type();
    auto vnni = false;
    auto transpose = false;
    if constexpr (isLoad) {
      vnni = op->hasAttr("DotB") ? true : false;
      // vnni = isDotB ? true : false;
    }
    auto elmType = tType.getElementType();
    VectorType newType = VectorType::get(1, i32Type);
    std::string funcName;
    if constexpr (isPrefetch) {
      funcName = "llvm_genx_raw_send2_noresult_i1_v8i32";
    } else {
      VectorType vecType;
      if constexpr (isLoad) {
        auto converter = this->getTypeConverter();
        auto resultType = converter->convertType(op->getResult(0).getType());
        vecType = cast<VectorType>(resultType);
        funcName = "llvm_genx_raw_send2_";
      } else {
        vecType = cast<VectorType>(adaptor.getValue().getType());
        funcName = "llvm_genx_raw_sends2_noresult_i1_v8i32_";
      }
      std::string typeStr;
      std::tie(typeStr, newType) =
          encodeVectorType(rewriter, vecType, rank == 1);
      funcName += typeStr;
    }
    unsigned cacheHint = encodeCacheHint(op);

    /// fill in parameters for raw.send
    // bit[1:0] EOT,sendc
    auto modifier = createIntConstant(i8Type, 0);
    auto execSize = createIntConstant(i8Type, 0);
    auto pred = createIntConstant(i1Type, 1);
    auto numSrc1 = createIntConstant(i8Type, 1);
    unsigned numDstVal = newType.getNumElements() / 16;
    if (rank == 1) {
      numDstVal *= 2;
    }
    auto numDst = createIntConstant(i8Type, numDstVal);
    // 15 for ugm
    auto sfid = createIntConstant(i8Type, 15);
    auto extMsg = createIntConstant(i32Type, 0);
    // message descriptor
    uint32_t rawSendMsg = 0;
    if (rank == 2) {
      rawSendMsg |= (isLoad || isPrefetch) ? 3 : 7;
      rawSendMsg |= (vnni ? 1 : 0) << 7;
      rawSendMsg |= (encodeDataum(elmType) - 1) << 9;
      rawSendMsg |= (transpose ? 1 : 0) << 15;
      rawSendMsg |= cacheHint << 17;
      rawSendMsg |= (isLoad ? numDstVal : 0) << 20;
      rawSendMsg |= 1 << 25;
    } else {
      // rank == 1
      rawSendMsg |= (isLoad || isPrefetch) ? 0 : 4;
      rawSendMsg |= 3 << 7;
      rawSendMsg |= 3 << 9;
      rawSendMsg |= int(log2(newType.getNumElements()) + 1) << 12;
      rawSendMsg |= 1 << 15;
      rawSendMsg |= cacheHint << 17;
      rawSendMsg |= (isLoad ? 2 * numDstVal : 0) << 20;
      rawSendMsg |= 1 << 25;
    }
    auto msg = createIntConstant(i32Type, rawSendMsg);
    auto payLoad = adaptor.getPtr();
    SmallVector<Value> args{modifier, execSize, pred, numSrc1, numDst,
                            sfid,     extMsg,   msg,  payLoad};
    if constexpr (isLoad) {
      funcName += "_i1_v8i32";
      auto old = rewriter.create<spirv::UndefOp>(loc, newType);
      args.push_back(old);
      auto retType = newType;
      auto funcType =
          rewriter.getFunctionType(ValueRange(args).getTypes(), retType);
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      auto funcOp =
          rewriter.create<spirv::FunctionCallOp>(loc, retType, funcName, args);
      auto converter = this->getTypeConverter();
      auto cast = rewriter.create<spirv::BitcastOp>(
          loc, converter->convertType(op.getType()), funcOp->getResult(0));
      rewriter.replaceOp(op, cast);
    } else {
      if constexpr (isPrefetch)
        args.erase(args.begin() + 4);
      else {
        if (rank == 2) {
          args.push_back(adaptor.getValue());
        } else if (rank == 1) {
          auto cast = rewriter.create<spirv::BitcastOp>(loc, newType,
                                                        adaptor.getValue());
          args.push_back(cast);
        }
      }
      auto funcType = rewriter.getFunctionType(ValueRange(args).getTypes(), {});
      Operation *opPtr = op;
      lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
      rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(), funcName, args);
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class DotToVC : public OpConversionPattern<DotOp> {
public:
  using OpConversionPattern<DotOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto aType =
        convertAndKeepShape(op.getA().getType().cast<RankedTensorType>());
    auto bType =
        convertAndKeepShape(op.getB().getType().cast<RankedTensorType>());
    auto cType =
        convertAndKeepShape(op.getC().getType().cast<RankedTensorType>());
    auto convertType =
        getTypeConverter()->convertType(op->getResult(0).getType());
    auto resultType = cast<VectorType>(convertType);
    auto elementType = aType.getElementType();
    auto numBytes = elementType.getIntOrFloatBitWidth() / 8;
    uint8_t rc = aType.getShape()[0];
    uint8_t sd = aType.getShape()[1] / numBytes;
    // refer to IGC/visa/Common_ISA_util.cpp#87
    auto encodePrecision = [&](Type type) -> uint8_t {
      if (type == rewriter.getBF16Type())
        return 9;
      else if (type == rewriter.getF16Type())
        return 10;
      else if (type == rewriter.getTF32Type())
        return 12;
      else {
        assert(0 && "add more support");
        return 0;
      }
    };
    uint8_t prec1 = encodePrecision(bType.getElementType());
    uint8_t prec2 = encodePrecision(aType.getElementType());
    unsigned infoVal = (rc << 24) | (sd << 16) | (prec2 << 8) | (prec1);
    auto infoAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), infoVal);
    auto info = rewriter.create<spirv::ConstantOp>(loc, rewriter.getI32Type(),
                                                   infoAttr);
    auto newResultType = encodeVectorType(rewriter, resultType).second;
    SmallVector<Value, 4> args{adaptor.getB(), adaptor.getA(), info};
    std::string funcName = "llvm_genx_dpas_nosrc0_";
    // if (op.getAcc())
    {
      funcName = "llvm_genx_dpas2_";
      auto i32Type = rewriter.getI32Type();
      auto createIntConstant = [&](Type type, unsigned value) {
        auto attr = rewriter.getIntegerAttr(type, value);
        return rewriter.create<spirv::ConstantOp>(loc, type, attr);
      };
      auto prec1Arg = createIntConstant(i32Type, prec1);
      auto prec2Arg = createIntConstant(i32Type, prec2);
      auto sdArg = createIntConstant(i32Type, sd);
      auto rcArg = createIntConstant(i32Type, rc);
      auto signless = createIntConstant(i32Type, 0);
      auto dpasAType = encodeVectorType(rewriter, aType).second;
      auto dpasBType = encodeVectorType(rewriter, bType).second;
      auto a =
          rewriter.create<spirv::BitcastOp>(loc, dpasAType, adaptor.getA());
      auto b =
          rewriter.create<spirv::BitcastOp>(loc, dpasBType, adaptor.getB());
      args.assign({adaptor.getC(), b, a, prec1Arg, prec2Arg, sdArg, rcArg,
                   signless, signless});
    }
    funcName += encodeVectorType(rewriter, resultType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, bType).first;
    funcName += "_";
    funcName += encodeVectorType(rewriter, aType).first;
    auto funcType =
        rewriter.getFunctionType(ValueRange(args).getTypes(), newResultType);
    Operation *opPtr = op;
    lookupOrInsertIntrinsic(rewriter, opPtr, funcName, funcType);
    auto funcOp = rewriter.create<spirv::FunctionCallOp>(loc, newResultType,
                                                         funcName, args);
    rewriter.replaceOp(op, funcOp);
    return success();
  }
};

class GlueToVC : public OpConversionPattern<GlueOp> {
public:
  using OpConversionPattern<GlueOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GlueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto operands = adaptor.getOperands();
    // only tensor type be left
    auto dstType =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    auto numElts = dstType.getNumElements();
    SmallVector<int32_t> indices(numElts);
    std::iota(indices.begin(), indices.end(), 0);
    auto attr = rewriter.getI32ArrayAttr(indices);
    auto num = operands.size();
    if (num == 1) {
      rewriter.replaceOp(op, operands[0]);
    } else if (num == 2) {
      rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(
          op, dstType, operands[0], operands[1], attr);
    } else if (num == 4) {
      auto subType = VectorType::get(numElts / 2, dstType.getElementType());
      indices.pop_back_n(numElts / 2);
      auto attr01 = rewriter.getI32ArrayAttr(indices);
      auto shfl01 = rewriter.create<spirv::VectorShuffleOp>(
          loc, subType, operands[0], operands[1], attr01);
      auto attr23 = rewriter.getI32ArrayAttr(indices);
      auto shfl23 = rewriter.create<spirv::VectorShuffleOp>(
          loc, subType, operands[2], operands[3], attr23);
      auto shfl = rewriter.create<spirv::VectorShuffleOp>(loc, dstType, shfl01,
                                                          shfl23, attr);
      rewriter.replaceOp(op, shfl);
    } else {
      assert(0 && "add more support for tt.glue to spirv");
    }
    return success();
  }
};

class ExtractToVC : public OpConversionPattern<ExtractOp> {
public:
  using OpConversionPattern<ExtractOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto base = adaptor.getBase();
    auto idx = op.getIdx();
    // only tensor type be left
    auto dstType =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    auto numElts = dstType.getNumElements();
    SmallVector<int32_t> indices(numElts);
    auto start = idx * numElts;
    std::iota(indices.begin(), indices.end(), start);
    auto attr = rewriter.getI32ArrayAttr(indices);
    rewriter.replaceOpWithNewOp<spirv::VectorShuffleOp>(op, dstType, base, base,
                                                        attr);
    return success();
  }
};

struct ReturnOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::ReturnOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::ReturnOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(op, TypeRange(), ValueRange(),
                                                 op->getAttrs());
    return success();
  }
};

struct GetProgramIdOpToSPIRVConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);

    Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(
        loc, rewriter.getIndexType(), dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, i32_ty, blockId);
    // Value blockId_idx =
    //     rewriter.create<::mlir::arith::TruncIOp>(loc, i32_ty, blockId);
    // auto *typeConverter = this->template
    // getTypeConverter<SPIRVTypeConverter>(); auto indexType =
    // typeConverter->getIndexType();

    // rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
    //     op, TypeRange{i32_ty}, ValueRange{blockId_idx});
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

/// Converts the given `srcAttr` to a new attribute of the given `dstType`.
/// Returns null attribute if conversion fails.
static IntegerAttr convertIntegerAttr(IntegerAttr srcAttr, IntegerType dstType,
                                      Builder builder) {
  // If the source number uses less active bits than the target bitwidth, then
  // it should be safe to convert.
  if (srcAttr.getValue().isIntN(dstType.getWidth()))
    return builder.getIntegerAttr(dstType, srcAttr.getInt());

  // XXX: Try again by interpreting the source number as a signed value.
  // Although integers in the standard dialect are signless, they can
  // represent a signed number. It's the operation decides how to interpret.
  // This is dangerous, but it seems there is no good way of handling this if
  // we still want to change the bitwidth. Emit a message at least.
  if (srcAttr.getValue().isSignedIntN(dstType.getWidth())) {
    auto dstAttr = builder.getIntegerAttr(dstType, srcAttr.getInt());
    // LLVM_DEBUG(llvm::dbgs() << "attribute '" << srcAttr << "' converted to '"
    //                         << dstAttr << "' for type '" << dstType <<
    //                         "'\n");
    return dstAttr;
  }

  //   LLVM_DEBUG(llvm::dbgs() << "attribute '" << srcAttr
  //                           << "' illegal: cannot fit into target type '"
  //                           << dstType << "'\n");
  return {};
}

/// Converts the given `srcAttr` to a new attribute of the given `dstType`.
/// Returns null attribute if `dstType` is not 32-bit or conversion fails.
static FloatAttr convertFloatAttr(FloatAttr srcAttr, FloatType dstType,
                                  Builder builder) {
  // Only support converting to float for now.
  if (!dstType.isF32())
    return FloatAttr();

  // Try to convert the source floating-point number to single precision.
  APFloat dstVal = srcAttr.getValue();
  bool losesInfo = false;
  APFloat::opStatus status =
      dstVal.convert(APFloat::IEEEsingle(), APFloat::rmTowardZero, &losesInfo);
  if (status != APFloat::opOK || losesInfo) {
    ;
    // LLVM_DEBUG(llvm::dbgs()
    //            << srcAttr << " illegal: cannot fit into converted type '"
    //            << dstType << "'\n");
    return FloatAttr();
  }

  return builder.getF32FloatAttr(dstVal.convertToFloat());
}

struct ConstantCompositeOpPattern final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "constant \n";
    constOp->dump();
    auto srcType = dyn_cast<ShapedType>(constOp.getType());
    if (!srcType || srcType.getNumElements() == 1)
      return failure();

    // arith.constant should only have vector or tenor types.
    assert((isa<VectorType, RankedTensorType>(srcType)));

    Type dstType = getTypeConverter()->convertType(srcType);
    if (!dstType)
      return failure();

    auto dstElementsAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!dstElementsAttr)
      return failure();

    ShapedType dstAttrType = dstElementsAttr.getType();
    auto vecType = cast<VectorType>(dstType);
    dstAttrType =
        VectorType::get(vecType.getNumElements(), vecType.getElementType());
    dstElementsAttr = dstElementsAttr.reshape(dstAttrType);

    // // If the composite type has more than one dimensions, perform
    // // linearization.
    // if (srcType.getRank() > 1) {
    //   if (isa<RankedTensorType>(srcType)) {
    //     dstAttrType = RankedTensorType::get(srcType.getNumElements(),
    //                                         srcType.getElementType());
    //     dstElementsAttr = dstElementsAttr.reshape(dstAttrType);
    //   } else {
    //   // TODO: add support for large vectors.
    //   return failure();
    //   }
    // }

    Type srcElemType = srcType.getElementType();
    Type dstElemType;
    // Tensor types are converted to SPIR-V array types; vector types are
    // converted to SPIR-V vector/array types.
    if (auto arrayType = dyn_cast<spirv::ArrayType>(dstType))
      dstElemType = arrayType.getElementType();
    else
      dstElemType = cast<VectorType>(dstType).getElementType();

    // If the source and destination element types are different, perform
    // attribute conversion.
    if (srcElemType != dstElemType) {
      SmallVector<Attribute, 8> elements;
      if (isa<FloatType>(srcElemType)) {
        for (FloatAttr srcAttr : dstElementsAttr.getValues<FloatAttr>()) {
          FloatAttr dstAttr =
              convertFloatAttr(srcAttr, cast<FloatType>(dstElemType), rewriter);
          if (!dstAttr)
            return failure();
          elements.push_back(dstAttr);
        }
      } else if (srcElemType.isInteger(1)) {
        return failure();
      } else {
        for (IntegerAttr srcAttr : dstElementsAttr.getValues<IntegerAttr>()) {
          IntegerAttr dstAttr = convertIntegerAttr(
              srcAttr, cast<IntegerType>(dstElemType), rewriter);
          if (!dstAttr)
            return failure();
          elements.push_back(dstAttr);
        }
      }

      // Unfortunately, we cannot use dialect-specific types for element
      // attributes; element attributes only works with builtin types. So we
      // need to prepare another converted builtin types for the destination
      // elements attribute.
      if (isa<RankedTensorType>(dstAttrType))
        dstAttrType =
            RankedTensorType::get(dstAttrType.getShape(), dstElemType);
      else
        dstAttrType = VectorType::get(dstAttrType.getShape(), dstElemType);

      dstElementsAttr = DenseElementsAttr::get(dstAttrType, elements);
    }

    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType,
                                                   dstElementsAttr);
    llvm::outs() << "constant \n";
    return success();
  }
};

} // namespace

void populateTritonGPUToVCPatterns(TritonGPUToSPIRVTypeConverter &typeConverter,
                                   MLIRContext *context,
                                   RewritePatternSet &patterns, int numWarps,
                                   PatternBenefit benefit) {
  patterns.add<MakeTensorPtrToVC>(typeConverter, context, benefit);
  patterns.add<AdvanceToVC>(typeConverter, context, benefit);
  patterns.add<LoadStorePrefetchToRawSend<PrefetchOp>>(typeConverter, context,
                                                       benefit);
  patterns.add<LoadStorePrefetchToRawSend<LoadOp>>(typeConverter, context,
                                                   benefit);
  patterns.add<LoadStorePrefetchToRawSend<StoreOp>>(typeConverter, context,
                                                    benefit);
  patterns.add<DotToVC>(typeConverter, context, benefit);
  patterns.add<GetProgramIdOpToSPIRVConversion>(typeConverter, context,
                                                benefit);
  patterns.add<ReturnOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<ConstantCompositeOpPattern>(typeConverter, context, benefit);
  patterns.add<GlueToVC>(typeConverter, context, benefit);
  patterns.add<ExtractToVC>(typeConverter, context, benefit);
}
