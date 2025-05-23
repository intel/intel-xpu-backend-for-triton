#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

namespace {
struct CachingBuilder : TritonLLVMOpBuilder {
  CachingBuilder(Location loc, OpBuilder &builder)
      : TritonLLVMOpBuilder(loc, builder) {}

  Value dense_val(const ShapedType &type, ArrayRef<Attribute> values) {
    auto attr = DenseElementsAttr::get(type, values);
    return getOrCreateConstant(type, attr);
  }

  Value i8_val(int64_t val) { return int_val(8, val); }
  Value i32_val(int64_t val) { return int_val(32, val); }

  Value int_val(short bitwidth, int64_t val) {
    Type ty = builder->getIntegerType(bitwidth);
    Attribute attr = builder->getIntegerAttr(ty, val);
    return getOrCreateConstant(ty, attr);
  }

private:
  DenseMap<std::pair<Type, Attribute>, Value> constCache;

  Value getOrCreateConstant(Type type, Attribute attr) {
    auto key = std::make_pair(type, attr);
    auto it = constCache.find(key);
    if (it != constCache.end())
      return it->second;
    auto cst = builder->create<LLVM::ConstantOp>(loc, type, attr);
    constCache[key] = cst;
    return cst;
  }
};

class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> results;

    {
      CachingBuilder b(loc, rewriter);
      // Create a constant vector containing all the possible values
      Value table;
      {
        auto elemTy = dyn_cast<FloatType>(op.getType().getElementType());
        assert(elemTy == f16_ty || elemTy == bf16_ty);
        SmallVector<Attribute, 16> values;
        for (double v : {0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5,
                         -2., -3., -4., -6.})
          values.push_back(b.builder->getFloatAttr(elemTy, v));
        table = b.dense_val(VectorType::get({16}, elemTy), values);
      }

      SmallVector<Value> values;
      Value src = adaptor.getSrc();
      auto i8Ty = b.builder->getI8Type();
      collectValues(b, src, values);

      for (auto value : values) {
        if (auto vecTy = dyn_cast_or_null<VectorType>(value.getType());
            !vecTy) {
          assert(value.getType() == i8Ty);
          Value idx1 = b.and_(value, b.i8_val(15));
          Value idx2 = b.lshr(value, b.i8_val(4));
          results.push_back(b.extract_element(table, idx1));
          results.push_back(b.extract_element(table, idx2));
        } else if (vecTy.getElementType() == b.builder->getI32Type()) {
          ShapedType i8VecTy = VectorType::get(4, i8Ty);
          auto andVect =
              b.dense_val(vecTy, b.builder->getI32IntegerAttr(0x0F0F0F0F));
          auto shVec = b.dense_val(vecTy, b.builder->getI32IntegerAttr(4));
          Value i32IdxVec1 = b.and_(value, andVect);
          Value i32IdxVec2 = b.and_(b.lshr(value, shVec), andVect);
          // Extract each value from i32 vectors and cast to i8 vectors
          for (int32_t i = 0, n = vecTy.getNumElements(); i < n; ++i) {
            auto idx = b.i32_val(i);
            Value i1 = b.extract_element(i32IdxVec1, idx);
            Value i2 = b.extract_element(i32IdxVec2, idx);
            Value idxVec1 = b.bitcast(i1, i8VecTy);
            Value idxVec2 = b.bitcast(i2, i8VecTy);
            extractFloats(b, idxVec1, idxVec2, vecTy.getNumElements(), results,
                          table);
          }
        } else {
          assert(vecTy.getElementType() == i8Ty);
          auto andVect = b.dense_val(vecTy, b.builder->getI8IntegerAttr(0x0F));
          auto shVec = b.dense_val(vecTy, b.builder->getI8IntegerAttr(4));
          Value idxVec1 = b.and_(value, andVect);
          Value idxVec2 = b.lshr(value, shVec);
          extractFloats(b, idxVec1, idxVec2, vecTy.getNumElements(), results,
                        table);
        }
      }
    }

    rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), results,
                                          rewriter, op.getType()));
    return success();
  }

private:
  static void collectValues(CachingBuilder &b, Value src,
                            SmallVector<Value> &values) {
    auto structTy = dyn_cast_or_null<LLVM::LLVMStructType>(src.getType());
    if (!structTy) {
      values.emplace_back(src);
      return;
    }

    // If the entire struct consists of subsequent insertvalue:
    // %str = llvm.mlir.undef : !llvm.struct<(i8, i8)>
    // %str0 = llvm.insertvalue %v0, %str[0] : !llvm.struct<(i8, i8)>
    // %str1 = llvm.insertvalue %v1, %str0[1] : !llvm.struct<(i8, i8)>
    // use the inserted values (%v0 and %v1) instead of adding the
    // extractvalue operations.
    auto i8Ty = b.builder->getI8Type();
    auto remaining = structTy.getBody().size();
    values.resize(remaining);
    for (auto ins = src.getDefiningOp<LLVM::InsertValueOp>();
         ins && ins.getPosition()[0] == remaining - 1;
         ins = ins.getContainer().getDefiningOp<LLVM::InsertValueOp>()) {
      values[--remaining] = ins.getValue();
      assert(values[remaining].getType() == i8Ty);
    }

    // Add the remaining values, if any.
    if (remaining) {
      for (auto [i, type] : llvm::enumerate(
               llvm::make_range(structTy.getBody().begin(),
                                structTy.getBody().begin() + remaining))) {
        assert(type == i8Ty);
        values[i] = b.extract_val(type, src, i);
      }
    }

    // Detect subsequent extractions of all vallues from an i8 vector:
    // %c0 = llvm.mlir.constant(0 : i32) : i32
    // %e0 = llvm.extractelement %i8vec[%c0 : i32] : vector<2xi8>
    // %c1 = llvm.mlir.constant(1 : i32) : i32
    // %e1 = llvm.extractelement %i8vec[%c1 : i32] : vector<2xi8>
    // If values[i] == e0 and values[i + 1] == e1, replace them with i8vec.
    if (replaceVectorExtracts(b, src, values)) {
      // Detect subsequent extractions from i32 vector and casts to i8 vector:
      // %c0 = llvm.mlir.constant(0 : i32) : i32
      // %i0 = llvm.extractelement %i32vec[%c0 : i32] : vector<2xi32>
      // %i8vec0 = llvm.bitcast %i0 : i32 to vector<4xi8>
      // %c1 = llvm.mlir.constant(1 : i32) : i32
      // %i1 = llvm.extractelement %i32vec[%c1 : i32] : vector<2xi32>
      // %i8vec1 = llvm.bitcast %i1 : i32 to vector<4xi8>
      // If values[i] == i8vec0 and values[i + 1] == i8vec1, replace them with
      // i32vec.
      replaceVectorCastExtracts(b, src, values);
    }
  }

  // Extract floats from the lookup table by indices.
  static void extractFloats(CachingBuilder &b, Value idxVec1, Value idxVec2,
                            int32_t size, SmallVector<Value> &results,
                            Value table) {
    auto off = results.size();
    results.resize(off + size * 2);
    for (int32_t i = 0; i < size; ++i) {
      Value idx = b.extract_element(idxVec1, b.i32_val(i));
      results[off + 2 * i] = b.extract_element(table, idx);
    }
    for (int32_t i = 0; i < size; ++i) {
      Value idx = b.extract_element(idxVec2, b.i32_val(i));
      results[off + 2 * i + 1] = b.extract_element(table, idx);
    }
  }

  // Detect the subsequent extractions of all vallues from an i8 vector and
  // replace them with the vector.
  static bool replaceVectorExtracts(CachingBuilder &b, Value src,
                                    SmallVector<Value> &values) {
    bool replaced = false;
    for (unsigned i = 0; i < values.size(); i++) {
      // If the value is an extractelement from a vector and the index is 0
      // and the subsequent values are the extraction of all values from the
      // same vector, replace all these values with the original vector.
      if (auto vec = isVectorExtract(values[i], 0);
          vec && replaceValues(
                     values, i,
                     dyn_cast<VectorType>(vec.getType()).getNumElements() - 1,
                     vec, isVectorExtract)) {
        assert(dyn_cast<VectorType>(vec.getType()).getElementType() ==
               b.builder->getI8Type());
        replaced = true;
      }
    }
    return replaced;
  }

  // If a value is a bitcast of a value extracted from an i32 vector and the
  // subsequent values are also extracts from the same vector, replace all
  // them with the original vector.
  static void replaceVectorCastExtracts(CachingBuilder &b, Value src,
                                        SmallVector<Value> &values) {
    auto isCastExtract = [i8Ty = b.builder->getI8Type(),
                          i32Ty = b.builder->getI32Type()](Value &value,
                                                           unsigned pos) {
      if (auto bitcast = value.getDefiningOp<LLVM::BitcastOp>()) {
        if (auto vecTy = dyn_cast_or_null<VectorType>(bitcast.getType());
            vecTy && vecTy.getElementType() == i8Ty &&
            vecTy.getNumElements() == 4) {
          auto operand = bitcast.getOperand();
          if (auto vec = isVectorExtract(operand, pos)) {
            if (auto elType =
                    dyn_cast<VectorType>(vec.getType()).getElementType();
                elType == i32Ty) {
              return vec;
            }
          }
        }
      }
      return Value();
    };

    for (unsigned i = 0; i < values.size(); i++) {
      if (auto vec = isCastExtract(values[i], 0)) {
        replaceValues(values, i,
                      dyn_cast<VectorType>(vec.getType()).getNumElements() - 1,
                      vec, isCastExtract);
      }
    }
  }

  // Check if the value is an extraction from a vector at the specified
  // position and the vector size is > 1, return the vector.
  static Value isVectorExtract(Value value, unsigned pos) {
    if (auto extract = value.getDefiningOp<LLVM::ExtractElementOp>()) {
      auto operand = extract.getOperand(0);
      if (auto vecTy = dyn_cast_or_null<VectorType>(operand.getType());
          vecTy && vecTy.getNumElements() > 1) {
        if (auto idx =
                extract.getPosition().getDefiningOp<LLVM::ConstantOp>()) {
          if (auto attr = dyn_cast_or_null<IntegerAttr>(idx.getValue());
              attr && attr.getInt() == pos) {
            return operand;
          }
        }
      }
    }
    return Value();
  }

  // Replace the value at the `position` with the `replacement` and erase the
  // `count` values after it, if the `map` function returns the `replacement`
  // for all these values.
  static bool
  replaceValues(SmallVector<Value> &values, unsigned position, unsigned count,
                Value replacement,
                const std::function<Value(Value &, unsigned)> &map) {
    if (position + count + 1 > values.size())
      return false;
    for (auto [i, v] : llvm::enumerate(
             llvm::make_range(values.begin() + position + 1,
                              values.begin() + position + 1 + count))) {
      if (map(v, i + 1) != replacement)
        return false;
    }
    values[position] = replacement;
    values.erase(values.begin() + position + 1,
                 values.begin() + position + 1 + count);
    return true;
  }
};
} // anonymous namespace

void mlir::triton::intel::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
