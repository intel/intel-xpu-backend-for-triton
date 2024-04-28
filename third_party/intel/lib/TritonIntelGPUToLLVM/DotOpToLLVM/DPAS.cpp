#include "../TritonGPUToLLVMBase.h"
#include "../Utility.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <thread>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu::intel;
using mlir::triton::gpu::DotOperandEncodingAttr;
using mlir::triton::gpu::intel::DpasEncodingAttr;

namespace {

class DotOpDPASConversionHelper {
public:
  using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

  DotOpDPASConversionHelper(DpasEncodingAttr dpasLayout,
                            ConversionPatternRewriter &rewriter,
                            TritonIntelGPUToLLVMTypeConverter *typeConverter,
                            Location loc)
      : dpasLayout(dpasLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(dpasLayout.getContext()) {}

  std::tuple<Type, Type, Type, Type> static getDPASOperandsType(
      DPASEngineType dpasType, MLIRContext *ctx, DpasEncodingAttr layout) {
    Type fp32Ty = type::f32Ty(ctx);
    Type fp16Ty = type::f16Ty(ctx);
    Type bf16Ty = type::bf16Ty(ctx);
    Type i32Ty = type::i32Ty(ctx);
    Type i16Ty = type::i16Ty(ctx);
    Type s32Ty = IntegerType::get(ctx, 32, IntegerType::Signed);

    unsigned threadsPerWarp = layout.getSubGroupSize();
    unsigned opsPerChannel = layout.getOpsPerChannel();
    SmallVector<unsigned> shapeC = layout.getShapeC();
    unsigned elemNumC = product<unsigned>(shapeC) / threadsPerWarp;
    SmallVector<unsigned> shapeA = layout.getShapeA();
    unsigned elemNumA = product<unsigned>(shapeA) / threadsPerWarp;
    SmallVector<unsigned> shapeB = layout.getShapeB();
    unsigned elemNumB = product<unsigned>(shapeB) / threadsPerWarp;
    switch (dpasType) {
    case DPASEngineType::FP32_FP32_FP16_FP16: {
      Type cTy = vec_ty(fp32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case DPASEngineType::FP16_FP16_FP16_FP16: {
      Type cTy = vec_ty(fp16Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case DPASEngineType::FP32_FP32_BF16_BF16: {
      Type cTy = vec_ty(fp32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case DPASEngineType::BF16_BF16_BF16_BF16: {
      Type cTy = vec_ty(bf16Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case DPASEngineType::FP32_FP32_TF32_TF32: {
      Type cTy = vec_ty(fp32Ty, elemNumC);
      Type aTy = vec_ty(i32Ty, elemNumA);                 // pack scalar to i32.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case DPASEngineType::U32_U32_U8_U8: {
      Type cTy = vec_ty(i32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack 2 i8 to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case DPASEngineType::S32_S32_S8_S8: {
      Type cTy = vec_ty(s32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack 2 i8 to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    default:
      llvm::report_fatal_error("Unsupported dpas type found");
    }

    return std::make_tuple<Type, Type, Type, Type>({}, {}, {}, {});
  }

  /// Generate the GEN dialect dpas operation. Rules (for PVC):
  ///  - SD = 8
  ///  - M = RC = 1,2,4,8 (we use 8)
  ///  - N = exec_size = SIMD_width = 16
  ///  - Size of A, B element type = {32,16,8}, for {tf32,bf16/f16,u8/i8}
  ///  - K=SD * num_packed_elems_in_Dword = {8,16,32}, for {tf32,bf16/f16,u8/i8}
  ///
  /// The per-lane intrinsic function generated is defined to perform the
  /// following operation:
  ///    D[0:M-1,laneId] = C[0:M-1,laneId] + A[0:M-1,laneId] * B[0:K-1,laneId]
  ///
  /// Example: A and B elements are f16, K=SD*2=16, SIMD_width=16, each
  /// lane gets K/SIMD_width=1 column of A
  ///
  ///    D[0:M-1,lane_id] =
  ///      gen.matrix.dpas C, A, B {pa, pb, rc=M, sd=8}
  ///        : (vector<8xf32>, vector<4xi32>, vector<4xi32>) -> vector<8xf32>
  ///
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    Value A = op.getA(), B = op.getB(), C = op.getC(), D = op.getResult();
    Value loadedA = adaptor.getA(), loadedB = adaptor.getB(),
          loadedC = adaptor.getC();

    auto ATensorTy = A.getType().cast<RankedTensorType>(),
         BTensorTy = B.getType().cast<RankedTensorType>(),
         CTensorTy = C.getType().cast<RankedTensorType>(),
         DTensorTy = D.getType().cast<RankedTensorType>();

    auto AEncoding = ATensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto BEncoding = BTensorTy.getEncoding().cast<DotOperandEncodingAttr>();

    auto ADpasEncoding =
        AEncoding.getParent().cast<triton::gpu::intel::DpasEncodingAttr>();
    auto BDpasEncoding =
        BEncoding.getParent().cast<triton::gpu::intel::DpasEncodingAttr>();

    auto repA = ADpasEncoding.getDPASRepetitions(ATensorTy.getShape(),
                                                 AEncoding.getOpIdx());
    auto repB = BDpasEncoding.getDPASRepetitions(BTensorTy.getShape(),
                                                 BEncoding.getOpIdx());
    assert(repA[1] == repB[0] && "Unexpected rep for A and B operands");

    unsigned repM = repA[0], repN = repB[1], repK = repA[1];

    auto dpasType = getDPASType(op);
    auto dpasEncoding = DTensorTy.getEncoding().cast<DpasEncodingAttr>();
    Type aTy, bTy, cTy, dTy;
    std::tie(dTy, cTy, aTy, bTy) =
        getDPASOperandsType(dpasType, op.getContext(), dpasEncoding);
    ValueTable ha = getValuesFromDotOperandLayoutStruct(
        loadedA, repM, repK,
        typeConverter->convertType(ATensorTy.getElementType()), aTy);
    ValueTable hb = getValuesFromDotOperandLayoutStruct(
        loadedB, repN, repK,
        typeConverter->convertType(BTensorTy.getElementType()), bTy);
    ValueTable fc = getValuesFromDotOperandLayoutStruct(
        loadedC, repM, repN,
        typeConverter->convertType(CTensorTy.getElementType()), cTy);

    Type resElemTy = DTensorTy.getElementType();

    TritonGEN::PrecisionType APrecision =
                                 getElementPrecision(ATensorTy, resElemTy),
                             BPrecision =
                                 getElementPrecision(BTensorTy, resElemTy);

    assert(APrecision == BPrecision &&
           "A and B precision enumerators do not match");

    LLVM_DEBUG({
      llvm::dbgs() << "repM = " << repM << "\n";
      llvm::dbgs() << "repK = " << repK << "\n";
      llvm::dbgs() << "repN = " << repN << "\n";
      llvm::dbgs() << "fc.size()= " << fc.size() << "\n";
    });

    auto generateDPASOp = [&](unsigned m, unsigned n, unsigned k) {
      auto valA = ha.at({m, k});
      auto valB = hb.at({n, k});
      auto valc = fc.at({m, n});

      auto pA = TritonGEN::PrecisionTypeAttr::get(A.getContext(), APrecision);
      auto pB = TritonGEN::PrecisionTypeAttr::get(B.getContext(), BPrecision);
      auto RC = IntegerAttr::get(rewriter.getIntegerType(32),
                                 dpasEncoding.getRepeatCount());

      auto ret = rewriter.create<TritonGEN::MatrixDPASOp>(loc, dTy, valc, valA,
                                                          valB, pA, pB, RC);

      fc.at({m, n}) = ret;
    };

    for (int k = 0; k < repK; ++k)
      for (int m = 0; m < repM; ++m)
        for (int n = 0; n < repN; ++n)
          generateDPASOp(m, n, k);

    Value res =
        composeValuesToDotOperandLayoutStruct(fc, repM, repN, resElemTy);

    rewriter.replaceOp(op, res);

    return success();
  }

private:
  /// Return the bit width corresponding to the given precision or std::nullopt
  /// if it cannot be computed.
  std::optional<unsigned> getBitWidth(TritonGEN::PrecisionType PT) const {
    switch (PT) {
    case TritonGEN::PrecisionType::U2:
    case TritonGEN::PrecisionType::S2:
      return 2;
    case TritonGEN::PrecisionType::U4:
    case TritonGEN::PrecisionType::S4:
      return 4;
    case TritonGEN::PrecisionType::U8:
    case TritonGEN::PrecisionType::S8:
      return 8;
    case TritonGEN::PrecisionType::BF16:
    case TritonGEN::PrecisionType::FP16:
      return 16;
    case TritonGEN::PrecisionType::TF32:
      return 32;
    default:
      return std::nullopt;
    }
  }

  Value composeValuesToDotOperandLayoutStruct(const ValueTable &vals,
                                              int64_t dim0, int64_t dim1,
                                              Type elemTy) const {

    std::vector<Value> elems;
    for (int m = 0; m < dim0; ++m)
      for (int k = 0; k < dim1; ++k) {
        auto matVal = vals.at({m, k});
        auto vecType = matVal.getType().cast<mlir::VectorType>();
        auto valTy = vecType.getElementType();
        for (int i = 0; i < vecType.getNumElements(); ++i) {
          auto val = extract_element(valTy, matVal, i32_val(i));

          elems.push_back(val);
        }
      }

    assert(!elems.empty() &&
           "unexpected empty result in composing the DPAS result.");

    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(elems.size(), elemTy));
    return packLLElements(loc, typeConverter, elems, rewriter, structTy);
  }

  ValueTable getValuesFromDotOperandLayoutStruct(Value val, int64_t dim0,
                                                 int64_t dim1, Type elemTy,
                                                 Type dotOperandType) const {
    SmallVector<Value> elems = unpackLLElements(loc, val, rewriter);

    int offset{};
    ValueTable vals;
    size_t totalElems = elems.size();
    size_t numElemsPerOperand = totalElems / (dim0 * dim1);
    VectorType dotOpTy = vec_ty(elemTy, numElemsPerOperand);

    for (int i = 0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        Value matVal = rewriter.create<LLVM::UndefOp>(loc, dotOpTy);
        for (int k = 0; k < numElemsPerOperand; ++k) {
          matVal = insert_element(dotOpTy, matVal, elems[offset++], i32_val(k));
        }
        vals[{i, j}] = bitcast(matVal, dotOperandType);
      }
    }
    return vals;
  }

  /// Return the precision for the given tensor type (the type of A or B) and
  /// result element type.
  TritonGEN::PrecisionType getElementPrecision(RankedTensorType tensorTy,
                                               Type resElemType) const {
    assert((isa<IntegerType>(resElemType) || isa<FloatType>(resElemType)) &&
           "Expecting an integer or floating point type");

    Type elemType = tensorTy.getElementType();
    unsigned width = elemType.getIntOrFloatBitWidth();
    assert(width <= 32 && "Unexpected width");

    if (isa<FloatType>(resElemType)) {
      if (width == 32)
        return TritonGEN::PrecisionType::TF32;
      if (isa<BFloat16Type>(elemType))
        return TritonGEN::PrecisionType::BF16;
      if (isa<Float16Type>(elemType))
        return TritonGEN::PrecisionType::FP16;
    } else if (width == 8) {
      return elemType.isUnsignedInteger() ? TritonGEN::PrecisionType::U8
                                          : TritonGEN::PrecisionType::S8;
    }

    return TritonGEN::PrecisionType::UNUSED;
  }

  DpasEncodingAttr dpasLayout;
  ConversionPatternRewriter &rewriter;
  TritonIntelGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx;
};

} // namespace

namespace fma_details {
LogicalResult convertDPAS(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonIntelGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  LLVM_DEBUG({
    auto module = op->getParentOfType<ModuleOp>();
    llvm::dbgs() << "module before DPAS generation\n";
    module->dump();
  });

  Value A = op.getA(), B = op.getB(), C = op.getC(), D = op.getResult();

  auto ATensorTy = A.getType().cast<RankedTensorType>(),
       BTensorTy = B.getType().cast<RankedTensorType>(),
       CTensorTy = C.getType().cast<RankedTensorType>(),
       DTensorTy = D.getType().cast<RankedTensorType>();

  assert(ATensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
         BTensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");
  assert(CTensorTy.getEncoding().isa<DpasEncodingAttr>() &&
         DTensorTy.getEncoding().isa<DpasEncodingAttr>() &&
         "Currently, we only support $c and $d with a dpas layout.");
  assert(CTensorTy.getShape()[0] == DTensorTy.getShape()[0] &&
         CTensorTy.getShape()[1] == DTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto dpasLayout = op.getResult()
                        .getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<DpasEncodingAttr>();

  DotOpDPASConversionHelper helper(dpasLayout, rewriter, typeConverter,
                                   op.getLoc());

  return helper.convertDot(op, adaptor);
}
} // namespace fma_details
