#include "../TritonGPUToLLVMBase.h"
#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu::intel;

namespace {

class DotOpDPASConversionHelper {
public:
  using ValueTable = std::map<std::array<unsigned, 3>, Value>;

  DotOpDPASConversionHelper(DpasEncodingAttr dpasLayout,
                            ConversionPatternRewriter &rewriter,
                            TritonIntelGPUToLLVMTypeConverter *typeConverter,
                            Location loc)
      : dpasLayout(dpasLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(dpasLayout.getContext()) {}

  std::tuple<Type, Type, Type, Type> static getDPASOperandsType(
      DPASAnalysis::DPASEngineType dpasType, MLIRContext *ctx,
      DpasEncodingAttr layout) {
    Type fp32Ty = type::f32Ty(ctx);
    Type fp16Ty = type::f16Ty(ctx);
    Type bf16Ty = type::bf16Ty(ctx);
    Type i32Ty = type::i32Ty(ctx);
    Type i16Ty = type::i16Ty(ctx);
    Type s32Ty = IntegerType::get(ctx, 32, IntegerType::Signed);

    unsigned threadsPerWarp = layout.getThreadsPerWarp();
    unsigned opsPerChannel = layout.getOpsPerChannel();
    SmallVector<unsigned> shapeC = layout.getDPASInstShapeC();
    unsigned elemNumC = product<unsigned>(shapeC) / threadsPerWarp;
    SmallVector<unsigned> shapeA = layout.getDPASInstShapeA();
    unsigned elemNumA = product<unsigned>(shapeA) / threadsPerWarp;
    SmallVector<unsigned> shapeB = layout.getDPASInstShapeB();
    unsigned elemNumB = product<unsigned>(shapeB) / threadsPerWarp;

    using DPASEngineType = DPASAnalysis::DPASEngineType;
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
    case DPASEngineType::BF16_BF16_FP8_FP8: {
      Type cTy = vec_ty(bf16Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case DPASEngineType::FP32_FP32_FP8_FP8: {
      Type cTy = vec_ty(fp32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack scalar to i16.
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

    auto ATensorTy = cast<RankedTensorType>(A.getType()),
         BTensorTy = cast<RankedTensorType>(B.getType()),
         CTensorTy = cast<RankedTensorType>(C.getType()),
         DTensorTy = cast<RankedTensorType>(D.getType());

    auto AEncoding = cast<DotOperandEncodingAttr>(ATensorTy.getEncoding());
    auto BEncoding = cast<DotOperandEncodingAttr>(BTensorTy.getEncoding());

    auto ADpasEncoding =
        cast<triton::gpu::intel::DpasEncodingAttr>(AEncoding.getParent());
    auto BDpasEncoding =
        cast<triton::gpu::intel::DpasEncodingAttr>(BEncoding.getParent());

    SmallVector<int64_t> repA = ADpasEncoding.getDPASRepetitions(
        ATensorTy.getShape(), AEncoding.getOpIdx());
    SmallVector<int64_t> repB = BDpasEncoding.getDPASRepetitions(
        BTensorTy.getShape(), BEncoding.getOpIdx());
    assert(repA[0] == repB[0] && "A and B should have the same batch size");
    assert(repA[2] == repB[1] && "Unexpected rep for A and B operands");
    unsigned repBatch = repA[0];
    unsigned repM = repA[1], repN = repB[2], repK = repA[2];

    auto dpasType = DPASAnalysis::getDPASType(op);
    auto dpasEncoding = cast<DpasEncodingAttr>(DTensorTy.getEncoding());
    Type aTy, bTy, cTy, dTy;
    std::tie(dTy, cTy, aTy, bTy) =
        getDPASOperandsType(dpasType, op.getContext(), dpasEncoding);
    ValueTable ha = getValuesFromDotOperandLayoutStruct(
        loadedA, repBatch, repM, repK,
        typeConverter->convertType(ATensorTy.getElementType()),
        DpasEncodingAttr::OpIdx::OperandA);
    ValueTable hb = getValuesFromDotOperandLayoutStruct(
        loadedB, repBatch, repN, repK,
        typeConverter->convertType(BTensorTy.getElementType()),
        DpasEncodingAttr::OpIdx::OperandB);
    ValueTable fc = getValuesFromDotOperandLayoutStruct(
        loadedC, repBatch, repM, repN,
        typeConverter->convertType(CTensorTy.getElementType()),
        DpasEncodingAttr::OpIdx::OperandC);

    Type resElemTy = DTensorTy.getElementType();

    TritonGEN::PrecisionType APrecision =
                                 getElementPrecision(ATensorTy, resElemTy),
                             BPrecision =
                                 getElementPrecision(BTensorTy, resElemTy);

    assert(APrecision == BPrecision &&
           "A and B precision enumerators do not match");

    LLVM_DEBUG({
      llvm::dbgs() << "repBatch = " << repBatch << "\n";
      llvm::dbgs() << "repM = " << repM << "\n";
      llvm::dbgs() << "repK = " << repK << "\n";
      llvm::dbgs() << "repN = " << repN << "\n";
      llvm::dbgs() << "fc.size()= " << fc.size() << "\n";
    });

    auto generateDPASOp = [&](unsigned b, unsigned m, unsigned n, unsigned k) {
      auto tb = TritonLLVMOpBuilder(loc, rewriter);
      Value valA = ha.at({b, m, k});
      Value valB = hb.at({b, n, k});
      Value valc = fc.at({b, m, n});

      TritonGEN::PrecisionTypeAttr pA =
          TritonGEN::PrecisionTypeAttr::get(A.getContext(), APrecision);
      TritonGEN::PrecisionTypeAttr pB =
          TritonGEN::PrecisionTypeAttr::get(B.getContext(), BPrecision);
      auto RC = IntegerAttr::get(rewriter.getIntegerType(32),
                                 dpasEncoding.getRepeatCount());
      fc.at({b, m, n}) = rewriter.create<TritonGEN::MatrixDPASOp>(
          loc, dTy, tb.bitcast(valc, cTy), tb.bitcast(valA, aTy),
          tb.bitcast(valB, bTy), pA, pB, RC);
    };

    ArrayRef<unsigned> repCluster = dpasEncoding.getRepCluster();
    unsigned rank = repCluster.size();

    auto innerLoop = [&](int b, int k, int outer, unsigned repNumM,
                         unsigned repNumN, unsigned repInner,
                         bool reverseLoop = false) {
      auto body = [&](int b, int k, int outer, int inner) {
        if (repNumM > repNumN)
          generateDPASOp(b, inner, outer, k);
        else
          generateDPASOp(b, outer, inner, k);
      };

      if (reverseLoop) {
        for (int inner = repInner - 1; inner >= 0; --inner)
          body(b, k, outer, inner);
        return;
      }

      for (int inner = 0; inner < repInner; ++inner)
        body(b, k, outer, inner);
    };

    // Use the smaller of the two dimensions as the outer loop for better DPAS
    // operands locality.
    bool aggressiveReusing =
        triton::tools::getBoolEnv("TRITON_INTEL_AGGRESSIVE_DPAS_REUSE");
    unsigned repNumM = repM * repCluster[rank - 2];
    unsigned repNumN = repN * repCluster[rank - 1];
    unsigned repOuter = repNumM > repNumN ? repNumN : repNumM;
    unsigned repInner = repNumM > repNumN ? repNumM : repNumN;
    for (int b = 0; b < repBatch; ++b)
      for (int k = 0; k < repK; ++k)
        for (int outer = 0; outer < repOuter; ++outer) {
          // Change the inner loop direction in odd outer loop iteration if
          // aggressive reuse DPAS operands.
          bool reverseLoop = aggressiveReusing && ((outer % 2) == 1);
          innerLoop(b, k, outer, repNumM, repNumN, repInner, reverseLoop);
        }

    Value res = composeValuesToDotOperandLayoutStruct(fc, repBatch, repM, repN,
                                                      resElemTy);

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
                                              int64_t dimBatch, int64_t dimRow,
                                              int64_t dimCol,
                                              Type elemTy) const {
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    size_t rank = repCluster.size();
    std::vector<Value> elems;
    for (unsigned b = 0; b < dimBatch; ++b) {
      for (int m = 0; m < dimRow; ++m) {
        for (int k = 0; k < dimCol; ++k) {
          for (int repRow = 0; repRow < repCluster[rank - 2]; ++repRow) {
            for (int repCol = 0; repCol < repCluster[rank - 1]; ++repCol) {
              Value matVal = vals.at({b, m * repCluster[rank - 2] + repRow,
                                      k * repCluster[rank - 1] + repCol});
              VectorType vecType = cast<mlir::VectorType>(matVal.getType());
              Type valTy = vecType.getElementType();
              for (int i = 0; i < vecType.getNumElements(); ++i) {
                Value val = tb.extract_element(valTy, matVal, tb.i32_val(i));
                elems.push_back(val);
              }
            }
          }
        }
      }
    }

    assert(!elems.empty() &&
           "unexpected empty result in composing the DPAS result.");

    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(elems.size(), elemTy));
    return packLLElements(loc, typeConverter, elems, rewriter, structTy);
  }

  ValueTable
  getValuesFromDotOperandLayoutStruct(Value val, int64_t batch, int64_t outer,
                                      int64_t inner, Type elemTy,
                                      DpasEncodingAttr::OpIdx opIdx) const {
    SmallVector<Value> elems = unpackLLElements(loc, val, rewriter);
    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    size_t rank = repCluster.size();
    unsigned repClusterOuter = 0u;
    unsigned repClusterInner = 0u;
    bool isOperandA = false;
    bool isOperandB = false;
    bool isFToTF32Enabled = false;
    switch (opIdx) {
    case DpasEncodingAttr::OpIdx::OperandA:
      // operand A
      repClusterOuter = repCluster[rank - 2];
      repClusterInner = 1;
      isOperandA = true;
      break;
    case DpasEncodingAttr::OpIdx::OperandB:
      // operand B
      repClusterInner = 1;
      repClusterOuter = repCluster[rank - 1];
      isOperandB = true;
      break;
    case DpasEncodingAttr::OpIdx::OperandC:
      // operand C
      repClusterOuter = repCluster[rank - 2];
      repClusterInner = repCluster[rank - 1];
      break;
    }

    size_t totalElems = elems.size();
    size_t numElemsPerOperand =
        totalElems /
        ((batch * outer * inner) * (repClusterOuter * repClusterInner));
    VectorType dotOpTy = vec_ty(elemTy, numElemsPerOperand);

    isFToTF32Enabled = elemTy.isFloat(32) && (isOperandA || isOperandB);

    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    int offset = 0;
    ValueTable vals;
    for (unsigned b = 0; b < batch; ++b) {
      for (int i = 0; i < outer; ++i) {
        for (int j = 0; j < inner; ++j) {
          for (int repOuter = 0; repOuter < repClusterOuter; ++repOuter) {
            for (int repInner = 0; repInner < repClusterInner; ++repInner) {
              Value matVal = rewriter.create<LLVM::UndefOp>(loc, dotOpTy);
              if (numElemsPerOperand != 1)
                for (int k = 0; k < numElemsPerOperand; ++k)
                  matVal = tb.insert_element(dotOpTy, matVal, elems[offset++],
                                             tb.i32_val(k));
              else
                matVal = elems[offset++];
              if (isFToTF32Enabled)
                matVal = rewriter.create<TritonGEN::FToTf32Op>(loc, matVal)
                             .getResult();
              vals[{b, i * repClusterOuter + repOuter,
                    j * repClusterInner + repInner}] = matVal;
            }
          }
        }
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
      if (isa<Float8E5M2Type>(elemType))
        return TritonGEN::PrecisionType::F8E5M2;
      if (isa<Float8E4M3FNType>(elemType)) {
        return TritonGEN::PrecisionType::F8E4M3FN;
      }
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

  auto ATensorTy = cast<RankedTensorType>(A.getType()),
       BTensorTy = cast<RankedTensorType>(B.getType()),
       CTensorTy = cast<RankedTensorType>(C.getType()),
       DTensorTy = cast<RankedTensorType>(D.getType());

  assert(isa<DotOperandEncodingAttr>(ATensorTy.getEncoding()) &&
         isa<DotOperandEncodingAttr>(BTensorTy.getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");
  assert(isa<DpasEncodingAttr>(CTensorTy.getEncoding()) &&
         isa<DpasEncodingAttr>(DTensorTy.getEncoding()) &&
         "Currently, we only support $c and $d with a dpas layout.");
  assert(CTensorTy.getShape()[0] == DTensorTy.getShape()[0] &&
         CTensorTy.getShape()[1] == DTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto dpasLayout = cast<DpasEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());

  DotOpDPASConversionHelper helper(dpasLayout, rewriter, typeConverter,
                                   op.getLoc());

  return helper.convertDot(op, adaptor);
}
} // namespace fma_details
