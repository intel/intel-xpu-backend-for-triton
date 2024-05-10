#include "../TritonGPUToLLVMBase.h"
#include "../Utility.h"
#include "mlir/IR/BuiltinTypes.h"

#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu::intel;

namespace {

class DotOpDPASConversionHelper {
public:
  using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

  DotOpDPASConversionHelper(DpasEncodingAttr dpasLayout, Type tensorTy,
                            ConversionPatternRewriter &rewriter,
                            const TargetInfoBase &targetInfo,
                            TritonIntelGPUToLLVMTypeConverter *typeConverter,
                            Location loc)
      : dpasLayout(dpasLayout), rewriter(rewriter),
        typeConverter(typeConverter), targetInfo(targetInfo), loc(loc),
        ctx(dpasLayout.getContext()), tensorTy(tensorTy) {}

  std::tuple<Type, Type, Type, Type> static getDPASOperandsType(
      DPASAnalysis::DPASEngineType dpasType, MLIRContext *ctx,
      DpasEncodingAttr layout) {
    Type fp32Ty = type::f32Ty(ctx);
    Type fp16Ty = type::f16Ty(ctx);
    Type bf16Ty = type::bf16Ty(ctx);
    Type i32Ty = type::i32Ty(ctx);
    Type i16Ty = type::i16Ty(ctx);
    Type s32Ty = IntegerType::get(ctx, 32, IntegerType::Signed);

    unsigned threadsPerWarp = layout.getSubGroupSize();
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
    assert(repA[1] == repB[0] && "Unexpected rep for A and B operands");

    unsigned repM = repA[0], repN = repB[1], repK = repA[1];

    auto dpasType = DPASAnalysis::getDPASType(op);
    auto dpasEncoding = cast<DpasEncodingAttr>(DTensorTy.getEncoding());
    Type aTy, bTy, cTy, dTy;
    std::tie(dTy, cTy, aTy, bTy) =
        getDPASOperandsType(dpasType, op.getContext(), dpasEncoding);
    ValueTable ha = getValuesFromDotOperandLayoutStruct(
        loadedA, repM, repK,
        typeConverter->convertType(ATensorTy.getElementType()), aTy, 0);
    ValueTable hb = getValuesFromDotOperandLayoutStruct(
        loadedB, repN, repK,
        typeConverter->convertType(BTensorTy.getElementType()), bTy, 1);
    ValueTable fc = getValuesFromDotOperandLayoutStruct(
        loadedC, repM, repN,
        typeConverter->convertType(CTensorTy.getElementType()), cTy, 2);

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

#if 0
    auto mod = op->getParentOfType<ModuleOp>();
    auto warpSize =
        i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpId = udiv(getThreadId(rewriter, loc), warpSize);
    Value laneId = urem(getThreadId(rewriter, loc), warpSize);
    Value programId =
        targetInfo.programId(rewriter, loc,
                             rewriter.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>(),
                             0);
#endif

    auto generateDPASOp = [&](unsigned m, unsigned n, unsigned k) {
      Value valA = ha.at({m, k});
      Value valB = hb.at({n, k});
      Value valc = fc.at({m, n});
#if 0
        targetInfo.printf(rewriter,
                          "A pid=%d sgid=%d,tid=%d, m=%d, k=%d, val=%f",
                          ValueRange{programId, warpId, laneId, i32_val(m),
                                     i32_val(k), valA});
        targetInfo.printf(rewriter,
                          "B pid=%d sgid=%d,tid=%d, n=%d, k=%d, val=%f",
                          ValueRange{programId, warpId, laneId, i32_val(n),
                                     i32_val(k), valB});
        targetInfo.printf(rewriter,
                          "C pid=%d sgid=%d,tid=%d, n=%d, k=%d, val=%f",
                          ValueRange{programId, warpId, laneId, i32_val(n),
                                     i32_val(k), valc});
#endif
      TritonGEN::PrecisionTypeAttr pA =
          TritonGEN::PrecisionTypeAttr::get(A.getContext(), APrecision);
      TritonGEN::PrecisionTypeAttr pB =
          TritonGEN::PrecisionTypeAttr::get(B.getContext(), BPrecision);
      auto RC = IntegerAttr::get(rewriter.getIntegerType(32),
                                 dpasEncoding.getRepeatCount());
      fc.at({m, n}) = rewriter.create<TritonGEN::MatrixDPASOp>(
          loc, dTy, bitcast(valc, cTy), bitcast(valA, aTy), bitcast(valB, bTy),
          pA, pB, RC);
    };

    ArrayRef<unsigned> repCluster = dpasEncoding.getRepCluster();
    for (int k = 0; k < repK; ++k)
      for (int m = 0; m < repM; ++m)
        for (int n = 0; n < repN; ++n)
          for (int repRow = 0; repRow < repCluster[0]; ++repRow)
            for (int repCol = 0; repCol < repCluster[1]; ++repCol)
              generateDPASOp(m * repCluster[0] + repRow,
                             n * repCluster[1] + repCol, k);

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

    //    int threadsPerWarp = triton::gpu::getWarpSize(dpasLayout);
    //
    //    Value warpSize = i32_val(threadsPerWarp);
    //    Value warpId = udiv(getThreadId(rewriter, loc), warpSize);
    //    Value laneId = urem(getThreadId(rewriter, loc), warpSize);
    //    Value programId =
    //        targetInfo.programId(rewriter, loc,
    //        rewriter.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>(),
    //        0);

    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    std::vector<Value> elems;
    for (int m = 0; m < dim0; ++m) {
      for (int k = 0; k < dim1; ++k) {
        for (int repRow = 0; repRow < repCluster[0]; ++repRow) {
          for (int repCol = 0; repCol < repCluster[1]; ++repCol) {
            Value matVal = vals.at(
                {m * repCluster[0] + repRow, k * repCluster[1] + repCol});

            //        LLVM::intel::llPrintf(rewriter, "A pid=%d sgid=%d,tid=%d,
            //        m=%d, n=%d, val=%f",
            //                              ValueRange{programId, warpId,
            //                              laneId, i32_val(m), i32_val(k),
            //                              matVal});
            VectorType vecType = cast<mlir::VectorType>(matVal.getType());
            Type valTy = vecType.getElementType();
            for (int i = 0; i < vecType.getNumElements(); ++i) {
              Value val = extract_element(valTy, matVal, i32_val(i));
              elems.push_back(val);
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

  ValueTable getValuesFromDotOperandLayoutStruct(Value val, int64_t outer,
                                                 int64_t inner, Type elemTy,
                                                 Type dotOperandType,
                                                 uint32_t opIdx) const {
    SmallVector<Value> elems = unpackLLElements(loc, val, rewriter);
    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    unsigned repClusterOuter = 0u;
    unsigned repClusterInner = 0u;
    switch (opIdx) {
    case 0:
      // operand A
      repClusterOuter = repCluster[0];
      repClusterInner = 1;
      break;
    case 1:
      // operand B
      repClusterInner = 1;
      repClusterOuter = repCluster[1];
      break;
    case 2:
      // operand C
      repClusterOuter = repCluster[0];
      repClusterInner = repCluster[1];
      break;
    default:
      assert(false && "invalid operand type in lowering");
      break;
    }

    size_t totalElems = elems.size();
    size_t numElemsPerOperand =
        totalElems / ((outer * inner) * (repClusterOuter * repClusterInner));
    VectorType dotOpTy = vec_ty(elemTy, numElemsPerOperand);

    int offset = 0;
    ValueTable vals;
    for (int i = 0; i < outer; ++i) {
      for (int j = 0; j < inner; ++j) {
        for (int repOuter = 0; repOuter < repClusterOuter; ++repOuter) {
          for (int repInner = 0; repInner < repClusterInner; ++repInner) {
            Value matVal = rewriter.create<LLVM::UndefOp>(loc, dotOpTy);
            for (int k = 0; k < numElemsPerOperand; ++k) {
              matVal =
                  insert_element(dotOpTy, matVal, elems[offset++], i32_val(k));
            }
            vals[{i * repClusterOuter + repOuter,
                  j * repClusterInner + repInner}] = matVal;
            //                bitcast(matVal, dotOperandType);
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
  const TargetInfoBase &targetInfo;
  Type tensorTy;
};

} // namespace

namespace fma_details {
LogicalResult convertDPAS(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonIntelGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter,
                          const TargetInfoBase &targetInfo) {
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

  DotOpDPASConversionHelper helper(dpasLayout, DTensorTy, rewriter, targetInfo,
                                   typeConverter, op.getLoc());

  return helper.convertDot(op, adaptor);
}
} // namespace fma_details
