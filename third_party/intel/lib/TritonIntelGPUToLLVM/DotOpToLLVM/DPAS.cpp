#include "../TritonGPUToLLVMBase.h"
#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "intel/include/Analysis/DPAS.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
#include <optional>

using namespace mlir;

namespace ttgi = mlir::triton::gpu::intel;

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

  template <typename DPASEngineType,
            typename = std::enable_if_t<
                llvm::is_one_of<DPASEngineType, ttgi::DPASEngineTypeXe2,
                                ttgi::DPASEngineTypeXe3P>::value>>
  std::tuple<Type, Type, Type, Type> static getDPASOperandsType(
      DPASEngineType dpasType, MLIRContext *ctx, DpasEncodingAttr layout) {
    if constexpr (std::is_same_v<DPASEngineType, ttgi::DPASEngineTypeXe2>)
      return getDPASOperandsType(static_cast<ttgi::DPASEngineTypeXe2>(dpasType),
                                 ctx, layout);
    if constexpr (std::is_same_v<DPASEngineType, ttgi::DPASEngineTypeXe3P>)
      return getDPASOperandsType(
          static_cast<ttgi::DPASEngineTypeXe3P>(dpasType), ctx, layout);
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
  template <typename OpTy, typename DPASEngineType,
            typename = std::enable_if_t<
                llvm::is_one_of<DPASEngineType, ttgi::DPASEngineTypeXe2,
                                ttgi::DPASEngineTypeXe3P>::value>>
  LogicalResult convertDot(OpTy op, typename OpTy::Adaptor adaptor) const {
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

    auto dpasType = ttgi::DPASAnalysis<DPASEngineType>::getDPASType(op);
    auto dpasEncoding = cast<DpasEncodingAttr>(DTensorTy.getEncoding());
    Type aTy, bTy, cTy, dTy;
    std::tie(dTy, cTy, aTy, bTy) = getDPASOperandsType<DPASEngineType>(
        dpasType, op.getContext(), dpasEncoding);
    // An fp8 operand that is extended to fp16 for DPAS (see
    // decomposeMixedModeDotOp) is converted here, one DPAS operand at a time,
    // instead of in full by the `tt.fp_to_fp` lowering ahead of the dot.
    std::optional<LazyFp8Operand> lazyA, lazyB;
    if constexpr (std::is_same<OpTy, DotOp>::value) {
      lazyA = getLazyFp8Operand(A, loadedA, repBatch, repM, repK,
                                DpasEncodingAttr::OpIdx::OperandA);
      lazyB = getLazyFp8Operand(B, loadedB, repBatch, repN, repK,
                                DpasEncodingAttr::OpIdx::OperandB);
    }
    ValueTable ha =
        lazyA ? ValueTable()
              : getValuesFromDotOperandLayoutStruct(
                    loadedA, ATensorTy, repBatch, repM, repK,
                    typeConverter->convertType(ATensorTy.getElementType()),
                    DpasEncodingAttr::OpIdx::OperandA);
    ValueTable hb =
        lazyB ? ValueTable()
              : getValuesFromDotOperandLayoutStruct(
                    loadedB, BTensorTy, repBatch, repN, repK,
                    typeConverter->convertType(BTensorTy.getElementType()),
                    DpasEncodingAttr::OpIdx::OperandB);
    auto getA = [&](unsigned b, unsigned m, unsigned k) {
      return lazyA ? lazyA->get({b, m, k}, rewriter, loc) : ha.at({b, m, k});
    };
    auto getB = [&](unsigned b, unsigned n, unsigned k) {
      return lazyB ? lazyB->get({b, n, k}, rewriter, loc) : hb.at({b, n, k});
    };
    ValueTable fc = getValuesFromDotOperandLayoutStruct(
        loadedC, CTensorTy, repBatch, repM, repN,
        typeConverter->convertType(CTensorTy.getElementType()),
        DpasEncodingAttr::OpIdx::OperandC);

    ValueTable scaleA, scaleB;
    if constexpr (std::is_same<OpTy, DotScaledOp>::value) {
      auto scaleAVal = adaptor.getAScale();
      if (scaleAVal) {
        scaleA = getScaleValuesFromDotOperandLayoutStruct(
            scaleAVal, repBatch, repM, repK, 3 /*opIdx*/,
            getIntOrFloatOrPtrBitWidth(ATensorTy.getElementType()) == 16);
      }

      auto scaleBVal = adaptor.getBScale();
      if (scaleBVal) {
        scaleB = getScaleValuesFromDotOperandLayoutStruct(
            scaleBVal, repBatch, repN, repK, 4 /*opIdx*/,
            getIntOrFloatOrPtrBitWidth(BTensorTy.getElementType()) == 16);
      }
    }

    Type resElemTy = DTensorTy.getElementType();

    TritonGEN::PrecisionType APrecision;
    TritonGEN::PrecisionType BPrecision;

    if constexpr (std::is_same<OpTy, DotOp>::value) {
      APrecision = getElementPrecision(ATensorTy, resElemTy),
      BPrecision = getElementPrecision(BTensorTy, resElemTy);

      assert(APrecision == BPrecision &&
             "A and B precision enumerators do not match");
    } else if constexpr (std::is_same<OpTy, DotScaledOp>::value) {
      auto aElemType = adaptor.getAElemType();
      APrecision = getElementPrecision(aElemType);
      auto bElemType = adaptor.getBElemType();
      BPrecision = getElementPrecision(bElemType);
      [[maybe_unused]] bool isBothFP8 =
          (aElemType == triton::ScaleDotElemType::E4M3 ||
           aElemType == triton::ScaleDotElemType::E5M2) &&
          (bElemType == triton::ScaleDotElemType::E4M3 ||
           bElemType == triton::ScaleDotElemType::E5M2);
      assert((isBothFP8 || APrecision == BPrecision) &&
             "A and B precision enumerators do not match");
    }

    LLVM_DEBUG({
      llvm::dbgs() << "repBatch = " << repBatch << "\n";
      llvm::dbgs() << "repM = " << repM << "\n";
      llvm::dbgs() << "repK = " << repK << "\n";
      llvm::dbgs() << "repN = " << repN << "\n";
      llvm::dbgs() << "fc.size()= " << fc.size() << "\n";
    });

    auto generateDPASOp = [&](unsigned b, unsigned m, unsigned n, unsigned k) {
      auto tb = TritonLLVMOpBuilder(loc, rewriter);
      Value valA = getA(b, m, k);
      Value valB = getB(b, n, k);
      Value valc = fc.at({b, m, n});

      TritonGEN::PrecisionTypeAttr pA =
          TritonGEN::PrecisionTypeAttr::get(A.getContext(), APrecision);
      TritonGEN::PrecisionTypeAttr pB =
          TritonGEN::PrecisionTypeAttr::get(B.getContext(), BPrecision);
      auto RC = IntegerAttr::get(rewriter.getIntegerType(32),
                                 dpasEncoding.getRepeatCount());

      if constexpr (std::is_same<OpTy, DotOp>::value) {
        fc.at({b, m, n}) = TritonGEN::MatrixDPASOp::create(
            rewriter, loc, dTy, tb.bitcast(valc, cTy), tb.bitcast(valA, aTy),
            tb.bitcast(valB, bTy), pA, pB, RC);
      } else if constexpr (std::is_same<OpTy, DotScaledOp>::value) {
        Value sA;
        if (!scaleA.empty())
          sA = scaleA.at({b, m, k});
        Value sB;
        if (!scaleB.empty())
          sB = scaleB.at({b, n, k});
        fc.at({b, m, n}) = TritonGEN::MatrixBlockScaleDPASOp::create(
            rewriter, loc, dTy, tb.bitcast(valc, cTy), tb.bitcast(valA, aTy),
            tb.bitcast(valB, bTy), sA, sB, pA, pB, RC);
      }
    };

    ArrayRef<unsigned> repCluster = dpasEncoding.getRepCluster();
    unsigned rank = repCluster.size();

    // The DPAS calls in order, as (b, m, n, k).
    SmallVector<std::array<unsigned, 4>> dpasOrder;
    auto innerLoop = [&](int b, int k, int outer, unsigned repNumM,
                         unsigned repNumN, unsigned repInner,
                         bool reverseLoop = false) {
      auto body = [&](int b, int k, int outer, int inner) {
        if (repNumM > repNumN)
          dpasOrder.push_back(
              {unsigned(b), unsigned(inner), unsigned(outer), unsigned(k)});
        else
          dpasOrder.push_back(
              {unsigned(b), unsigned(outer), unsigned(inner), unsigned(k)});
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

    // A converted operand is created one DPAS call ahead of its first use, so
    // that its conversion overlaps with that call. Right before its own call,
    // each call waits for its conversion; further ahead, the converted
    // operands of more calls are live at once, and the GEMMs closest to the
    // register limit spill again.
    for (size_t i = 0; i < dpasOrder.size(); ++i) {
      for (size_t j = i; j < std::min(i + 2, dpasOrder.size()); ++j) {
        auto [b, m, n, k] = dpasOrder[j];
        getA(b, m, k);
        getB(b, n, k);
      }
      auto [b, m, n, k] = dpasOrder[i];
      generateDPASOp(b, m, n, k);
    }

    Value res = composeValuesToDotOperandLayoutStruct(fc, DTensorTy, repBatch,
                                                      repM, repN, resElemTy);

    rewriter.replaceOp(op, res);
    for (std::optional<LazyFp8Operand> *lazy : {&lazyA, &lazyB})
      if (*lazy && (*lazy)->cast->use_empty())
        rewriter.eraseOp((*lazy)->cast);

    return success();
  }

private:
  /// An fp8 DPAS operand whose fp8 -> fp16 conversion is created on demand,
  /// one DPAS operand (a run of consecutive elements) at a time.
  struct LazyFp8Operand {
    triton::intel::Fp8ToFp16Lowering lowering;
    // The cast that stands for the conversion; erased once the dot is lowered.
    Operation *cast;
    // Range [first, first + count) of the elements of each DPAS operand.
    std::map<std::array<unsigned, 3>, std::pair<unsigned, unsigned>> ranges;
    ValueTable converted;

    Value get(std::array<unsigned, 3> key, ConversionPatternRewriter &rewriter,
              Location loc) {
      if (auto it = converted.find(key); it != converted.end())
        return it->second;
      auto [first, count] = ranges.at(key);
      SmallVector<unsigned> indices(count);
      std::iota(indices.begin(), indices.end(), first);
      SmallVector<Value> elems = lowering.convert(indices, rewriter);
      Value val = elems[0];
      if (count != 1) {
        auto tb = TritonLLVMOpBuilder(loc, rewriter);
        VectorType vecTy = vec_ty(f16_ty, count);
        val = LLVM::UndefOp::create(rewriter, loc, vecTy);
        for (unsigned i = 0; i < count; ++i)
          val = tb.insert_element(vecTy, val, elems[i], tb.i32_val(i));
      }
      return converted[key] = val;
    }
  };

  /// Returns the fp8 source of `operand` if the dot converts it itself (see
  /// isDPASOperandConvertedByDot). `lowered` is the operand's lowered value:
  /// the cast of the fp8 source that the `tt.fp_to_fp` lowering left in place
  /// of the conversion.
  std::optional<LazyFp8Operand>
  getLazyFp8Operand(Value operand, Value lowered, int64_t batch, int64_t outer,
                    int64_t inner, DpasEncodingAttr::OpIdx opIdx) const {
    // Follow the `tt.fp_to_fp` lowering, which left a cast of the fp8 source
    // if the dot converts it, rather than calling isDPASOperandConvertedByDot
    // again: lowering the ops in between can split their block (e.g. the
    // predicated stores of a `ttg.local_alloc`), and it requires one block.
    auto cvt = operand.getDefiningOp<FpToFpOp>();
    auto cast = lowered.getDefiningOp<UnrealizedConversionCastOp>();
    if (!cvt || !cast || cast->getNumOperands() != 1 ||
        !isa<LLVM::LLVMStructType>(cast->getOperand(0).getType()))
      return std::nullopt;
    // The `tt.fp_to_fp` lowering emitted no conversion for `cvt`, so this must
    // not fail from here on.
    Value llSrc = cast->getOperand(0);
    std::optional<triton::intel::Fp8ToFp16Lowering> lowering =
        triton::intel::Fp8ToFp16Lowering::get(cvt, llSrc, rewriter);
    assert(lowering && "expected an fp8 -> fp16 conversion");

    // The element order of getValuesFromDotOperandLayoutStruct.
    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    size_t rank = repCluster.size();
    unsigned repClusterOuter = opIdx == DpasEncodingAttr::OpIdx::OperandA
                                   ? repCluster[rank - 2]
                                   : repCluster[rank - 1];
    size_t numElemsPerOperand =
        lowering->size() / ((batch * outer * inner) * repClusterOuter);
    LazyFp8Operand lazy{std::move(*lowering), cast, {}, {}};
    unsigned offset = 0;
    for (unsigned b = 0; b < batch; ++b)
      for (int i = 0; i < outer; ++i)
        for (int j = 0; j < inner; ++j)
          for (unsigned repOuter = 0; repOuter < repClusterOuter; ++repOuter) {
            lazy.ranges[{b, unsigned(i * repClusterOuter + repOuter),
                         unsigned(j)}] = {offset, numElemsPerOperand};
            offset += numElemsPerOperand;
          }
    return lazy;
  }

  std::tuple<Type, Type, Type, Type> static getDPASOperandsType(
      ttgi::DPASEngineTypeXe2 dpasType, MLIRContext *ctx,
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

    switch (dpasType) {
    case ttgi::DPASEngineTypeXe2::FP32_FP32_FP16_FP16:
    case ttgi::DPASEngineTypeXe2::FP32_FP32_BF16_BF16: {
      Type cTy = vec_ty(fp32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe2::FP16_FP16_FP16_FP16: {
      Type cTy = vec_ty(fp16Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe2::BF16_BF16_BF16_BF16: {
      Type cTy = vec_ty(bf16Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe2::FP32_FP32_TF32_TF32: {
      Type cTy = vec_ty(fp32Ty, elemNumC);
      Type aTy = vec_ty(i32Ty, elemNumA);                 // pack scalar to i32.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe2::U32_U32_U8_U8: {
      Type cTy = vec_ty(i32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack 2 i8 to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe2::S32_S32_S8_S8: {
      Type cTy = vec_ty(s32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack 2 i8 to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    default:
      llvm::report_fatal_error(
          "Unsupported dpas type found: " +
          StringRef(std::to_string(static_cast<int>(dpasType))));
    }

    return std::make_tuple<Type, Type, Type, Type>({}, {}, {}, {});
  }

  std::tuple<Type, Type, Type, Type> static getDPASOperandsType(
      ttgi::DPASEngineTypeXe3P dpasType, MLIRContext *ctx,
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

    switch (dpasType) {
    case ttgi::DPASEngineTypeXe3P::FP32_FP32_FP16_FP16:
    case ttgi::DPASEngineTypeXe3P::FP32_FP32_BF16_BF16: {
      Type cTy = vec_ty(fp32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe3P::FP16_FP16_FP16_FP16: {
      Type cTy = vec_ty(fp16Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe3P::BF16_BF16_BF16_BF16: {
      Type cTy = vec_ty(bf16Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA);                 // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe3P::FP32_FP32_TF32_TF32: {
      Type cTy = vec_ty(fp32Ty, elemNumC);
      Type aTy = vec_ty(i32Ty, elemNumA);                 // pack scalar to i32.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe3P::U32_U32_U8_U8: {
      Type cTy = vec_ty(i32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack 2 i8 to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe3P::S32_S32_S8_S8: {
      Type cTy = vec_ty(s32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack 2 i8 to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe3P::BF16_BF16_FP8_FP8: {
      Type cTy = vec_ty(bf16Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    case ttgi::DPASEngineTypeXe3P::FP32_FP32_FP8_FP8:
    case ttgi::DPASEngineTypeXe3P::FP32_FP32_FP4_FP4: {
      Type cTy = vec_ty(fp32Ty, elemNumC);
      Type aTy = vec_ty(i16Ty, elemNumA / 2);             // pack scalar to i16.
      Type bTy = vec_ty(i32Ty, elemNumB / opsPerChannel); // pack scalar to i32.
      return {cTy, cTy, aTy, bTy};
    }
    default:
      llvm::report_fatal_error(
          "Unsupported dpas type found: " +
          StringRef(std::to_string(static_cast<int>(dpasType))));
    }

    return std::make_tuple<Type, Type, Type, Type>({}, {}, {}, {});
  }

  /// Return the bit width corresponding to the given precision or
  /// std::nullopt if it cannot be computed.
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
                                              RankedTensorType tensorTy,
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

    SmallVector<Value> resultVals(elems.begin(), elems.end());
    return packTensorElements(loc, typeConverter, resultVals, rewriter,
                              tensorTy);
  }

  ValueTable getValuesFromDotOperandLayoutStruct(
      Value val, RankedTensorType tensorTy, int64_t batch, int64_t outer,
      int64_t inner, Type elemTy, DpasEncodingAttr::OpIdx opIdx) const {
    SmallVector<Value> elems =
        unpackTensorElements(loc, val, rewriter, tensorTy);
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
              Value matVal = LLVM::UndefOp::create(rewriter, loc, dotOpTy);
              if (numElemsPerOperand != 1)
                for (int k = 0; k < numElemsPerOperand; ++k)
                  matVal = tb.insert_element(dotOpTy, matVal, elems[offset++],
                                             tb.i32_val(k));
              else
                matVal = elems[offset++];
              if (isFToTF32Enabled)
                matVal = TritonGEN::FToTf32Op::create(rewriter, loc, matVal)
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

  ValueTable getScaleValuesFromDotOperandLayoutStruct(Value val, int64_t batch,
                                                      int64_t outer,
                                                      int64_t inner,
                                                      unsigned opIdx,
                                                      bool isFp16) const {
    SmallVector<Value> elems = unpackLLElements(loc, val, rewriter);

    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    size_t rank = repCluster.size();
    unsigned repClusterOuter = 0u;
    unsigned repClusterInner = 1u;
    switch (opIdx) {
    case 3:
      // operand scale A
      repClusterOuter = repCluster[rank - 2];
      break;
    case 4:
      // operand scale B
      repClusterOuter = repCluster[rank - 1];
      break;
    default:
      llvm_unreachable("error");
    }

    size_t totalElems = elems.size();
    unsigned packedElem = mlir::ceil<unsigned>(
        totalElems, batch * outer * inner * repClusterOuter * repClusterInner);
    unsigned innerStepping = isFp16 ? 2 : 1;

    int offset = 0;
    auto tb = TritonLLVMOpBuilder(loc, rewriter);
    ValueTable vals;
    for (unsigned b = 0; b < batch; ++b) {
      for (int i = 0; i < outer; ++i) {
        for (int j = 0; j < inner; j += innerStepping) {
          for (int repOuter = 0; repOuter < repClusterOuter; ++repOuter) {
            for (int repInner = 0; repInner < repClusterInner; ++repInner) {
              Value matVal;
              if (packedElem != 1) {
                VectorType scaleOpTy = vec_ty(i8_ty, packedElem);
                matVal = LLVM::UndefOp::create(rewriter, loc, scaleOpTy);
                for (int k = 0; k < packedElem; ++k)
                  matVal =
                      tb.insert_element(matVal, elems[offset++], tb.i32_val(k));
              } else
                matVal = elems[offset++];

              unsigned offsetOuter = i * repClusterOuter + repOuter;
              unsigned offsetInner = j * repClusterInner + repInner;
              vals[{b, offsetOuter, offsetInner}] = matVal;

              if (isFp16) {
                // For fp16 bdpas, the size of k dimension is 16.
                // The triton ops share the scale for every 32 elements.
                // So we need to reuse the scale operands for two continuous
                // bdpas ops on the K dim.
                vals[{b, offsetOuter, offsetInner + 1}] = matVal;
              }
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
      if (isa<Float8E4M3FNType>(elemType))
        return TritonGEN::PrecisionType::F8E4M3FN;
    } else if (width == 8) {
      return elemType.isUnsignedInteger() ? TritonGEN::PrecisionType::U8
                                          : TritonGEN::PrecisionType::S8;
    }

    return TritonGEN::PrecisionType::UNUSED;
  }

  /// Return the precision for the given tensor type (the type of A or B).
  TritonGEN::PrecisionType
  getElementPrecision(ScaleDotElemType elemType) const {
    switch (elemType) {
    case ScaleDotElemType::E2M1:
      return TritonGEN::PrecisionType::F4E2M1;
    case ScaleDotElemType::BF16:
      return TritonGEN::PrecisionType::BF16;
    case ScaleDotElemType::FP16:
      return TritonGEN::PrecisionType::FP16;
    case ScaleDotElemType::E4M3:
      return TritonGEN::PrecisionType::F8E4M3FN;
    case ScaleDotElemType::E5M2:
      return TritonGEN::PrecisionType::F8E5M2;
    default:
      llvm::report_fatal_error("Unsupported ScaleDotElemType found");
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

namespace mlir::triton::intel {

bool isDPASOperandConvertedByDot(FpToFpOp op) {
  if (tools::getBoolEnv("TRITON_INTEL_DISABLE_LAZY_FP8_DPAS_OPERANDS"))
    return false;
  // A scalar conversion (e.g. of a loaded scalar) is never a dot operand.
  auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
  auto dstTy = dyn_cast<RankedTensorType>(op.getType());
  if (!srcTy || !dstTy ||
      !isa<Float8E4M3FNType, Float8E5M2Type>(srcTy.getElementType()) ||
      !dstTy.getElementType().isF16() || op.getRounding() || !op->hasOneUse())
    return false;
  // A single use as A or B of a dot lowered to DPAS (see DotOpToLLVM.cpp), in
  // the same block: a conversion hoisted out of a loop (e.g. of the Q operand
  // of attention) must stay there, not be redone in every iteration.
  OpOperand &use = *op->use_begin();
  auto dot = dyn_cast<DotOp>(use.getOwner());
  if (!dot || dot->getBlock() != op->getBlock() || use.getOperandNumber() > 1 ||
      !isa<DpasEncodingAttr>(dot.getType().getEncoding()) ||
      gpu::getShapePerCTA(dot.getA().getType())[1] == 1)
    return false;
  auto enc = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  auto dpas = enc ? dyn_cast<DpasEncodingAttr>(enc.getParent()) : nullptr;
  if (!dpas || dpas != dot.getType().getEncoding() || dstTy.getRank() != 2)
    return false;
  // The lowered operand must hold every element once (no register broadcast),
  // and each DPAS operand must be a multiple of 4 elements, so that the
  // converter's groups of elements start where they do when the whole operand
  // is converted.
  if (gpu::getUniqueElemsPerThread(srcTy) != gpu::getTotalElemsPerThread(srcTy))
    return false;
  auto opIdx = DpasEncodingAttr::OpIdx(enc.getOpIdx());
  SmallVector<int64_t> reps = dpas.getDPASRepetitions(dstTy.getShape(), opIdx);
  ArrayRef<unsigned> repCluster = dpas.getRepCluster();
  int64_t repClusterOuter = opIdx == DpasEncodingAttr::OpIdx::OperandA
                                ? repCluster[0]
                                : repCluster[1];
  int64_t numOperands = reps[0] * reps[1] * reps[2] * repClusterOuter;
  int64_t totalElems = gpu::getTotalElemsPerThread(srcTy);
  if (numOperands <= 0 || totalElems % numOperands != 0 ||
      (totalElems / numOperands) % 4 != 0)
    return false;
  // Only for a dot with at least 4 DPAS calls per K slice: with fewer, there
  // is too little DPAS work to overlap a conversion with, and each call waits
  // for the conversion of its operands.
  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  int64_t dpasPerKSlice =
      dpas.getDPASRepetitions(aTy.getShape(),
                              DpasEncodingAttr::OpIdx::OperandA)[1] *
      repCluster[0] *
      dpas.getDPASRepetitions(bTy.getShape(),
                              DpasEncodingAttr::OpIdx::OperandB)[2] *
      repCluster[1];
  if (dpasPerKSlice < 4)
    return false;
  // Only for a dot whose operands and accumulator take at least half of the
  // registers of a lane in 256-GRF mode: there, converting all of an operand up
  // front spills, while a dot well within the registers runs faster with the
  // conversion up front. With 4 DPAS calls per K slice and an fp32
  // accumulator, this only rejects dots with K < 64, such as the P * V dot of
  // decode attention (BLOCK_M = 16, BLOCK_N = 32).
  constexpr int64_t minBytesPerLane = 512;
  auto bytesPerLane = [](Value v) {
    auto ty = cast<RankedTensorType>(v.getType());
    return int64_t(gpu::getTotalElemsPerThread(ty)) *
           ty.getElementTypeBitWidth() / 8;
  };
  return bytesPerLane(dot.getA()) + bytesPerLane(dot.getB()) +
             bytesPerLane(dot.getC()) >=
         minBytesPerLane;
}

} // namespace mlir::triton::intel

namespace fma_details {
template <typename OpTy>
LogicalResult convertDPAS(OpTy op, typename OpTy::Adaptor adaptor,
                          TritonIntelGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto mod = op->template getParentOfType<ModuleOp>();

  LLVM_DEBUG({
    llvm::dbgs() << "module before DPAS generation\n";
    mod->dump();
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

  bool supportDPASWithBF8 = mod->hasAttr(
      ttgi::TritonIntelGPUDialect::getSupportDPASWithBF8AttrName());

  if (!supportDPASWithBF8)
    return helper.convertDot<OpTy, ttgi::DPASEngineTypeXe2>(op, adaptor);

  return helper.convertDot<OpTy, ttgi::DPASEngineTypeXe3P>(op, adaptor);
}

template LogicalResult
convertDPAS<DotOp>(DotOp op, DotOp::Adaptor adaptor,
                   TritonIntelGPUToLLVMTypeConverter *typeConverter,
                   ConversionPatternRewriter &rewriter);

template LogicalResult
convertDPAS<DotScaledOp>(DotScaledOp op, DotScaledOp::Adaptor adaptor,
                         TritonIntelGPUToLLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter);

} // namespace fma_details
