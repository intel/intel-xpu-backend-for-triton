#include "../TritonGPUToLLVMBase.h"
#include "../Utility.h"

#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <thread>

using namespace mlir;
using namespace mlir::triton;
using mlir::triton::gpu::DotOperandEncodingAttr;
using mlir::triton::gpu::DpasEncodingAttr;
using ::XPU::ConvertTritonGPUOpToLLVMPattern;
using ::XPU::ConvertTritonGPUOpToLLVMPatternBase;
using ::XPU::TritonGPUToLLVMTypeConverter;

namespace {

class DotOpDPASConversionHelper {
public:
  using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

  DotOpDPASConversionHelper(DpasEncodingAttr dpasLayout,
                            ConversionPatternRewriter &rewriter,
                            TritonGPUToLLVMTypeConverter *typeConverter,
                            Location loc)
      : dpasLayout(dpasLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(dpasLayout.getContext()) {}

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

    auto repA =
        AEncoding.getDPASRep(ATensorTy.getShape(), ATensorTy.getElementType());
    auto repB =
        BEncoding.getDPASRep(BTensorTy.getShape(), BTensorTy.getElementType());
    assert(repA[1] == repB[0] && "Unexpected rep for A and B operands");

    unsigned repM = repA[0], repN = repB[1], repK = repA[1];

    ValueTable ha = getValuesFromDotOperandLayoutStruct(
        loadedA, repM, repK, ATensorTy.getElementType());
    ValueTable hb = getValuesFromDotOperandLayoutStruct(
        loadedB, repN, repK, BTensorTy.getElementType());
    Type resElemTy = DTensorTy.getElementType();
    SmallVector<Value> fc =
        typeConverter->unpackLLElements(loc, loadedC, rewriter);

    GENX::PrecisionType APrecision = getElementPrecision(ATensorTy, resElemTy),
                        BPrecision = getElementPrecision(BTensorTy, resElemTy);

    LLVM_DEBUG({
      llvm::dbgs() << "repM = " << repM << "\n";
      llvm::dbgs() << "repK = " << repK << "\n";
      llvm::dbgs() << "repN = " << repN << "\n";
      llvm::dbgs() << "fc.size()= " << fc.size() << "\n";
    });

    unsigned RC =
        AEncoding.getParent().cast<DpasEncodingAttr>().getRepeatCount();
    unsigned cNumElems = RC;
    auto CTy = vec_ty(resElemTy, cNumElems);

    for (unsigned m = 0; m < repM; ++m) {
      for (unsigned n = 0; n < repN; ++n) {
        Value C = undef(CTy);
        for (unsigned v = 0; v < cNumElems; ++v)
          C = insert_element(
              CTy, C, fc[m * repN * cNumElems + n * cNumElems + v], i32_val(v));

        for (size_t k = 0; k < repK; k++) {
          Value A = ha[{m, k}], B = hb[{n, k}];
          C = generateDPASOp(C, A, B, RC, APrecision, BPrecision);
        }

        for (unsigned v = 0; v < cNumElems; ++v)
          fc[m * repN * cNumElems + n * cNumElems + v] =
              extract_element(resElemTy, C, i32_val(v));
      }
    }

    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), resElemTy));
    Value res = typeConverter->packLLElements(loc, fc, rewriter, structTy);
    rewriter.replaceOp(op, res);

    return success();
  }

private:
  /// Return the bit width corresponding to the given precision or std::nullopt
  /// if it cannot be computed.
  std::optional<unsigned> getBitWidth(GENX::PrecisionType PT) const {
    switch (PT) {
    case GENX::PrecisionType::U2:
    case GENX::PrecisionType::S2:
      return 2;
    case GENX::PrecisionType::U4:
    case GENX::PrecisionType::S4:
      return 4;
    case GENX::PrecisionType::U8:
    case GENX::PrecisionType::S8:
      return 8;
    case GENX::PrecisionType::BF16:
    case GENX::PrecisionType::FP16:
      return 16;
    case GENX::PrecisionType::TF32:
      return 32;
    default:
      return std::nullopt;
    }
  }

  /// Generate the GENX dialect dpas operation. Rules (for PVC):
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
  ///      genx.matrix.dpas C, A, B {pa, pb, rc=M, sd=8}
  ///        : (vector<8xf32>, vector<8xf16>, vector<16xf16>) -> vector<8xf32>
  ///
  Value generateDPASOp(Value C, Value A, Value B, unsigned RepeatCount,
                       GENX::PrecisionType APrecision,
                       GENX::PrecisionType BPrecision) const {
    assert(RepeatCount == 8 && "RepeatCount should be 8");
    assert(APrecision == BPrecision &&
           "A and B precision enumerators do not match");

    auto ATy = cast<VectorType>(A.getType()),
         BTy = cast<VectorType>(B.getType()),
         CTy = cast<VectorType>(C.getType());
    VectorType resTy = CTy;

    assert(ATy.hasRank() && ATy.getRank() == 1 && "Unexpected vector");
    assert(BTy.hasRank() && BTy.getRank() == 1 && "Unexpected vector");
    assert(CTy.hasRank() && CTy.getRank() == 1 && "Unexpected vector");

    auto pA = GENX::PrecisionTypeAttr::get(A.getContext(), APrecision);
    auto pB = GENX::PrecisionTypeAttr::get(B.getContext(), BPrecision);
    auto RC = IntegerAttr::get(rewriter.getIntegerType(32), RepeatCount);

    return rewriter.create<GENX::MatrixDPASOp>(loc, resTy, C, A, B, pA, pB, RC);
  }

  ValueTable getValuesFromDotOperandLayoutStruct(Value val, int64_t dim0,
                                                 int64_t dim1,
                                                 Type elemTy) const {
    SmallVector<Value> elems =
        typeConverter->unpackLLElements(loc, val, rewriter);
    ValueTable vals;
    for (int64_t i = 0; i < dim0; i++) {
      for (int64_t j = 0; j < dim1; j++)
        vals[{i, j}] = elems[dim1 * i + j];
    }
    return vals;
  }

  /// Return the precision for the given tensor type (the type of A or B) and
  /// result element type.
  GENX::PrecisionType getElementPrecision(RankedTensorType tensorTy,
                                          Type resElemType) const {
    assert((isa<IntegerType>(resElemType) || isa<FloatType>(resElemType)) &&
           "Expecting an integer or floating point type");

    Type elemType = tensorTy.getElementType();
    unsigned width = elemType.getIntOrFloatBitWidth();
    assert(width <= 32 && "Unexpected width");

    if (isa<FloatType>(resElemType)) {
      if (width == 32)
        return GENX::PrecisionType::TF32;
      if (width == 16)
        return isa<FloatType>(elemType)
                   ? GENX::PrecisionType::FP16
                   : GENX::PrecisionType::BF16; // bf is passed as i16.
    } else if (width == 8) {
      return elemType.isUnsignedInteger() ? GENX::PrecisionType::U8
                                          : GENX::PrecisionType::S8;
    }

    return GENX::PrecisionType::PRECISION_UNUSED;
  }

  DpasEncodingAttr dpasLayout;
  ConversionPatternRewriter &rewriter;
  TritonGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx;
};

} // namespace

namespace XPU {
LogicalResult convertDPAS(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonGPUToLLVMTypeConverter *typeConverter,
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
} // namespace XPU
