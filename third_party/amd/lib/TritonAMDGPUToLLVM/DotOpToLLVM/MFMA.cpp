/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifdef USE_ROCM

#include "../PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::LLVM::AMD::shuffleXor;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

struct DotOpMFMAConversionHelper {
  AMDMfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  const LLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConversionHelper(AMDMfmaEncodingAttr mfmaLayout,
                                     ConversionPatternRewriter &rewriter,
                                     const LLVMTypeConverter *typeConverter,
                                     Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }

  Value generateMFMAOp(StringRef mfmaInsnName, Value valA, Value valB,
                       Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    OperationState loweredOp(loc, mfmaInsnName);
    loweredOp.addTypes(resType);
    loweredOp.addOperands({valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    return rewriter.create(loweredOp)->getResult(0);
  }

  int getNumSubmatrices(Type elementType, int mDim, int nDim) const {
    if (mDim == 64 && nDim == 4 || mDim == 4 && nDim == 64)
      return 1;
    assert(mDim == nDim);
    switch (mDim) {
    case 32:
    case 16:
      return 1;
      break;
    case 4:
      assert(elementType.getIntOrFloatBitWidth() <= 32 &&
             "fp64 is not supported yet");
      assert(elementType.getIntOrFloatBitWidth() != 8 ||
             elementType.isInteger(8) && "fp8 is not supported yet");
      return 16;
      break;
    default:
      llvm::report_fatal_error("unsupported nonKDim in MFMA dot");
    }
    return -1;
  }

  Value processSubBlocks(int numSubBlocks, Value acc, bool reduceSubBlocks,
                         bool zeroSubBlocks) const {
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 &&
           "numSubBlocks in not pow 2!");
    if (numSubBlocks == 1)
      return acc;
    constexpr int waveSize = 64;
    int subBlockSize = waveSize / numSubBlocks;
    Value laneId = getThreadId();
    laneId = and_(laneId, i32_val(waveSize - 1));
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = extract_element(elemType, acc, i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < waveSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc =
              shuffleXor(loc, rewriter, accScalar[i], subBlockSize);
          if (elemType.isInteger(32))
            accScalar[i] = add(accScalar[i], other_acc);
          else
            accScalar[i] = fadd(accScalar[i], other_acc);
        }
        subBlockSize *= 2;
      }
    }
    if (zeroSubBlocks) {
      Value zero;
      if (elemType.isInteger(32))
        zero = i32_val(0);
      else
        zero = f32_val(0.0);
      auto cond = icmp_ult(laneId, i32_val(subBlockSize));
      for (int i = 0; i < numScalars; ++i)
        accScalar[i] = select(cond, accScalar[i], zero);
    }

    Value reducedAcc = undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc = insert_element(vecTy, reducedAcc, accScalar[i], i32_val(i));
    return reducedAcc;
  }

  /// @brief MFMA 4x4 is computes 16 matrix multiplications, this functions adds
  /// these 16 matrices to get final 4x4 matrix
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }

  /// @brief Zeroes out redundant values in all sub-blocks except first one
  ///
  /// Every wave in mfma 4x4 layout holds only 4 unique values(scalar or
  /// vectors) in blocks of 4 consecutive threads, There are 16 copies of these
  /// 4 values across all threads of the wave. Need to zero out 15 copies to use
  /// accumulator between dot operations.
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    auto mfmaVersion = mfmaLayout.getVersionMajor();
    assert((mDim == nDim && (mDim == 32 || mDim == 16 || mDim == 4)) ||
           (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = a.getType().cast<RankedTensorType>();
    auto bTensorTy = b.getType().cast<RankedTensorType>();
    auto dTensorTy = d.getType().cast<RankedTensorType>();
    auto elemTyA = aTensorTy.getElementType();
    auto elemTyB = bTensorTy.getElementType();

    StringRef mfmaInsnName;
    auto maybeMfmaInsn =
        MfmaInsn::selectMfma(mDim, nDim, elemTyA, elemTyB, mfmaVersion);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");
    else
      mfmaInsnName = (*maybeMfmaInsn).getInsnName();

    auto aEncoding = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto bEncoding = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    int kWidth = aEncoding.getKWidth();

    auto repA =
        mfmaLayout.getMFMARepForOperands(aTensorTy.getShape(), kWidth, 0);
    auto repB =
        mfmaLayout.getMFMARepForOperands(bTensorTy.getShape(), kWidth, 1);

    assert(repA[1] == repB[0]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[0];
    auto numRepN = repB[1];
    auto numRepK = repA[1];

    ValueTable ha = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepM, numRepK, kWidth, aTensorTy.getElementType());
    ValueTable hb = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepN, numRepK, kWidth, aTensorTy.getElementType());
    auto dstElemTy = dTensorTy.getElementType();
    auto fc = unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::getWarpSize(mfmaLayout);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), mDim, nDim);
    auto elemsPerVec = mDim * nDim * subBlocks / warpSize;

    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        Value acc = undef(vecTy);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          acc = insert_element(
              vecTy, acc, fc[m * numRepN * elemsPerVec + n * elemsPerVec + v],
              i32_val(v));
        }
        acc = zeroAuxiliarBlocks(subBlocks, acc);
        for (size_t k = 0; k < numRepK; k++) {
          acc = mfmaLayout.getIsTransposed()
                    ? generateMFMAOp(mfmaInsnName, hb[{n, k}], ha[{m, k}], acc)
                    : generateMFMAOp(mfmaInsnName, ha[{m, k}], hb[{n, k}], acc);
        }
        acc = reduceSubBlocks(subBlocks, acc);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          fc[m * numRepN * elemsPerVec + n * elemsPerVec + v] =
              extract_element(dstElemTy, acc, i32_val(v));
        }
      }
    }

    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);
    rewriter.replaceOp(op, res);

    return success();
  }

  /**
   * @brief Converts dot operand structure to value table and converts types
   * appropriate for mfma instructions
   */
  ValueTable getValuesFromDotOperandLayoutStruct(Value value, int n0, int n1,
                                                 int kWidth, Type type) const {
    auto elems = unpackLLElements(loc, value, rewriter);
    ValueTable vals;
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < n1; j++) {
        Type elemTy = typeConverter->convertType(type);
        Type ty = vec_ty(elemTy, kWidth);
        Value rawElems = undef(ty);
        for (int k = 0; k < kWidth; ++k) {
          rawElems = insert_element(
              ty, rawElems, elems[kWidth * (n1 * i + j) + k], i32_val(k));
        }

        Value convertedElems;
        if (type.isF32()) {
          convertedElems = extract_element(type, rawElems, i32_val(0));
        } else if (type.getIntOrFloatBitWidth() == 8) {
          if (kWidth == 4)
            convertedElems = bitcast(rawElems, i32_ty);
          if (kWidth == 8)
            convertedElems = bitcast(rawElems, i64_ty);
        } else {
          assert(type.isBF16() || type.isF16());
          convertedElems = rawElems;
        }
        vals[{i, j}] = convertedElems;
      }
    }
    return vals;
  }
};

} // namespace

namespace mlir::triton::AMD {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return tensor.getType().cast<RankedTensorType>();
  };

  assert(rankedTType(op.getA()).getEncoding().isa<DotOperandEncodingAttr>() &&
         rankedTType(op.getB()).getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(cTensorTy.getEncoding().isa<AMDMfmaEncodingAttr>() &&
         "Currently, we only support $c with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mfmaLayout = op.getResult()
                        .getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<AMDMfmaEncodingAttr>();

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}
} // namespace mlir::triton::AMD

#endif // ifdef USE_ROCM
