#include "../TritonGPUToLLVMBase.h"
#include "../Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/Support/ErrorHandling.h"

using ValueTable = std::map<std::pair<int, int>, Value>;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::XPU::TritonGPUToLLVMTypeConverter;

namespace {

// Data loader for DPAS instruction.
template <unsigned opIdx> class DpasMatmulLoader {
public:
  DpasMatmulLoader(DpasEncodingAttr dpasLayout, RankedTensorType tensorTy,
                   unsigned warpsPerTile, ArrayRef<Value> smemStrides,
                   SmallVector<int64_t> instrShape,
                   ConversionPatternRewriter &rewriter,
                   TritonGPUToLLVMTypeConverter *typeConverter, Location loc)
      : dpasLayout(dpasLayout), tensorTy(tensorTy), smemStrides(smemStrides),
        rewriter(rewriter), loc(loc) {
    static_assert(opIdx == 0 || opIdx == 1);

    unsigned kDim = (opIdx == 0) ? 1 : 0;
    repKDimStride = mul(i32_val(instrShape[kDim]), smemStrides[kDim]);
    repNonKDimStride = mul(i32_val(instrShape[kDim ^ 1] * warpsPerTile),
                           smemStrides[kDim ^ 1]);
    warpMatStride = mul(i32_val(instrShape[kDim ^ 1]), smemStrides[kDim ^ 1]);

    unsigned opsPerChannel =
        dpasLayout.getOpsPerChannel(tensorTy.getElementType());
    unsigned threadsPerWarp = getThreadsPerWarp();

    int rowsPerWarp =
        threadsPerWarp / ((opIdx == 0) ? dpasLayout.getSystolicDepth()
                                       : dpasLayout.getExecutionSize());
    numPtrs = (dpasLayout.getRepeatCount() / rowsPerWarp) * opsPerChannel;
  }

  SmallVector<Value> computeOffsets(Value warpId, Value laneId,
                                    Value cSwizzleOffset) {
    return computeLdsMatOffs(warpId, laneId, cSwizzleOffset);
  }

  int getNumPtrs() const { return numPtrs; }

  // Compute matrix load offsets.
  SmallVector<Value> computeLdsMatOffs(Value warpOff, Value lane,
                                       Value cSwizzleOffset);
  // Load the matrix value.
  Value loadMatrix(int repOuter, int repInner, const ArrayRef<Value> ptrs,
                   LLVM::LLVMStructType structTy, Type smemTy,
                   Value cSwizzleOffset) const;

private:
  unsigned getThreadsPerWarp() const {
    return product<unsigned>(triton::gpu::getThreadsPerWarp(dpasLayout));
  }

  DpasEncodingAttr dpasLayout;
  RankedTensorType tensorTy;

  SmallVector<Value> smemStrides;
  Value repNonKDimStride;
  Value repKDimStride;

  // Stride in number of matrices to increment on non-k dim across warps
  Value warpMatStride;
  int numPtrs;

  ConversionPatternRewriter &rewriter;
  Location loc;
};

template <unsigned opIdx>
SmallVector<Value>
DpasMatmulLoader<opIdx>::computeLdsMatOffs(Value warpId, Value laneId,
                                           Value cSwizzleOffset) {
  SmallVector<Value> offs(numPtrs);

  unsigned systolicDepth = dpasLayout.getSystolicDepth();
  unsigned repeatCount = dpasLayout.getRepeatCount();
  unsigned executionSize = dpasLayout.getExecutionSize();
  unsigned opsPerChannel =
      dpasLayout.getOpsPerChannel(tensorTy.getElementType());
  unsigned threadsPerWarp = getThreadsPerWarp();

  Value laneRowIndex, laneColIndex;
  unsigned repRowsPerInst, rowsPerWarp;
  switch (opIdx) {
  case 0: {
    rowsPerWarp = threadsPerWarp / systolicDepth;
    repRowsPerInst = repeatCount / rowsPerWarp;
    laneRowIndex = udiv(laneId, i32_val(systolicDepth));
    laneColIndex = urem(laneId, i32_val(systolicDepth));
    laneColIndex = mul(laneColIndex, i32_val(opsPerChannel));
  } break;
  case 1: {
    rowsPerWarp = threadsPerWarp / executionSize;
    repRowsPerInst = systolicDepth / rowsPerWarp;
    rowsPerWarp = rowsPerWarp * opsPerChannel;
    laneRowIndex = udiv(laneId, i32_val(executionSize));
    laneRowIndex = mul(laneRowIndex, i32_val(opsPerChannel));
    laneColIndex = urem(laneId, i32_val(executionSize));
  } break;
  }

  // outer index offset
  Value iOff = mul(warpId, warpMatStride);

  SharedEncodingAttr sharedLayout =
      tensorTy.getEncoding().cast<SharedEncodingAttr>();
  const int perPhase = sharedLayout.getPerPhase();
  const int maxPhase = sharedLayout.getMaxPhase();
  const int vec = sharedLayout.getVec();

  unsigned index = 0;
  Value rowsPerWarpVal = i32_val(rowsPerWarp);
  Value zeroVal = i32_val(0);
  Value perPhaseVal = i32_val(perPhase);
  Value maxPhaseVal = i32_val(maxPhase);
  Value vecVal = i32_val(vec);

  for (int rep = 0; rep < repRowsPerInst; ++rep) {
    Value repRowIndex = mul(i32_val(rep), rowsPerWarpVal);
    for (unsigned opsIdx = 0; opsIdx < opsPerChannel; ++opsIdx) {
      // inner index base
      Value jBase = laneColIndex;
      // outer index base
      Value iBase = add(repRowIndex, laneRowIndex);
      switch (opIdx) {
      case 0:
        jBase = add(jBase, i32_val(opsIdx));
        break;
      case 1:
        iBase = add(iBase, i32_val(opsIdx));
        break;
      }

      // inner index offset
      Value jOff = zeroVal;
      // swizzle: col_swizzled = (col / vec) ^ phase * vec
      Value phase = urem(udiv(iBase, perPhaseVal), maxPhaseVal);
      jOff = add(jOff, udiv(cSwizzleOffset, vecVal));
      jOff = mul(xor_(jOff, phase), vecVal);

      Value i = add(mul(iBase, smemStrides[0]), iOff);
      Value j = add(mul(jBase, smemStrides[1]), jOff);

      offs[index++] = add(i, j);
    }
  }

  return offs;
}

template <unsigned opIdx>
Value DpasMatmulLoader<opIdx>::loadMatrix(int repOuter, int repInner,
                                          const ArrayRef<Value> ptrs,
                                          LLVM::LLVMStructType structTy,
                                          Type smemTy,
                                          Value cSwizzleOffset) const {
  Type elemTy = structTy.getBody()[0];
  assert(
      llvm::any_of(structTy.getBody(), [&](Type ty) { return ty == elemTy; }) &&
      "The struct should have the same element types.");

  Value offsetOuter = mul(i32_val(repOuter), repNonKDimStride);
  Value offsetInner = mul(i32_val(repInner), repKDimStride);
  Value offset = add(offsetOuter, offsetInner);

  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
  size_t elemNum = structTy.getBody().size();
  for (int i = 0; i < elemNum; i++) {
    Value readPtr =
        gep(ptr_ty(rewriter.getContext(), 3), smemTy, ptrs[i], offset);
    Value val = rewriter.create<LLVM::LoadOp>(loc, elemTy, readPtr);
    llvmStruct = insert_val(structTy, llvmStruct, val, i);
  }

  return llvmStruct;
}

Value composeValuesToDotOperandLayoutStruct(
    const ValueTable &vals, int n0, int n1,
    TritonGPUToLLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter) {
  std::vector<Value> elems;
  for (int m = 0; m < n0; ++m) {
    for (int k = 0; k < n1; ++k) {
      Value matVal = vals.at({m, k});
      auto matType = matVal.getType().cast<LLVM::LLVMStructType>();
      Type valTy = matType.getBody()[0];
      for (int i = 0; i < matType.getBody().size(); ++i) {
        auto val = extract_val(valTy, matVal, i);
        elems.push_back(val);
      }
    }
  }
  assert(!elems.empty() && "Expecting non-empty vector");

  Type elemTy = elems[0].getType();
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(elems.size(), elemTy));

  return typeConverter->packLLElements(loc, elems, rewriter, structTy);
}

Type getSharedMemTy(Type argType) {
  MLIRContext *ctx = argType.getContext();
  if (argType.isF16())
    return type::f16Ty(ctx);
  else if (argType.isBF16())
    return type::i16Ty(ctx);
  else if (argType.isF32())
    return type::f32Ty(ctx);
  else if (argType.getIntOrFloatBitWidth() == 8)
    return type::i8Ty(ctx);
  else
    llvm::report_fatal_error("mma16816 data type not supported");
}

template <unsigned opIdx>
std::function<void(int, int)>
getLoadMatrixFn(Value tensor, const SharedMemoryObject &smemObj,
                DpasEncodingAttr dpasLayout, unsigned warpsPerTile,
                SmallVector<int64_t> instrShape, Value warpId,
                Value outerWarpDim, Value laneId, ValueTable &vals,
                TritonGPUToLLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, Location loc) {
  static_assert(opIdx == 0 || opIdx == 1);

  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  Type eltTy = tensorTy.getElementType();

  auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();
  ArrayRef<unsigned> order = sharedLayout.getOrder();

  // (a, b) is the coordinate.
  auto load = [=, &rewriter, &vals](int a, int b) {
    DpasMatmulLoader<opIdx> loader(dpasLayout, tensorTy, warpsPerTile,
                                   smemObj.strides, instrShape, rewriter,
                                   typeConverter, loc);

    // Offset of a slice within the original tensor in shared memory.
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    SmallVector<Value> offs =
        loader.computeOffsets(outerWarpDim, laneId, cSwizzleOffset);

    // Initialize pointers.
    const int numPtrs = loader.getNumPtrs();
    SmallVector<Value> ptrs(numPtrs);

    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
    Type smemTy = getSharedMemTy(eltTy);
    for (int i = 0; i < numPtrs; ++i)
      ptrs[i] =
          gep(ptr_ty(rewriter.getContext(), 3), smemTy, smemBase, offs[i]);

    // Load from shared memory.
    int64_t totalElem = product<int64_t>(instrShape);
    unsigned threadsPerWarp = product<unsigned>(getThreadsPerWarp(dpasLayout));
    auto matTy = LLVM::LLVMStructType::getLiteral(
        eltTy.getContext(),
        SmallVector<Type>(totalElem / threadsPerWarp, eltTy));

    vals[{a, b}] = loader.loadMatrix(a, b, ptrs, matTy, smemTy, cSwizzleOffset);
  };

  return load;
}

template <unsigned opIdx>
Value loadOperand(ConversionPatternRewriter &rewriter, Location loc,
                  Value threadId, DotOperandEncodingAttr encoding,
                  TritonGPUToLLVMTypeConverter *typeConverter, Value tensor,
                  const SharedMemoryObject &smemObj) {
  static_assert(opIdx == 0 || opIdx == 1);

  auto dpasLayout = encoding.getParent().cast<DpasEncodingAttr>();
  const SmallVector<unsigned> warpsPerCTA = dpasLayout.getWarpsPerCTA();

  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  const ArrayRef<int64_t> tensorShape = tensorTy.getShape();
  auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();
  const ArrayRef<unsigned> order = sharedLayout.getOrder();

  Type elemTy = tensorTy.getElementType();
  SmallVector<int64_t> elemsPerInstr =
      encoding.getDPASElemsPerInstr(elemTy.getIntOrFloatBitWidth());
  SmallVector<int64_t> numReps = encoding.getDPASRep(tensorShape, elemTy);

  Value warpSize = i32_val(triton::gpu::getWarpSize(dpasLayout));
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);

  SmallVector<Value> multiDimWarpId =
      mlir::LLVM::delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  double ceilRes =
      ceil(static_cast<double>(tensorShape[opIdx]) / elemsPerInstr[opIdx]);
  Value outerWarpDim = urem(multiDimWarpId[opIdx], i32_val(ceilRes));
  int warpsPerTile = std::min<int>(warpsPerCTA[opIdx], ceilRes);

  // Get the function to use to load the operand.
  ValueTable vals;
  std::function<void(int, int)> loadFn = getLoadMatrixFn<opIdx>(
      tensor, smemObj, dpasLayout, warpsPerTile, elemsPerInstr, warpId,
      outerWarpDim, laneId, vals, typeConverter, rewriter, loc);

  // Load the operand.
  int64_t numRepOuter = numReps[opIdx];
  int64_t numRepK = numReps[(opIdx == 0) ? 1 : 0];

  for (int m = 0; m < numRepOuter; ++m)
    for (int k = 0; k < numRepK; ++k)
      loadFn(m, k);

  // Format the values into an LLVM::Struct.
  return composeValuesToDotOperandLayoutStruct(vals, numRepOuter, numRepK,
                                               typeConverter, loc, rewriter);
}

} // namespace

namespace XPU {
namespace SharedToDotOperandDPAS {

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter,
                    Value threadId) {
  switch (opIdx) {
  case 0:
    return loadOperand<0>(rewriter, loc, threadId, encoding, typeConverter,
                          tensor, smemObj);
  case 1:
    return loadOperand<1>(rewriter, loc, threadId, encoding, typeConverter,
                          tensor, smemObj);
  default:
    llvm_unreachable("unexpected operand idx");
  }
  return Value();
}
} // namespace SharedToDotOperandDPAS
} // namespace XPU
