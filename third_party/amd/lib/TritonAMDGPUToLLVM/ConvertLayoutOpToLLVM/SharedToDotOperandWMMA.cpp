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
#include "SharedToDotOperandHelper.h"
#include "Utility.h"

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace SharedToDotOperandWMMA {

/**
 * @brief This function maps particular load of wmma dot operand to element
 * indexes(row, col)
 *
 * Whole tensor is broken into "blocks" of warps along "non-K" axis.
 * One block could be processed by multiple warps.
 * One warp works on a piece of tensor size elemsPerInstr[0] x K.
 * Each of these pieces is broken into "tiles" of size elemsPerInstr[0] x
 * elemsPerInstr[1].
 *
 * Total offset of element is a sum of following values:
 * 1. Offset of warp block in tensor
 * 2. Offset of warp inside one warp block
 * 3. Offset of tile in one warp
 * 4. Offset of one lane data in a tile
 * 5. Offset of particular element of tensor processed by one lane
 *
 * This function computes these offsets for axes independently
 *
 * @param rewriter
 * @param loc
 * @param elemsPerInstr operand tile shape consumed by one WMMA instruction
 * @param warpId id component of 2d warp grid along non-K axis
 * @param laneId lane id in warp [0..63]
 * @param numOfElems number of elements accessed by thread per repetition
 * @param reps number of instructions repetition to fully cover dot operand
 * @param smemStrides strides in LDS tensor
 * @param loadVecSize number of elements loaded by one operation
 * @param iNonKDim non-K dimension of dot operand
 * @return vector (i-th element corresponds to i-th load instruction) of
 * 2-element vectors(tensor row and col).
 */
llvm::SmallVector<llvm::SmallVector<Value>> computeTensorElemMappingInBlock(
    ConversionPatternRewriter &rewriter, Location loc,
    const ArrayRef<int64_t> &elemsPerInstr, Value warpId, Value laneId,
    int numOfElems, ArrayRef<int64_t> reps, ArrayRef<Value> smemOffsets,
    int loadVecSize, unsigned iNonKDim, [[maybe_unused]] unsigned iKDim) {
  auto numK = reps[1];
  const int loadsPerThread = numOfElems / loadVecSize;
  llvm::SmallVector<llvm::SmallVector<Value>> mapping(numK * loadsPerThread);

  Value _0 = i32_val(0);
  Value nonKDim = i32_val(iNonKDim);
  Value warpVOffset = mul(warpId, i32_val(elemsPerInstr[0]));

  for (int tile = 0; tile < numK; ++tile) {
    Value tileVOffset = _0;
    Value tileHOffset = i32_val(tile * elemsPerInstr[1]);

    Value laneVOffset = laneId;
    Value laneHOffset = _0;

    for (int loadId = 0; loadId < loadsPerThread; ++loadId) {
      Value elemVOffset = _0;
      Value elemHOffset = i32_val(loadId * loadVecSize);

      Value sliceVOffset =
          add(add(add(tileVOffset, laneVOffset), elemVOffset), warpVOffset);
      Value sliceHOffset = add(add(tileHOffset, laneHOffset), elemHOffset);

      Value row = add(sliceVOffset, smemOffsets[0]);
      Value col = add(sliceHOffset, smemOffsets[1]);

      mapping[loadsPerThread * tile + loadId] = {row, col};
    }
  }

  return mapping;
}

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread) {
  assert((opIdx == 0 || opIdx == 1) && "unexpected operand idx");
  int kDimIdx = opIdx == 0 ? 1 : 0;
  int nonKDimIdx = opIdx == 0 ? 0 : 1;

  auto wmmaLayout = encoding.getParent().cast<AMDWmmaEncodingAttr>();
  auto nonKDim = wmmaLayout.getMNKDimPerWMMAInstr()[nonKDimIdx];
  assert(nonKDim == 16);
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<MemDescType>();
  ArrayRef<int64_t> shape = aTensorTy.getShape();
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  auto elemTy = aTensorTy.getElementType();
  int kWidth = encoding.getKWidth();
  auto elemsPerInstr = wmmaLayout.getWMMAElemsPerInstrForOperands();
  auto wmmaInstrK = elemsPerInstr[kDimIdx];

  auto numReps = wmmaLayout.getWMMARepForOperands(shape, elemTy, kWidth, opIdx);
  auto numRepNonK = numReps[nonKDimIdx];
  auto numRepK = numReps[kDimIdx];

  unsigned iWarpSize = triton::gpu::getWarpSize(wmmaLayout);
  unsigned iNumLanes = iWarpSize / 2;
  assert(iWarpSize == 32);
  Value warpSize = i32_val(iWarpSize);
  Value numLanes = i32_val(iNumLanes);
  Value linearWarpId = udiv(thread, warpSize);
  Value lane = urem(thread, numLanes); // share elem between two threads

  unsigned numElemsPerThreadPerRep =
      wmmaLayout.getMNKDimPerWMMAInstr()[kDimIdx];

  Value warp = udiv(thread, warpSize);
  unsigned int maxNumWarps = shape[nonKDimIdx] / elemsPerInstr[nonKDimIdx];
  int warpsPerBlockNonK = std::min(warpsPerCTA[nonKDimIdx], maxNumWarps);
  elemTy = typeConverter->convertType(elemTy);

  SmallVector<Value> loadedValues;
  SmallVector<Value> offsets;
  Value smemBase;
  Value spatialWarpId = AMD::getWarpIdInBlock(
      rewriter, loc, linearWarpId, warpsPerCTA, elemsPerInstr[0],
      shape[nonKDimIdx], nonKDimIdx, triton::gpu::getOrder(wmmaLayout));
  if (opIdx == 0) {
    offsets = AMD::computeOffsetsAType(
        rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
        spatialWarpId, lane, warpsPerBlockNonK, numElemsPerThreadPerRep,
        numReps, smemObj, sharedLayout, nonKDim, wmmaInstrK);
  } else {
    assert(opIdx == 1);
    offsets = AMD::computeOffsetsBType(
        rewriter, loc, computeTensorElemMappingInBlock, elemsPerInstr,
        spatialWarpId, lane, warpsPerBlockNonK, numElemsPerThreadPerRep,
        numReps, smemObj, sharedLayout, nonKDim, wmmaInstrK);
  }
  smemBase = AMD::computeBasePtr(rewriter, loc, smemObj);

  Type resElemTy = typeConverter->convertType(elemTy);
  Type smemPtrTy = ptr_ty(rewriter.getContext(), 3);

  int loadsPerThread = offsets.size() / (numRepNonK * numRepK);
  int elemsPerLoad = numElemsPerThreadPerRep / loadsPerThread;
  assert(numElemsPerThreadPerRep % loadsPerThread == 0);
  for (int nonK = 0; nonK < numRepNonK; ++nonK) {
    for (int k = 0; k < numRepK; ++k) {
      auto vecTy = vec_ty(resElemTy, numElemsPerThreadPerRep);
      Value valVec = undef(vecTy);
      for (unsigned loadId = 0; loadId < loadsPerThread; ++loadId) {
        auto loadVecTy = vec_ty(elemTy, elemsPerLoad);
        Value loadOffset = offsets[nonK * loadsPerThread * numRepK +
                                   k * loadsPerThread + loadId];
        Value loadAddress = gep(smemPtrTy, elemTy, smemBase, loadOffset);
        Value loadedValue = load(loadVecTy, loadAddress);
        for (int elemId = 0; elemId < elemsPerLoad; ++elemId) {
          Value elemVal = extract_element(elemTy, loadedValue, i32_val(elemId));
          loadedValues.push_back(elemVal);
        }
      }
    }
  }

  MLIRContext *ctx = wmmaLayout.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(loadedValues.size(), loadedValues[0].getType()));
  auto result =
      packLLElements(loc, typeConverter, loadedValues, rewriter, structTy);
  return result;
}

} // namespace SharedToDotOperandWMMA

#endif // ifdef USE_ROCM
