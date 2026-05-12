#include "intel/include/Dialect/TritonIntelGPU/Transforms/BlockIOUtils.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

namespace mlir::triton::gpu::intel {

template <typename T> static T product(const std::vector<T> &vec) {
  return std::accumulate(vec.begin(), vec.end(), static_cast<T>(1),
                         std::multiplies<T>());
}

// Return the tileHeight, tileWidth, numElemPerPackedVal, vBlocks, row Dim and
// column Dim.
template <bool isLoad>
BlockIOTileSizeInfo
getBlockIOTileSize(const LinearLayout &ll, unsigned memContiguousDim,
                   unsigned elemSizeInBits, AxisInfo *maskAxisInfo,
                   bool oneMatrixPerLoadForBT) {

  if (elemSizeInBits > 64)
    return BlockIOTileSizeInfo::unknown();

  const size_t rank = ll.getOutDims().size();
  std::vector<unsigned> tileShape(rank, 1);

  const LinearLayout::BasesT &bases = ll.getBases();
  auto getBase = [&](const std::string &inDim) {
    for (const auto &base : bases) {
      StringAttr attr = base.first;
      if (attr.getValue().compare(inDim) == 0)
        return base.second;
    }
    llvm_unreachable(("Could not find the input dim:" + inDim +
                      ", on the ll:" + ll.toString())
                         .c_str());
  };

  auto validateBase = [](const std::vector<int> &vec) {
    // Check there is only one element that is greater than 0
    return llvm::count_if(vec, [](int x) { return x > 0; }) == 1;
  };

  auto getFirstNonZeroDim = [](const std::vector<int> &vec) {
    auto it = llvm::find_if(vec, [](int x) { return x > 0; });
    return (it != vec.end()) ? std::distance(vec.begin(), it) : -1;
  };

  using BaseType = LinearLayout::BasesT::value_type::second_type;
  const BaseType &basesOfLane = getBase("lane");

  if (!validateBase(basesOfLane[0]))
    return BlockIOTileSizeInfo::unknown();

  // The IGC scalar backend always vectorize the non-uniform value in row
  // major. So the first non-zero dimension of the lane base is used as column
  // dim for block io.
  int fastChangeDim = getFirstNonZeroDim(basesOfLane[0]);

  // The mask constancy has to be power of 2 for block IO.
  if (maskAxisInfo &&
      !llvm::isPowerOf2_64(maskAxisInfo->getConstancy(fastChangeDim)))
    return BlockIOTileSizeInfo::unknown();

  unsigned maskConstancyFastChangeDimLimit =
      maskAxisInfo ? maskAxisInfo->getConstancy(fastChangeDim)
                   : std::numeric_limits<unsigned>::max();
  bool transpose = fastChangeDim != memContiguousDim;

  // Walk thru the register bases in incremental order to get the register
  // index for the packed value for block io.
  // TODO: improve the register packing order to support swizzled linear
  // layout.
  const BaseType &basesOfRegister = getBase("register");
  int numElemPerPackedVal = 1;
  constexpr unsigned MAX_BITS_NORMAL = 64;
  // Hardware supports the d64 for transposing. But for packing
  // transpose, we'd prefer smaller d32 type cause hardware could
  // transpose more to reduce the number of mov operation in register.
  constexpr unsigned MAX_BITS_TRANSPOSE = 32;
  constexpr unsigned MAX_BITS_WIDTH_NORMAL = 64 * 8;       // 64 bytes.
  constexpr unsigned MAX_BITS_WIDTH_TRANSPOSE = 8 * 4 * 8; // 8xd32. (and 4xd64)
  constexpr unsigned TRANSPOSE_LOAD_D64_HEIGHT = 8;
  constexpr unsigned MAX_TILE_HEIGHT_STORE = 8;
  constexpr unsigned MAX_TILE_HEIGHT_LOAD = 32;
  unsigned MAX_TILE_HEIGHT;
  if constexpr (isLoad) {
    MAX_TILE_HEIGHT = (transpose && elemSizeInBits == 64)
                          ? TRANSPOSE_LOAD_D64_HEIGHT
                          : MAX_TILE_HEIGHT_LOAD;
  } else {
    MAX_TILE_HEIGHT = MAX_TILE_HEIGHT_STORE;
  }
  unsigned MAX_BITS_WIDTH =
      transpose ? MAX_BITS_WIDTH_TRANSPOSE : MAX_BITS_WIDTH_NORMAL;

  unsigned maxElemPackedVal = mlir::ceil<unsigned>(
      transpose ? MAX_BITS_TRANSPOSE : MAX_BITS_NORMAL, elemSizeInBits);
  SetVector<unsigned> regPackBases;
  for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
       ++regBaseIter) {
    if (numElemPerPackedVal >= maxElemPackedVal) {
      // Reached the maximum number of elements per packed value.
      break;
    }
    const std::vector<int> &base = basesOfRegister[regBaseIter];
    if (!validateBase(base))
      continue; // Skip as the register can not be trivial packed.
    int dim = getFirstNonZeroDim(base);
    if (memContiguousDim == dim) {
      if (tileShape[dim] != base[dim])
        continue; // Skip the register not in dense tile.
      // The value can be loaded as packed value.
      tileShape[dim] <<= 1;
      numElemPerPackedVal <<= 1;
      regPackBases.insert(1 << regBaseIter);
    }
  }

  // For the transpose case, we have to pack the elements to d32.
  if (transpose && numElemPerPackedVal != maxElemPackedVal)
    return BlockIOTileSizeInfo::unknown();

  // We already get the basic tile shape in packing values.
  // To increase the tile shape along each lane dimension.
  for (const std::vector<int> &base : basesOfLane) {
    if (!validateBase(base))
      break; // break if the lane bases are not trivial.
    int dim = getFirstNonZeroDim(base);
    if (tileShape[dim] != base[dim]) {
      // TODO: Check whether we can add an VNNI pack to make a larger tile.
      break;
    }
    tileShape[dim] <<= 1;
  }

  const unsigned numLanes = 1 << basesOfLane.size();
  // The slice of a name is not distributed densely across the lane. It is not
  // supported by block io.
  if ((product<unsigned>(tileShape) / numElemPerPackedVal) != numLanes)
    return BlockIOTileSizeInfo::unknown();

  unsigned sliceRank = 0;
  int rowDim = -1;
  for (size_t i = 0; i < rank; ++i) {
    if (tileShape[i] > 1) {
      sliceRank++;
      // if the slice has more than one non-zero size. Chose the
      // non-fast change dim as the row dim.
      if (i != fastChangeDim)
        rowDim = i;
    }
  }

  // The block IO only supports 2D shape.
  if (sliceRank > 2)
    return BlockIOTileSizeInfo::unknown();

  // When transposed, width and height constraints swap between fastChangeDim
  // and rowDim: fastChangeDim is constrained by block io tile width in the
  // non-transposed case and by block io tile height in the transposed case,
  // while rowDim uses the opposite limits.

  // The tile shape sizes should not exceed the hardware limit.
  unsigned fastChangeDimLimit =
      !transpose ? MAX_BITS_WIDTH / elemSizeInBits : MAX_TILE_HEIGHT;
  unsigned rowDimLimit =
      !transpose ? MAX_TILE_HEIGHT : MAX_BITS_WIDTH / elemSizeInBits;

  // The tile shape sizes should not exceed the mask constancy limit.
  fastChangeDimLimit =
      std::min(fastChangeDimLimit, maskConstancyFastChangeDimLimit);

  unsigned maskConstancyRowDimLimit = std::numeric_limits<unsigned>::max();
  if (rowDim >= 0) {
    // The mask constancy has to be power of 2 for block IO.
    if (maskAxisInfo &&
        !llvm::isPowerOf2_64(maskAxisInfo->getConstancy(rowDim)))
      return BlockIOTileSizeInfo::unknown();
    if (maskAxisInfo)
      maskConstancyRowDimLimit = maskAxisInfo->getConstancy(rowDim);
  }

  rowDimLimit = std::min(rowDimLimit, maskConstancyRowDimLimit);

  if (tileShape[fastChangeDim] > fastChangeDimLimit)
    return BlockIOTileSizeInfo::unknown();

  if (rowDim >= 0 && tileShape[rowDim] > rowDimLimit)
    return BlockIOTileSizeInfo::unknown();

  if (!oneMatrixPerLoadForBT && transpose &&
      tileShape[memContiguousDim] == numElemPerPackedVal) {
    // Increase the tile shape along the col dimension for transpose case.
    for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
         ++regBaseIter) {
      if (regPackBases.contains(1 << regBaseIter))
        continue; // Skip the register already packed.
      const std::vector<int> &base = basesOfRegister[regBaseIter];
      if (!validateBase(base))
        continue; // Skip as the bases are not trivial.
      int dim = getFirstNonZeroDim(base);
      if (dim != fastChangeDim ||
          tileShape[fastChangeDim] != base[fastChangeDim])
        continue; // Skip the register not mapped to the row dim.
      if ((tileShape[fastChangeDim] << 1) > MAX_TILE_HEIGHT)
        break; // The col dim is the height.
      if ((tileShape[fastChangeDim] << 1) > maskConstancyFastChangeDimLimit)
        break; // Should not exceed the mask constancy limit.
      tileShape[fastChangeDim] <<= 1;
      regPackBases.insert(1 << regBaseIter);
    }
  }

  // Note: we only walk thru register packing order by increasing the
  // tileHeight and vBlocks for simplicity. This may cause low efficiency in
  // block store for some cases because the block store doesn't support
  // vBlocks > 1. Illustration of the tile shape and register packing:
  // clang-format off
  //                 vBlocks=2
  //                     ^
  //          ┌───────────────────┐
  //     tileWidth=16        tileWidth=16
  //           ^                   ^
  // ┌───────────────────┬───────────────────┐
  // lane 0 1 2 .....  15 lane 0 1 2 .....  15
  // ┌────┬────┬────┬────┬────┬────┬────┬────┐
  // │R0  │    │    │    │R1  │    │    │    │
  // │    │    │    │    │    │    │    │    │
  // ├────┼────┼────┼────┼────┼────┼────┼────┤
  // │R2  │    │    │    │R3  │    │    │    │
  // │    │    │    │    │    │    │    │    │
  // └────┴────┴────┴────┴────┴────┴────┴────┘
  // We will pack the R0 and R2 as the first matrix. R1 and R3 as the second matrix with vBlocks=2 for 2 matrixes.
  // But the tile shape following maybe more efficient for block store because block store only supports vBlocks=1.
  //               tileWidth=32
  //                     ^
  // ┌───────────────────┬───────────────────┐
  // lane 0 1 2 .....  15 lane 0 1 2 .....  15
  // ┌────┬────┬────┬────┬────┬────┬────┬────┐
  // │R0  │    │    │    │R1  │    │    │    │
  // │    │    │    │    │    │    │    │    │
  // ├────┼────┼────┼────┼────┼────┼────┼────┤
  // │R2  │    │    │    │R3  │    │    │    │
  // │    │    │    │    │    │    │    │    │
  // └────┴────┴────┴────┴────┴────┴────┴────┘
  // clang-format on

  // Increase the tile shape along the row dimension. (Increase the
  // tileHeight.)
  for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
       ++regBaseIter) {
    if (regPackBases.contains(1 << regBaseIter))
      continue; // Skip the register already packed.
    const std::vector<int> &base = basesOfRegister[regBaseIter];
    if (!validateBase(base))
      continue; // Skip as the bases are not trivial.
    int dim = getFirstNonZeroDim(base);
    if (rowDim < 0 && dim != fastChangeDim) {
      rowDim = dim;
      // The mask constancy has to be power of 2 for block IO.
      if (maskAxisInfo &&
          !llvm::isPowerOf2_64(maskAxisInfo->getConstancy(rowDim)))
        return BlockIOTileSizeInfo::unknown();
      if (maskAxisInfo)
        maskConstancyRowDimLimit = maskAxisInfo->getConstancy(rowDim);
    }
    if (dim != rowDim || tileShape[rowDim] != base[rowDim])
      continue; // Skip the register not mapped to the row dim.
    if (!transpose) {
      if ((tileShape[rowDim] << 1) > MAX_TILE_HEIGHT)
        break; // If the tile height is limited, we stop here.
    } else {
      if (((tileShape[rowDim] << 1) * elemSizeInBits) > MAX_BITS_WIDTH)
        break; // The row is the width.
    }
    // The size should not exceed the mask constancy limit.
    if ((tileShape[rowDim] << 1) > maskConstancyRowDimLimit)
      break;
    tileShape[rowDim] <<= 1;
    regPackBases.insert(1 << regBaseIter);
  }

  if (transpose) {
    // For transpose, the row dim has to be the memory contiguous dim.
    // If rowDim is determined (>= 0) and it is not memory contiguous dim,
    // reject.
    if (rowDim >= 0 && rowDim != memContiguousDim)
      return BlockIOTileSizeInfo::unknown();
  }

  if (rowDim < 0) {
    int lastDim = static_cast<int>(rank - 1);
    rowDim = (fastChangeDim != lastDim) ? lastDim : lastDim - 1;
  }

  if (transpose && elemSizeInBits == 64) {
    // D64 transpose only supports 8 rows.
    if (tileShape[fastChangeDim] != TRANSPOSE_LOAD_D64_HEIGHT)
      return BlockIOTileSizeInfo::unknown();
  }

  unsigned vBlocks = 1;
  if (!transpose) {
    // Increase the tile shape along the column dimension. (Increase the
    // vBlocks.)
    for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
         ++regBaseIter) {
      if (regPackBases.contains(1 << regBaseIter))
        continue; // Skip the register already packed.
      const std::vector<int> &base = basesOfRegister[regBaseIter];
      if (!validateBase(base))
        continue; // Skip as the bases are not trivial.
      int dim = getFirstNonZeroDim(base);
      if (dim != fastChangeDim || (tileShape[dim] * vBlocks) != base[dim])
        continue;
      if ((tileShape[fastChangeDim] * (vBlocks << 1)) >
          maskConstancyFastChangeDimLimit)
        break; // Should not exceed the mask constancy limit.
      vBlocks <<= 1;
      regPackBases.insert(1 << regBaseIter);
    }
  }
  for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
       ++regBaseIter) {
    if (regPackBases.contains(1 << regBaseIter))
      continue; // Skip the register already packed.
    // insert the remaining register base.
    regPackBases.insert(1 << regBaseIter);
  }
  int tileHeight = tileShape[transpose ? fastChangeDim : rowDim];
  int tileWidth =
      tileShape[transpose ? rowDim : fastChangeDim] / numElemPerPackedVal;

  // Cap vBlocks for loads based on HW constraints.
  if constexpr (isLoad) {
    constexpr int MAX_WIDTH_BYTES = 64;
    unsigned packedElemSizeInBits = elemSizeInBits * numElemPerPackedVal;
    unsigned totalBytesPerRowPerMatrix = tileWidth * packedElemSizeInBits / 8;
    if (totalBytesPerRowPerMatrix > 0) {
      vBlocks =
          std::min(vBlocks, static_cast<unsigned>(MAX_WIDTH_BYTES /
                                                  totalBytesPerRowPerMatrix));
    }
    vBlocks = std::min(vBlocks, 4u);
    constexpr unsigned GRF_SIZE = 64;
    if (tileHeight * tileWidth * packedElemSizeInBits / 8 < GRF_SIZE)
      vBlocks = 1;
  }

  return BlockIOTileSizeInfo(tileHeight, tileWidth, numElemPerPackedVal,
                             vBlocks, rowDim, fastChangeDim, transpose,
                             std::move(regPackBases));
}

// Explicit instantiations.
template BlockIOTileSizeInfo getBlockIOTileSize<true>(const LinearLayout &,
                                                      unsigned, unsigned,
                                                      AxisInfo *, bool);
template BlockIOTileSizeInfo getBlockIOTileSize<false>(const LinearLayout &,
                                                       unsigned, unsigned,
                                                       AxisInfo *, bool);

bool check2DBlockAddressPayloadRestriction(unsigned packedElemSizeInBits,
                                           unsigned tileWidth) {
  // Return false if tile width is not supported by HW.
  // Note: Tile width is not changeable.
  switch (packedElemSizeInBits) {
  case 8:
    if (tileWidth < 4 || tileWidth > 64)
      return false;
    break;
  case 16:
    if (tileWidth < 2 || tileWidth > 32)
      return false;
    break;
  case 32:
    if (tileWidth > 16)
      return false;
    break;
  case 64:
    if (tileWidth > 8)
      return false;
    break;
  default:
    // invalid element type for 2D block io.
    return false;
  }
  return true;
}

DpasEncodingAttr::OpIdx getOpIdx(RankedTensorType tensorTy) {
  if (hasDpasEncoding(tensorTy))
    return DpasEncodingAttr::OpIdx::OperandC;
  assert(hasDotDpasEncoding(tensorTy) && "Expecting dot layout");
  auto dotLayout =
      cast<triton::gpu::DotOperandEncodingAttr>(tensorTy.getEncoding());
  return static_cast<DpasEncodingAttr::OpIdx>(dotLayout.getOpIdx());
}

DpasEncodingAttr getDpasLayout(RankedTensorType tensorTy) {
  Attribute encoding = tensorTy.getEncoding();
  return cast<DpasEncodingAttr>(
      hasDpasEncoding(tensorTy)
          ? encoding
          : cast<triton::gpu::DotOperandEncodingAttr>(encoding).getParent());
}

FailureOr<LinearLayout> computeTransposeShuffleMapping(
    RankedTensorType tensorType, const LinearLayout &regMapping,
    int64_t numElemsPerLoad, unsigned numPackedVals, unsigned tileHeight,
    unsigned threadsPerWarp, bool hasDPASOperandType, MLIRContext *ctx) {
  StringAttr kRegister = StringAttr::get(ctx, "register");
  LinearLayout shuffleMapping =
      LinearLayout::identity1D(numElemsPerLoad, kRegister, kRegister);

  // Improve this. The current 2D block load only transposes the matrix at
  // i32 granularity. We still need to perform an additional in-register
  // transpose from i32 -> (N × ElemSizeInBits) tiles, using the tile width.
  // At the moment, we can only achieve this using a bitcast operation,
  // which implicitly uses the sub-group size as the transpose width. To
  // optimize further, we should implement this with inline VISA
  // instructions.

  // tileHeight becomes width after transposing.
  unsigned widthToTranspose = tileHeight;
  if (hasDPASOperandType) {
    // For the DPAS related layout, we will do the shuffle at first in the
    // unpacking of the elements at the DPAS operands granularity.
    // And then we will do the transposing. So the transposing width is DPAS
    // op shapes.
    DpasEncodingAttr::OpIdx opIdx = getOpIdx(tensorType);
    DpasEncodingAttr dpasLayout = getDpasLayout(tensorType);
    switch (opIdx) {
    case DpasEncodingAttr::OpIdx::OperandA: {
      widthToTranspose = dpasLayout.getDPASInstShapeA()[1];
      break;
    }
    case DpasEncodingAttr::OpIdx::OperandB: {
      widthToTranspose = dpasLayout.getDPASInstShapeB()[1];
      break;
    }
    case DpasEncodingAttr::OpIdx::OperandC: {
      widthToTranspose = dpasLayout.getDPASInstShapeC()[1];
      break;
    }
    }
    // For shuffle the transposed Dot operands matrix, we can shuffle the
    // loaded matrix in an reverse order.
    auto invertMapping = regMapping.invert();
    bool foundSurjective = false;
    for (unsigned numElemsPerSurjectiveTile = numElemsPerLoad;
         numElemsPerSurjectiveTile > 0; numElemsPerSurjectiveTile >>= 1) {
      auto layout =
          invertMapping.resizeInDim(kRegister, numElemsPerSurjectiveTile)
              .resizeOutDim(kRegister, numElemsPerSurjectiveTile);
      if (layout.isSurjective()) {
        shuffleMapping =
            layout * LinearLayout::identity1D(numElemsPerLoad /
                                                  numElemsPerSurjectiveTile,
                                              kRegister, kRegister);
        foundSurjective = true;
        break;
      }
    }
    if (!foundSurjective)
      return failure();
  }

  if (numPackedVals > 1 && (widthToTranspose) != threadsPerWarp)
    return failure();

  return shuffleMapping;
}

bool validate2DBlockLoadTile(const LinearLayout &ll, unsigned memContiguousDim,
                             unsigned elemSizeInBits,
                             RankedTensorType tensorType,
                             bool oneMatrixPerLoadForBT,
                             AxisInfo *maskAxisInfo) {
  auto sizeInfo = getBlockIOTileSize<true>(ll, memContiguousDim, elemSizeInBits,
                                           maskAxisInfo, oneMatrixPerLoadForBT);
  if (!sizeInfo.isValid())
    return false;

  unsigned packedElemSizeInBits = elemSizeInBits * sizeInfo.numElemPerPackedVal;
  if (!check2DBlockAddressPayloadRestriction(packedElemSizeInBits,
                                             sizeInfo.tileWidth))
    return false;

  constexpr int MAX_WIDTH = 64;
  unsigned totalBytesPerRowPerMatrix =
      sizeInfo.tileWidth * packedElemSizeInBits / 8;
  if (totalBytesPerRowPerMatrix > MAX_WIDTH)
    return false;

  // For transposed loads, verify computeTransposeShuffleMapping will succeed.
  // sizeInfo.vBlocks is already capped by getBlockIOTileSize<true>.
  if (sizeInfo.transpose && sizeInfo.regPackedBases.has_value()) {
    MLIRContext *ctx = ll.getBases().begin()->first.getContext();
    StringAttr kRegister = StringAttr::get(ctx, "register");
    std::vector<std::vector<int>> bases(sizeInfo.regPackedBases->size());
    llvm::transform(*sizeInfo.regPackedBases, bases.begin(),
                    [](int base) { return std::vector<int>{base}; });
    LinearLayout regMapping({{kRegister, bases}},
                            {{kRegister, ll.getInDimSize(kRegister)}},
                            /*requireSurjective=*/true);

    unsigned threadsPerWarp = ll.getInDimSize(StringAttr::get(ctx, "lane"));
    int64_t numElemsPerLoad =
        mlir::ceil(sizeInfo.tileHeight * sizeInfo.tileWidth *
                       sizeInfo.numElemPerPackedVal * sizeInfo.vBlocks,
                   (int)threadsPerWarp);

    bool hasDPASOperandType = false;
    if (hasDpasEncoding(tensorType) || hasDotDpasEncoding(tensorType)) {
      auto dpasLayout = getDpasLayout(tensorType);
      unsigned opsPerChannel = dpasLayout.getOpsPerChannel();
      auto opIdx = getOpIdx(tensorType);

      bool matchesDPASPrecision =
          (opsPerChannel == 4 && elemSizeInBits == 8) ||
          (opsPerChannel == 2 && elemSizeInBits == 16) ||
          (opsPerChannel == 1 && elemSizeInBits == 32);
      if (matchesDPASPrecision && (opIdx == DpasEncodingAttr::OpIdx::OperandA ||
                                   opIdx == DpasEncodingAttr::OpIdx::OperandB))
        hasDPASOperandType = true;
      if (opIdx == DpasEncodingAttr::OpIdx::OperandC)
        hasDPASOperandType = true;
    }

    if (failed(computeTransposeShuffleMapping(
            tensorType, regMapping, numElemsPerLoad,
            sizeInfo.numElemPerPackedVal, sizeInfo.tileHeight, threadsPerWarp,
            hasDPASOperandType, ctx)))
      return false;
  }

  return true;
}

} // namespace mlir::triton::gpu::intel
