#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::gpu::intel {
namespace {
constexpr inline unsigned minSubGroupTransposeWidth = 8;

bool canTypeBeConvertedForSubGroupTranspose(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case([](FloatType floatTy) {
        // Support via bitcasting to integer type.
        return isValidElementTypeForSubGroupTranspose(
            IntegerType::get(floatTy.getContext(), floatTy.getWidth()));
      })
      .Case([](IntegerType intTy) {
        // Support via extending to supported type.
        return isValidElementTypeForSubGroupTranspose(intTy) ||
               intTy.getWidth() < minSubGroupTransposeWidth;
      })
      .Case([](PointerType) {
        // Support via ptrtoint
        return true;
      })
      .Default(false);
}

// Return a vector such as:
// [[0, 1], [0, 2], [0, 4], ..., [0, laneSize / 2], [laneSize, 0], ...,
// [registerSize / 2, 0]],
// i.e., mapping registers to lanes till laneSize and performing an ID
// conversion afterwards.
std::vector<std::vector<int32_t>>
buildSubGroupTransposeRegisterBases(int32_t registerSize, int32_t laneSize) {
  std::vector<std::vector<int32_t>> bases;
  std::vector<int32_t> curr(2);
  for (int32_t i = 1; i < laneSize; i *= 2) {
    curr[1] = i;
    bases.push_back(curr);
  }
  curr[1] = 0;
  for (int32_t i = laneSize; i < registerSize; i *= 2) {
    curr[0] = i;
    bases.push_back(curr);
  }
  return bases;
}

// Return a vector such as:
// [[0, 1], [0, 2], [0, 4], ..., [0, laneSize / 2], [1, 0], ...,
// [registerSize / (laneSize * 2), 0]],
// i.e., mapping registers to lanes till laneSize and performing an ID
// conversion afterwards.
std::vector<std::vector<int32_t>>
buildContiguousSubGroupTransposeRegisterBases(int32_t registerSize,
                                              int32_t laneSize) {
  std::vector<std::vector<int32_t>> bases;
  std::vector<int32_t> curr(2);
  int32_t i = 1;
  for (; i < laneSize; i *= 2) {
    curr[1] = i;
    bases.push_back(curr);
  }
  curr[1] = 0;
  for (int32_t j = 1; i < registerSize; i *= 2, j *= 2) {
    curr[0] = j;
    bases.push_back(curr);
  }
  return bases;
}

// Return a vector such as:
// [[registerSize / laneSize, 0], [registerSize / laneSize * 2, 0], ...,
// [registerSize / 2, 0]]
// i.e., mapping registers to lanes till laneSize and performing an ID
// conversion afterwards.
std::vector<std::vector<int32_t>>
buildContiguousSubGroupTransposeLaneBases(int32_t registerSize,
                                          int32_t laneSize) {
  std::vector<std::vector<int32_t>> bases;
  std::vector<int32_t> curr(2);
  for (int32_t i = registerSize / laneSize; i < registerSize; i *= 2) {
    curr[0] = i;
    bases.push_back(curr);
  }
  return bases;
}

// Return a vector such as:
// [[0, 1], [0, 2], [0, 4], ..., [0, laneSize / 2], [1, 0], ...,
// [registerSize / (2 * laneSize), 0]]
// i.e., mapping registers to lanes till laneSize and repeating the pattern
// afterwards.
std::vector<std::vector<int32_t>>
buildSubGroupShuffleRegisterBases(int32_t registerSize, int32_t laneSize) {
  std::vector<std::vector<int32_t>> bases;
  std::vector<int32_t> curr(2);
  for (int32_t i = 1; i < laneSize; i *= 2) {
    curr[1] = i;
    bases.push_back(curr);
  }
  curr[1] = 0;
  for (int32_t i = laneSize, val = 1; i < registerSize; i *= 2, val *= 2) {
    curr[0] = val;
    bases.push_back(curr);
  }
  return bases;
}

// Return a vector such as:
// [[1, 0], [2, 0], [4, 0], ..., [registerSize / laneSize, 0], [0, 1], ...,
// [0, laneSize/2]]
// i.e., mapping registers to registers till registerSize / laneSize (all
// contiguous registers) and then to lanes.
std::vector<std::vector<int32_t>>
buildContiguousSubGroupShuffleRegisterBases(int32_t registerSize,
                                            int32_t laneSize) {
  std::vector<std::vector<int32_t>> bases;
  std::vector<int32_t> curr(2);
  int i = 1;
  for (; i < registerSize / laneSize; i *= 2) {
    curr[0] = i;
    bases.push_back(curr);
  }
  curr[0] = 0;
  for (int32_t val = 1; i < registerSize; i *= 2, val *= 2) {
    curr[1] = val;
    bases.push_back(curr);
  }
  return bases;
}

// Return a vector such as:
// [[1, 0], [2, 0], [4, 0], ..., [laneSize / 2, 0]],
// i.e., mapping lanes to registers.
std::vector<std::vector<int32_t>>
buildSubGroupTransposeLaneBases(int32_t laneSize) {
  std::vector<std::vector<int32_t>> bases;
  std::vector<int32_t> curr(2);
  for (int32_t i = 1; i < laneSize; i *= 2) {
    curr[0] = i;
    bases.push_back(curr);
  }
  return bases;
}

} // namespace

bool isDpasToDotShortcut(RankedTensorType dpasTy, RankedTensorType dotTy) {
  auto dpasLayout = dyn_cast<DpasEncodingAttr>(dpasTy.getEncoding());
  auto dotOperandLayout = dyn_cast<DotOperandEncodingAttr>(dotTy.getEncoding());
  // dpas -> dot_operand conversion when:
  if (dpasLayout && dotOperandLayout &&
      dotOperandLayout.getParent() == dpasLayout) {
    SmallVector<unsigned> shapeC = dpasLayout.getDPASInstShapeC();
    SmallVector<unsigned> shapeA = dpasLayout.getDPASInstShapeA();
    if (dotOperandLayout.getOpIdx() == 0 && /* A operands. */
        dpasLayout.getWarpsPerCTA().back() ==
            1 && /* The warpsPerCTA is [..., 1]. */
        shapeA[0] == shapeC[0] &&
        shapeA[1] == shapeC[1] /* C shape is equal to A shape */
    )
      return true;
  }

  return false;
}

bool cvtIsSubGroupShuffle(RankedTensorType srcTy, RankedTensorType dstTy) {
  MLIRContext *ctx = srcTy.getContext();
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  std::optional<LinearLayout> srcLayout = toLinearLayout(srcTy);
  if (!srcLayout)
    return false;

  std::optional<LinearLayout> dstLayout = toLinearLayout(dstTy);
  if (!dstLayout)
    return false;

  LinearLayout comp = dstLayout->invertAndCompose(*srcLayout);
  std::optional<LinearLayout> conversion = comp.quotient(kBlock);
  if (!conversion)
    return false;
  conversion = conversion->quotient(kWarp);
  if (!conversion)
    return false;

  // TODO: Support more kind of shuffles.
  // Expected conversion is:
  // - register=1 -> (0, 1)
  // ...
  // - register=2**i -> (0, 2**i)
  // ...
  // - register=M -> (0, 2**(M-1))
  // - register=M+1 -> (1, 0)
  // ...
  // - register=2**k -> (2**(K-M), 0)
  // ...
  // - register=2**N -> (2**(N-M), 0)
  // - lane=1 -> (0, 0)
  // ...
  // - lane=2**j -> (0, 0)
  // ...
  //   lane=2**M -> (0, 0)
  // where out dims are: [register (size 2**N), lane (size 2**M)]
  //
  // With N >= M.
  //
  // Or, when the elements managed by a given work-item are in contiguous
  // positions:
  // - register=1 -> (1, 0)
  // ...
  // - register=2**i -> (2**i, 0)
  // ...
  // - register=M -> (2**(N - M), 0)
  // ...
  // - register=2**k -> (0, 1)
  // ...
  // - register=2**N -> (0, 2**(M-1))
  // - lane=1 -> (0, 0)
  // ...
  // - lane=2**j -> (0, 0)
  // ...
  //   lane=2**M -> (0, 0)
  // where out dims are: [register (size 2**(N - M)), lane (size 2**(M + 1))]
  //
  // With N >= M.
  int32_t registerInDimSize = conversion->getInDimSize(kRegister);
  int32_t laneOutDimSize = conversion->getOutDimSize(kLane);
  return conversion->sublayoutIsZero({kLane}, {kRegister, kLane}) &&
         (conversion->getBases().lookup(kRegister) ==
              buildSubGroupShuffleRegisterBases(registerInDimSize,
                                                laneOutDimSize) ||
          conversion->getBases().lookup(kRegister) ==
              buildContiguousSubGroupShuffleRegisterBases(registerInDimSize,
                                                          laneOutDimSize));
}

bool isValidElementTypeForSubGroupTranspose(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case([](IntegerType intTy) {
        unsigned width = intTy.getWidth();
        return width == 8 || width == 16 || width == 32 || width == 64;
      })
      .Default(false);
}

bool cvtIsSubGroupTranspose(RankedTensorType srcTy, RankedTensorType dstTy) {
  if (!canTypeBeConvertedForSubGroupTranspose(srcTy.getElementType()))
    return false;

  MLIRContext *ctx = srcTy.getContext();
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  std::optional<LinearLayout> srcLayout = toLinearLayout(srcTy);
  if (!srcLayout)
    return false;

  std::optional<LinearLayout> dstLayout = toLinearLayout(dstTy);
  if (!dstLayout)
    return false;

  LinearLayout comp = dstLayout->invertAndCompose(*srcLayout);
  std::optional<LinearLayout> conversion = comp.quotient(kBlock);
  if (!conversion)
    return false;
  conversion = conversion->quotient(kWarp);
  if (!conversion)
    return false;

  // Expected conversion is:
  // - register=1 -> (0, 1)
  // ...
  // - register=2**i -> (0, 2**i)
  // ...
  // - register=M -> (0, 2**M)
  // ...
  // - register=2**k -> (2**k, 0)
  // ...
  // - register=N -> (2**N, 0)
  // - lane=1 -> (1, 0)
  // ...
  // - lane=2**j -> (2**j, 0)
  // ...
  //   lane=2**M -> (2**M, 0)
  // where out dims are: [register (size 2**(N + 1)), lane (size 2**(M + 1))]
  //
  // With N >= M.
  //
  // Alternatively, we can also lower transpositions in which the output matrix
  // has more than one contiguous row owned by the same thread, resulting in:
  //
  // - register=1 -> (0, 1)
  // ...
  // - register=2**i -> (0, 2**i)
  // ...
  // - register=M -> (0, 2**M)
  // ...
  // - register=2**k -> (1, 0)
  // ...
  // - register=N -> (2**(N-k), 0)
  // - lane=1 -> (2**(N-k+1), 0)
  // ...
  // - lane=2**j -> (2**(N-k+j), 0)
  // ...
  //   lane=2**M -> (2**(N-k+M), 0)
  // where out dims are: [register (size 2**(N + 1)), lane (size 2**(M + 1))]
  //
  // With N >= M.
  //
  // This is what we call the "contiguous" case.
  int32_t registerInDimSize = conversion->getInDimSize(kRegister);
  int32_t laneInDimSize = conversion->getInDimSize(kLane);
  return (conversion->getBases().lookup(kRegister) ==
              buildSubGroupTransposeRegisterBases(registerInDimSize,
                                                  laneInDimSize) &&
          conversion->getBases().lookup(kLane) ==
              buildSubGroupTransposeLaneBases(laneInDimSize)) ||
         (conversion->getBases().lookup(kRegister) ==
              buildContiguousSubGroupTransposeRegisterBases(registerInDimSize,
                                                            laneInDimSize) &&
          conversion->getBases().lookup(kLane) ==
              buildContiguousSubGroupTransposeLaneBases(registerInDimSize,
                                                        laneInDimSize));
}

std::optional<LinearLayout>
getReinterpretCastMapping(MLIRContext *ctx, const LinearLayout &srcLayout,
                          const LinearLayout &dstLayout) {
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");
  // The reinterpret layout casting need to get the map layout from:
  // dst = DtoSMap.compose(src)
  // StoDMap = (dst.invertAndCompose(src)).invert() = src.invertAndCompose(dst)
  auto comp = srcLayout.invertAndCompose(dstLayout).quotient({kWarp, kBlock});
  // Base on the reinterpret cast semantic, the mapping has to be invertible.
  if (comp && !comp->isInvertible())
    return std::nullopt;
  return comp;
}

std::optional<SubGroupReinterpretPackInfo>
getSubGroupReinterpretPackInfo(MLIRContext *ctx,
                               const LinearLayout &conversion) {
  StringAttr kLane = str_attr("lane");

  if (!conversion.hasInDim(kLane) || !conversion.hasOutDim(kLane))
    return std::nullopt;

  auto laneBases = conversion.getBases().lookup(kLane);
  for (size_t i = 0; i < conversion.getInDimSizeLog2(kLane); ++i) {
    unsigned lane2Lane = laneBases[i][conversion.getOutDimIndex(kLane)];
    if (!lane2Lane)
      continue;

    bool isPack = i != 0;
    unsigned curLaneBase = 1 << i;
    unsigned packedRegisterSize = isPack ? curLaneBase : lane2Lane;
    // Reinterpret cast only support shuffle value contiguously.
    unsigned expectedLane = isPack ? curLaneBase / packedRegisterSize
                                   : curLaneBase * packedRegisterSize;
    if (lane2Lane != expectedLane) {
      break;
    }
    return SubGroupReinterpretPackInfo{isPack, packedRegisterSize};
  }

  return std::nullopt;
}

bool cvtIsSubGroupReinterpret(ConvertLayoutOp op) {
  // The sub-group bitcast shuffle operation is lowered to a GenISA intrinsic
  // not implemented by the LTS driver.
  auto mod = op->getParentOfType<ModuleOp>();
  if (mod && mod->hasAttr(TritonIntelGPUDialect::getIsLTSAttrName()))
    return false;

  RankedTensorType srcTy = op.getSrc().getType();
  RankedTensorType dstTy = op.getType();
  MLIRContext *ctx = srcTy.getContext();
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");

  std::optional<LinearLayout> srcLayout = toLinearLayout(srcTy);
  if (!srcLayout)
    return false;

  std::optional<LinearLayout> dstLayout = toLinearLayout(dstTy);
  if (!dstLayout)
    return false;

  std::optional<LinearLayout> conversion =
      getReinterpretCastMapping(ctx, *srcLayout, *dstLayout);
  if (!conversion)
    return false;

  // The conversion which can be used for reinterpret cast has to be a mapping
  // from lanes to registers/lanes, i.e.,:
  // For 2xi16 -> i32, the conversion is:
  // - register=1 -> (0, 16)
  // - lane=1 -> (1, 0)
  //   lane=2 -> (0, 1)
  //   lane=4 -> (0, 2)
  //   lane=8 -> (0, 4)
  //   lane=16 -> (0, 8)
  // where out dims are: [register (size 2), lane (size 32)]
  //
  // The reverse convert for i32 -> 2xi16, the conversion is:
  // - register=1 -> (0, 1)
  // - lane=1 -> (0, 2)
  //   lane=2 -> (0, 4)
  //   lane=4 -> (0, 8)
  //   lane=8 -> (0, 16)
  //   lane=16 -> (1, 0)
  // where out dims are: [register (size 2), lane (size 32)]
  auto laneBases = conversion->getBases().lookup(kLane);

  // Check whether the mapping is valid for reinterpret cast.
  for (size_t i = 0; i < conversion->getInDimSizeLog2(kLane); i++) {
    auto lane2Reg = laneBases[i][conversion->getOutDimIndex(kRegister)];
    auto lane2Lane = laneBases[i][conversion->getOutDimIndex(kLane)];
    if (lane2Reg && lane2Lane) // invalid.
      return false;
    if (!lane2Reg && !lane2Lane) // invalid.
      return false;
  }

  std::optional<SubGroupReinterpretPackInfo> packInfo =
      getSubGroupReinterpretPackInfo(ctx, *conversion);
  if (!packInfo)
    return false;

  bool packOrUnpack = packInfo->isPack;
  unsigned packedRegisterSize = packInfo->packedRegisterSize;
  if (packedRegisterSize == 1)
    return false;

  // Reinterpret-cast lane mapping can be viewed as follows.
  // Pack case: lane base is shifted right by log2(packedElems) (example: 1).
  //   lane  1 >> 1 = 0  (lanes 0 and 1 are combined; map to a register base)
  //         2 >> 1 = 1
  //         4 >> 1 = 2
  //         8 >> 1 = 4
  //        16 >> 1 = 8
  // register=1 = 16     shuffle the register value to the lane 16.
  // Unpack case: lane base is shifted left by log2(packedElems) (example: 1).
  //   lane  1 << 1 = 2
  //         2 << 1 = 4
  //         4 << 1 = 8
  //         8 << 1 = 16
  //        16 << 1 = 0 (32 % 32 - exceeds lane-base range; map to a register
  //        base)
  //  register=1 = 1     shuffle the register value to the lane 1.
  unsigned shiftedOutLaneNum = llvm::Log2_32(packedRegisterSize);
  unsigned threadsPerWarp = conversion->getInDimSize(kLane);
  for (size_t i = 0; i < conversion->getInDimSizeLog2(kLane); i++) {
    int lane2Lane = laneBases[i][conversion->getOutDimIndex(kLane)];
    int curLaneBase = 1 << i;
    unsigned expectedMappedLane =
        (packOrUnpack ? curLaneBase >> shiftedOutLaneNum
                      : curLaneBase << shiftedOutLaneNum) %
        threadsPerWarp;
    if (lane2Lane != expectedMappedLane)
      return false;
  }

  // IGC doesn't support bitcast >= i128. Fallback to shared memory in this
  // case.
  Type elemType = srcTy.getElementType();
  unsigned bitsPerElement =
      isa<PointerType>(elemType)
          ? kPtrBitWidth
          : std::max<int>(8, elemType.getIntOrFloatBitWidth());
  if (packedRegisterSize * bitsPerElement >= 128)
    return false;

  // Check the register base mapped to the lane base to complement the shuffled
  // elements out from lane base.
  unsigned reg2LaneBitMap = 0;
  for (size_t i = 0; i < conversion->getInDimSizeLog2(kRegister); ++i) {
    auto reg2Lane = conversion->getBases().lookup(
        kRegister)[i][conversion->getOutDimIndex(kLane)];
    if (reg2Lane) {
      reg2LaneBitMap |= reg2Lane;
    }
  }
  unsigned expectedComlementLaneBitMap = (1 << shiftedOutLaneNum) - 1;
  unsigned remainedLaneNum = threadsPerWarp / packedRegisterSize;
  expectedComlementLaneBitMap = packOrUnpack
                                    ? expectedComlementLaneBitMap
                                          << llvm::Log2_32(remainedLaneNum)
                                    : expectedComlementLaneBitMap;
  return reg2LaneBitMap == expectedComlementLaneBitMap;
}

} // namespace mlir::triton::gpu::intel
