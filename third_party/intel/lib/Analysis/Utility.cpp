#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

namespace tt = mlir::triton;

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

bool isNonNegative(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  // tt.get_program_id always returns [0, 2^31-1].
  if (isa<tt::GetProgramIdOp>(defOp))
    return true;

  // tt.get_num_programs returns [1, 2^31-1].
  if (isa<tt::GetNumProgramsOp>(defOp))
    return true;

  // tt.make_range with non-negative start.
  if (auto makeRange = dyn_cast<tt::MakeRangeOp>(defOp))
    return makeRange.getStartAttr().getInt() >= 0;

  // Non-negative constant (scalar or tensor).
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getValue().isNonNegative();
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      if (denseAttr.getElementType().isSignlessInteger()) {
        return llvm::all_of(denseAttr.getValues<APInt>(),
                            [](const APInt &v) { return v.isNonNegative(); });
      }
    }
  }

  // arith.addi / arith.muli of two non-negative values. Assumes no signed
  // overflow, which holds for the bounded index expressions this helper is
  // applied to (`programId * blockSize (+ ...)`, well within the i32 positive
  // range).
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp))
    return isNonNegative(addOp.getLhs()) && isNonNegative(addOp.getRhs());
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp))
    return isNonNegative(mulOp.getLhs()) && isNonNegative(mulOp.getRhs());

  // arith.extui zero-extends into a wider type, so the result MSB is always
  // clear. (arith.remui / arith.divui are intentionally NOT treated as
  // unconditionally non-negative: their unsigned results can have the sign bit
  // set -- e.g. divui(x, 1) == x -- which is negative under the signed
  // comparisons callers use.)
  if (isa<arith::ExtUIOp>(defOp))
    return true;

  // arith.divsi: non-negative iff both dividend and divisor are non-negative.
  if (auto divOp = dyn_cast<arith::DivSIOp>(defOp))
    return isNonNegative(divOp.getLhs()) && isNonNegative(divOp.getRhs());

  // arith.remsi: result has the same sign as the dividend (truncation toward
  // zero), so a non-negative dividend guarantees a non-negative result.
  if (auto remOp = dyn_cast<arith::RemSIOp>(defOp))
    return isNonNegative(remOp.getLhs());

  // arith.extsi preserves the signed value; non-negative iff source is.
  // (arith.trunci is intentionally NOT handled: truncating a non-negative
  // value can set the sign bit of the narrower type, e.g. i32 128 -> i8 -128.)
  if (auto extOp = dyn_cast<arith::ExtSIOp>(defOp))
    return isNonNegative(extOp.getIn());

  // arith.shrsi (arithmetic right shift) replicates the sign bit; non-negative
  // iff the shifted value is non-negative.
  if (auto shrOp = dyn_cast<arith::ShRSIOp>(defOp))
    return isNonNegative(shrOp.getLhs());

  // arith.maxsi: non-negative if either operand is non-negative.
  if (auto maxOp = dyn_cast<arith::MaxSIOp>(defOp))
    return isNonNegative(maxOp.getLhs()) || isNonNegative(maxOp.getRhs());

  // arith.minsi: non-negative iff both operands are non-negative.
  if (auto minOp = dyn_cast<arith::MinSIOp>(defOp))
    return isNonNegative(minOp.getLhs()) && isNonNegative(minOp.getRhs());

  // arith.select yields one of its two value operands; non-negative iff both
  // candidate values are non-negative.
  if (auto selOp = dyn_cast<arith::SelectOp>(defOp))
    return isNonNegative(selOp.getTrueValue()) &&
           isNonNegative(selOp.getFalseValue());

  // arith.andi is non-negative if either operand is non-negative, since
  // MSB(a & b) = MSB(a) & MSB(b).
  if (auto andOp = dyn_cast<arith::AndIOp>(defOp))
    return isNonNegative(andOp.getLhs()) || isNonNegative(andOp.getRhs());

  // tt.splat / tt.expand_dims / tt.broadcast: propagate from source.
  if (auto splatOp = dyn_cast<tt::SplatOp>(defOp))
    return isNonNegative(splatOp.getSrc());
  if (auto expandOp = dyn_cast<tt::ExpandDimsOp>(defOp))
    return isNonNegative(expandOp.getSrc());
  if (auto broadcastOp = dyn_cast<tt::BroadcastOp>(defOp))
    return isNonNegative(broadcastOp.getSrc());

  return false;
}

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

} // namespace mlir::triton::gpu::intel
