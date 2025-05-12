#include <vector>

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir::triton::gpu::intel;

namespace mlir::triton::gpu {
namespace {

// We use the following nomenclature in this file.
//
//  - ctaLayout: A layout for one block, i.e. input dims [register, lane, warp]
//    for register layouts, and input dims [offset] for shared layouts.
//  - cgaLayout: Arrangement of multiple blocks, i.e. input dims [block].
//
// Note that this is inconsistent with the type name CTALayoutAttr.  That type
// is equivalent to our cgaLayout.
//
// IMO the name CTALayoutAttr is wrong.  If we tried to be consistent anyway,
// then we'd have to rename ctaLayout to "warpLayout".  I think that's more
// confusing than being inconsistent about "cgaLayout", especially when we have
// to consider the size of the warpLayout (surely that's not the "warpSize").

#define S(v) StringAttr::get(ctx, (v))

// Returns a 1D -> ND layout that's equivalent to creating a 1D -> 1D mapping of
// size product(shape) and then reshaping to permute(shape, order).
LinearLayout identityND(StringAttr inDimName, ArrayRef<unsigned> shape,
                        ArrayRef<unsigned> order,
                        ArrayRef<StringAttr> outDimNames) {
  assert(shape.size() == order.size() && "shape and order must have same size");

  MLIRContext *ctx = inDimName.getContext();
  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < shape.size(); i++) {
    // Start with the most-minor dimension, which is order[0].
    int dim = order[i];
    ret *= LinearLayout::identity1D(shape[dim], inDimName, outDimNames[dim]);
  }
  return ret;
}

// Make a LinearLayout that maps a block-id to an N-dimensional index.
//
// The tensor is split up into CTAsPerCGA pieces, which are distributed among
// the CTAsPerCGA CTAs (i.e. blocks) in the CGA (i.e. groups).
//
// See the nomenclature note at the top of the file for an explanation of why
// this is called makeCgaLayout when it accepts a CTALayoutAttr.
LinearLayout makeCgaLayout(CTALayoutAttr layout) {
  MLIRContext *ctx = layout.getContext();
  StringAttr kBlock = S("block");

  int rank = layout.getCTAOrder().size();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < rank; i++) {
    // Start with the most minor dimension, which is order[0].
    int dim = layout.getCTAOrder()[i];
    int split = layout.getCTASplitNum()[dim];
    int ctas = layout.getCTAsPerCGA()[dim];
    assert(ctas % split == 0 && "split must divide ctas");
    ret *= LinearLayout::identity1D(split, kBlock, outDimNames[dim]) *
           LinearLayout::zeros1D(ctas / split, kBlock, outDimNames[dim]);
  }

  // Transpose to standard order (dim0, dim1, ...).
  return ret.transposeOuts(outDimNames);
}

// Shrinks the output set of a layout function while leaving the input set
// unchanged, by making high-order inputs in inDimName map to the same output.
// Attempts to shrink down to desiredSize, but this is not always possible just
// by modifying one the specified input dimension.
//
// We do this by making the most-major inputs to the layout map to 0.  This
// effectively duplicates data along that input dimension.  For example, this
// layout has out-dim size 32:
//
//   L(register=1) = 8
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 16.
//
// If we shrink it to size 16 along the `lane` dimension, we set L(lane=2) to 0:
//
//   L(register=1) = 8
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 0.
//
// This means that lane=2 has the same data as lane=0.
//
// If we shrink to size 8 along the lane dimension, we set L(lane=1) = 0 as
// well.  But when we do this, we have to remove bit 1 (the value of L(lane=1))
// from all other bases:
//
//   L(register=1) = 4
//   L(register=2) = 2
//   L(register=1) = 1
//   L(lane=1) = 0
//   L(lane=2) = 0.
//
// Note this only works because the bases are powers of two.  I don't quite know
// what to do when they're not.
LinearLayout shrinkCodomain(const LinearLayout &layout, StringAttr inDimName,
                            StringAttr outDimName, int desiredSize) {
  assert(llvm::isPowerOf2_32(desiredSize) &&
         "desiredSize must be a power of 2");
  int outDimIdx = layout.getOutDimIndex(outDimName);
  int desiredZeros =
      llvm::Log2_32(layout.getOutDimSize(outDimName) / desiredSize);
  if (desiredZeros == 0) {
    return layout;
  }

  // Find the desiredZeros most-major basis vectors that are not already zero.
  // These are the ones we will set to zero.
  SmallVector<int> basesToZero;
  for (int i = layout.getInDimSizeLog2(inDimName) - 1;
       i >= 0 && basesToZero.size() < desiredZeros; i--) {
    int basis = layout.getBasis(inDimName, i, outDimName);
    if (basis != 0) {
      basesToZero.push_back(basis);
    }
  }

  // Bail if all the bases are already zero; nothing more we can do.
  if (basesToZero.empty()) {
    return layout;
  }

  // The algorithm below only works because the bases are powers of two.  I'm
  // not sure what to do otherwise.
  assert(llvm::all_of(basesToZero,
                      [&](int basis) { return llvm::isPowerOf2_32(basis); }) &&
         "bad bases");

  // We want to zero out the bases in `basesToZero`, and also "shift out" the
  // corresponding bits from all other bases.  For example if we remove the
  // basis with value 8 = 0b100, then if another basis has value 26 = 0b11010,
  // the 1 in its 3rd position gets removed and it becomes 10 = 0b1010.
  //
  // We could manually alter the bases in `layout` to achieve this, but it's
  // perhaps simpler to use the linearity of LLs to our advantage.
  //
  // Consider the function O which is the identity map from out-dims to
  // out-dims.  We can easily calculate what happens when we remove the relevant
  // bases from O.  Call this new function O'.
  //
  // Because of linearity, removing the bases from L is equivalent to composing
  // L with O'.  So that's what we do below.

  // Construct the out-dims -> out-dims identity layout O.
  LinearLayout outputIdentity = LinearLayout::empty();
  for (StringAttr dim : layout.getOutDimNames()) {
    outputIdentity *=
        LinearLayout::identity1D(layout.getOutDimSize(dim), dim, dim);
  }

  // Modify O to remove the relevant bases.
  //
  // TODO(jlebar): I don't like manually modifying bases here.  Perhaps this
  // should be a function on LinearLayout.
  LinearLayout::BasesT newBases = outputIdentity.getBases();
  llvm::sort(basesToZero);
  for (int basis : basesToZero) {
    int idx = llvm::Log2_32(basis);
    assert(idx >= 0 && "bad basis");
    for (size_t i = newBases[outDimName].size() - 1; i > idx; i--) {
      newBases[outDimName][i][outDimIdx] =
          newBases[outDimName][i - 1][outDimIdx];
    }
    newBases[outDimName][idx][outDimIdx] = 0;
  }

  // Construct O'.
  LinearLayout transform(std::move(newBases),
                         llvm::to_vector(layout.getOutDimNames()));

  // Compose O' with L.
  return layout.compose(transform);
}

// Combines the layout of a CTA (input dims [register, lane, warp]) with the
// layout of a CGA (i.e. a block), and ensures that the resulting layout has the
// given shape.
//
// See the nomenclature note at the top of the file for why the variable with
// type CTALayoutAttr is called cgaLayoutAttr.
LinearLayout combineCtaCgaWithShape(LinearLayout ctaLayout,
                                    CTALayoutAttr cgaLayoutAttr,
                                    ArrayRef<int64_t> shape) {
  int rank = shape.size();
  assert(ctaLayout.getNumOutDims() == rank &&
         "ctaLayout must have the same rank as shape");
  assert(cgaLayoutAttr.getCTAOrder().size() == rank &&
         "cgaLayoutAttr must have the same rank as shape");
  MLIRContext *ctx = cgaLayoutAttr.getContext();

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  llvm::SmallDenseMap<StringAttr, int64_t> labeledShape;
  for (auto [dim, size] : llvm::zip(outDimNames, shape)) {
    labeledShape[dim] = size;
  }

  LinearLayout cgaLayout =
      ensureLayoutNotLargerThan(makeCgaLayout(cgaLayoutAttr), labeledShape)
          .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  // Calculate the shape of the ctaLayout, which is `shape` divided by the
  // cgaLayout's size.
  llvm::SmallDenseMap<StringAttr, int64_t> ctaShape;
  assert(llvm::to_vector(ctaLayout.getOutDimNames()) ==
             llvm::to_vector(cgaLayout.getOutDimNames()) &&
         "bad layout");
  for (auto dim : ctaLayout.getOutDimNames()) {
    ctaShape[dim] =
        std::max(int64_t{1}, labeledShape[dim] / cgaLayout.getOutDimSize(dim));
  }

  ctaLayout = ensureLayoutNotSmallerThan(ctaLayout, ctaShape);
  ctaLayout = ensureLayoutNotLargerThan(ctaLayout, ctaShape);

  LinearLayout ret =
      (std::move(ctaLayout) * std::move(cgaLayout)).transposeOuts(outDimNames);
  for (auto dim : ret.getOutDimNames()) {
    assert(ret.getOutDimSize(dim) == labeledShape[dim] && "bad shape");
  }
  return ret;
}

} // anonymous namespace

// clang-format off
// The layout example repeat_count=8, systolic_depth=8,
// execution_size=16 and operands_per_chan=2 for warp size 32.
// For A operand:
//                       K = 16 (K = systolic depth * opsPerChan)
// <---------------------------------------------------------------------------->
// t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   ^
// t16  t17  t18  t19  t20  t21  t22  t23  t24  t25  t26  t27  t28  t29  t30  t31   |
// t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
// t16  t17  t18  t19  t20  t21  t22  t23  t24  t25  t26  t27  t28  t29  t30  t31   |
// t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   | M = 8 (repeat count)
// t16  t17  t18  t19  t20  t21  t22  t23  t24  t25  t26  t27  t28  t29  t30  t31   |
// t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
// t16  t17  t18  t19  t20  t21  t22  t23  t24  t25  t26  t27  t28  t29  t30  t31   v
// In this case, the LinearLayout bases are:
// Register:  {{2,0}, {4,0}}
// Lane:      {{0,1}, {0,2}, {0,4}, {0,8}, {1,0}}
// clang-format on
std::vector<std::vector<int32_t>> DPASRegBasesA(int opsPerChannel,
                                                int repeatCount,
                                                int threadsPerWarp,
                                                int systolicDepth) {
  std::vector<std::vector<int32_t>> regBases;

  // pack the value to i16 for scalar bit width <=16.
  assert((opsPerChannel == 4 || opsPerChannel == 2 || opsPerChannel == 1) &&
         "invalid opsPerChannel number.");
  int packedOpsPerLane = opsPerChannel == 4 ? 2 : 1;
  int packedColNum = (systolicDepth * opsPerChannel) / packedOpsPerLane;
  int rowsPerWarp = mlir::ceil<int>(threadsPerWarp, packedColNum);
  int warpRepeats = repeatCount / rowsPerWarp;

  for (int opc = 1; opc < packedOpsPerLane; opc *= 2) {
    regBases.push_back({0, opc});
  }

  for (int warp = 1; warp < warpRepeats; warp *= 2) {
    regBases.push_back({warp * rowsPerWarp, 0});
  }

  return regBases;
}

std::vector<std::vector<int32_t>>
DPASLaneBasesA(int opsPerChannel, int threadsPerWarp, int systolicDepth) {
  std::vector<std::vector<int32_t>> laneBases;

  // pack the value to i16 for scalar bit width <=16.
  assert((opsPerChannel == 4 || opsPerChannel == 2 || opsPerChannel == 1) &&
         "invalid opsPerChannel number.");
  int packedOpsPerLane = opsPerChannel == 4 ? 2 : 1;
  int packedColNum = (systolicDepth * opsPerChannel) / packedOpsPerLane;

  for (int tid = 1; tid < packedColNum; tid *= 2) {
    laneBases.push_back({0, packedOpsPerLane * tid});
  }
  for (int tid = packedColNum; tid < threadsPerWarp; tid *= 2) {
    laneBases.push_back({tid / packedColNum, 0});
  }

  return laneBases;
}

// For B operand:
//               execution size = 16
//<-------------------------------------------------->
// t0  t1  t2  t3  ~ t12 t13 t14 t15   ^              ^
//.   .   .   .   .   .   .   .   .   | opsPerChan=2 |
// t0  t1  t2  t3  ~ t12 t13 t14 t15   v              |
// t16 t17 t18 t19 ~ t28 t29 t30 t31                  |
//.   .   .   .   .   .   .   .   .                  |
// t16 t17 t18 t19 ~ t28 t29 t30 t31                  | systolic depth = 8
// t0  t1  t2  t3  ~ t12 t13 t14 t15                  |
//.   .   .   .   .   .   .   .   .                  |
// t0  t1  t2  t3  ~ t12 t13 t14 t15                  |
// t16 t17 t18 t19 ~ t28 t29 t30 t31                  |
//.   .   .   .   .   .   .   .   .                  |
// t16 t17 t18 t19 ~ t28 t29 t30 t31                  v
// In this case, the LinearLayout bases are:
// Register:  {{1,0}, {4,0}, {8,0}}
// Lane:      {{0,1}, {0,2}, {0,4}, {0,8}, {2,0}}
std::vector<std::vector<int32_t>> DPASRegBasesB(int opsPerChannel,
                                                int executionSize,
                                                int threadsPerWarp,
                                                int systolicDepth) {
  int rowsPerWarp = threadsPerWarp / executionSize;
  int warpRepeats = systolicDepth / rowsPerWarp;
  std::vector<std::vector<int32_t>> regBases;

  for (int opc = 1; opc < opsPerChannel; opc *= 2) {
    regBases.push_back({opc, 0});
  }
  for (int rid = rowsPerWarp; rid < systolicDepth; rid *= 2) {
    regBases.push_back({rid * opsPerChannel, 0});
  }

  return regBases;
}

std::vector<std::vector<int32_t>>
DPASLaneBasesB(int opsPerChannel, int threadsPerWarp, int executionSize) {
  std::vector<std::vector<int32_t>> laneBases;

  for (int tid = 1; tid < executionSize; tid *= 2) {
    laneBases.push_back({0, tid});
  }
  int rowsPerWarp = threadsPerWarp / executionSize;
  for (int row = 1; row < rowsPerWarp; row *= 2) {
    laneBases.push_back({row * opsPerChannel, 0});
  }

  return laneBases;
}

// For C operand:
//        execution size = 16
//<---------------------------------->
// t0  t1  t2  t3  ~ t12 t13 t14 t15          ^
// t16 t17 t18 t19 ~ t28 t29 t30 t31          |
// .   .   .   .   .   .   .   .   .          |
// .   .   .   .   .   .   .   .   .          | repeatCount = 8
// t0  t1  t2  t3  ~ t12 t13 t14 t15          |
// t16 t17 t18 t19 ~ t28 t29 t30 t31          v
// In this case, the LinearLayout bases are:
// Register:  {{2,0}, {4,0}}
// Lane:      {{0,1}, {0,2}, {0,4}, {0,8}, {1,0}}
std::vector<std::vector<int32_t>>
DPASRegBasesC(int repeatCount, int executionSize, int threadsPerWarp) {
  int rowsPerWarp = threadsPerWarp / executionSize;

  std::vector<std::vector<int32_t>> regBases;

  for (int rid = rowsPerWarp; rid < repeatCount; rid *= 2) {
    regBases.push_back({rid, 0});
  }

  return regBases;
}

std::vector<std::vector<int32_t>>
DPASLaneBasesC(int repeatCount, int executionSize, int threadsPerWarp) {
  std::vector<std::vector<int32_t>> laneBases;

  for (int tid = 1; tid < executionSize; tid *= 2) {
    laneBases.push_back({0, tid});
  }
  int rowsPerWarp = threadsPerWarp / executionSize;
  for (int row = 1; row < rowsPerWarp; row *= 2) {
    laneBases.push_back({row, 0});
  }

  return laneBases;
}

LinearLayout DPAStoLinearLayout(ArrayRef<int64_t> shape, Attribute layout,
                                unsigned opIdx) {
  assert(opIdx < 3 && "opIdx must be 0, 1, or 2");
  auto dpas = dyn_cast<DpasEncodingAttr>(layout);
  assert(dpas && "Must be DPAS layout");

  int rank = shape.size();
  assert(rank == dpas.getWarpsPerCTA().size() && (rank == 2 || rank == 3) &&
         "Invalid rank");

  MLIRContext *ctx = dpas.getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");

  auto warpsPerCTA = dpas.getWarpsPerCTA();
  int threadsPerWarp = product<unsigned>(dpas.getThreadsPerWarp());
  unsigned opsPerChannel = dpas.getOpsPerChannel();
  auto repCluster = dpas.getRepCluster();
  auto tileLayout = LinearLayout::empty();
  int systolicDepth = dpas.getSystolicDepth();
  int repeatCount = dpas.getRepeatCount();
  int executionSize = dpas.getExecutionSize();
  unsigned KDim, nonKDim;
  if (opIdx == 0) { // Operand A
    auto regBasesA = DPASRegBasesA(opsPerChannel, repeatCount, threadsPerWarp,
                                   systolicDepth);
    auto laneBasesA =
        DPASLaneBasesA(opsPerChannel, threadsPerWarp, systolicDepth);
    tileLayout = LinearLayout({{kRegister, regBasesA}, {kLane, laneBasesA}},
                              ArrayRef(outDimNames).take_back(2));
    // A only repeats by repCluster[rank - 2]
    nonKDim = rank - 2;
    KDim = rank - 1;
    tileLayout *= LinearLayout::identity1D(repCluster[nonKDim], kRegister,
                                           outDimNames[nonKDim]);

    // K-dimension is shared among warps
    tileLayout *=
        LinearLayout::zeros1D(warpsPerCTA[KDim], kWarp, outDimNames[KDim]);
    tileLayout *= LinearLayout::identity1D(warpsPerCTA[nonKDim], kWarp,
                                           outDimNames[nonKDim]);
    if (rank == 3)
      tileLayout *=
          LinearLayout::identity1D(warpsPerCTA[0], kWarp, outDimNames[0]);

  } else if (opIdx == 1) { // Operand B
    auto regBasesB = DPASRegBasesB(opsPerChannel, executionSize, threadsPerWarp,
                                   systolicDepth);
    auto laneBasesB =
        DPASLaneBasesB(opsPerChannel, threadsPerWarp, executionSize);
    tileLayout = LinearLayout({{kRegister, regBasesB}, {kLane, laneBasesB}},
                              ArrayRef(outDimNames).take_back(2));
    // B only repeats by repCluster[rank - 1]
    nonKDim = rank - 1;
    KDim = rank - 2;
    tileLayout *= LinearLayout::identity1D(repCluster[nonKDim], kRegister,
                                           outDimNames[nonKDim]);

    // K-dimension is shared among warps
    tileLayout *= LinearLayout::identity1D(warpsPerCTA[nonKDim], kWarp,
                                           outDimNames[nonKDim]);
    tileLayout *=
        LinearLayout::zeros1D(warpsPerCTA[KDim], kWarp, outDimNames[KDim]);
    if (rank == 3)
      tileLayout *=
          LinearLayout::identity1D(warpsPerCTA[0], kWarp, outDimNames[0]);
  } else { // opIdx=2 -> Operand C
    auto regBasesC = DPASRegBasesC(repeatCount, executionSize, threadsPerWarp);
    auto laneBasesC =
        DPASLaneBasesC(repeatCount, executionSize, threadsPerWarp);
    tileLayout = LinearLayout({{kRegister, regBasesC}, {kLane, laneBasesC}},
                              ArrayRef(outDimNames).take_back(2));
    // The per-inst layout is repeated at each repCluster.
    // Hence, multiply with the identity layouts starting from the
    // least significant dimension.
    nonKDim = rank - 2;
    KDim = rank - 1;
    tileLayout *= LinearLayout::identity1D(repCluster[KDim], kRegister,
                                           outDimNames[KDim]);
    tileLayout *= LinearLayout::identity1D(repCluster[nonKDim], kRegister,
                                           outDimNames[nonKDim]);

    // // The identical layout is repeated among warps
    tileLayout *=
        LinearLayout::identity1D(warpsPerCTA[KDim], kWarp, outDimNames[KDim]);
    tileLayout *= LinearLayout::identity1D(warpsPerCTA[nonKDim], kWarp,
                                           outDimNames[nonKDim]);
    if (rank == 3)
      tileLayout *=
          LinearLayout::identity1D(warpsPerCTA[0], kWarp, outDimNames[0]);
  }

  // Lastly, the layout repeats to match the shape.
  // Operand A/B repeats through the K-dimension first then repeats
  // through the non-K dimension.
  SmallVector<int64_t> numReps = dpas.getDPASRepetitions(shape, opIdx);

  // numReps is always 3D, we should add 1 to dim id when rank is 2
  int repDimK = rank == 2 ? KDim + 1 : KDim;
  int repDimNonK = rank == 2 ? nonKDim + 1 : nonKDim;
  tileLayout *=
      LinearLayout::identity1D(numReps[repDimK], kRegister, outDimNames[KDim]);
  tileLayout *= LinearLayout::identity1D(numReps[repDimNonK], kRegister,
                                         outDimNames[nonKDim]);
  if (rank == 3)
    tileLayout *=
        LinearLayout::identity1D(numReps[0], kRegister, outDimNames[0]);

  return combineCtaCgaWithShape(std::move(tileLayout),
                                CTALayoutAttr::getDefault(ctx, rank), shape);
}

LinearLayout dotOperandDpasToLinearLayout(DotOperandEncodingAttr dotDpasLayout,
                                          ArrayRef<int64_t> shape) {
  auto dpasLayout = cast<intel::DpasEncodingAttr>(dotDpasLayout.getParent());

  return DPAStoLinearLayout(shape, dpasLayout, dotDpasLayout.getOpIdx());
}

namespace {

// maybe unused?
static LinearLayout broadcastedDotOperandLayout(MLIRContext *ctx,
                                                ArrayRef<unsigned> shape,
                                                ArrayRef<unsigned> order,
                                                unsigned broadcastDim,
                                                StringAttr inDimName) {
  int rank = shape.size();
  auto dimNames = standardOutDimNames(ctx, rank);
  LinearLayout layout = LinearLayout::empty();

  for (auto d : order) {
    if (d == broadcastDim) {
      layout *= LinearLayout::zeros1D(shape[d], inDimName, dimNames[d]);
    } else {
      layout *= LinearLayout::identity1D(shape[d], inDimName, dimNames[d]);
    }
  }
  return layout;
}

} // namespace

LinearLayout
subgroup2DBlockToLinearLayout(ArrayRef<int64_t> blockShape,
                              intel::Subgroup2DBlockEncodingAttr layout,
                              unsigned kWidth, unsigned opIdx) {
  auto ctx = layout.getContext();
  int rank = blockShape.size();
  assert(rank == layout.getRank());
  auto dimNames = standardOutDimNames(ctx, rank);
  SmallVector<unsigned> loadTileSize = llvm::to_vector(layout.getInstrShape());
  assert(loadTileSize.size() == 2);
  loadTileSize[1] *= layout.getNumBlocks();
  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  auto warpOrder = getMatrixOrder(rank, /*rowMajor*/ true);

  auto printVector = [](const auto vec) {
    for (size_t i = 0; i < vec.size(); i++) {
      llvm::errs() << vec[i] << " ";
    }
  };

  auto printBases = [&printVector](const auto base) {
    for (size_t i = 0; i < base.size(); i++) {
      llvm::errs() << i << " : ";
      printVector(base[i]);
      llvm::errs() << "\n";
    }
  };

  if (opIdx == 0) {
    LinearLayout::BasesT bases;

    // start with the DPAS tile
    int opsPerChannel = 2; // TODO: how to get this?
    auto regBases = DPASRegBasesA(
        opsPerChannel, DpasEncodingAttr::DPASCapability::repeatCount,
        layout.getThreadsPerWarp(),
        DpasEncodingAttr::DPASCapability::systolicDepth);

    llvm::errs() << "regBases\n";
    printBases(regBases);

    bases[kRegister] = regBases;

    auto laneBases =
        DPASLaneBasesA(opsPerChannel, layout.getThreadsPerWarp(),
                       DpasEncodingAttr::DPASCapability::systolicDepth);

    llvm::errs() << "laneBases:\n";
    printBases(laneBases);

    bases[kLane] = laneBases;

    auto ctaLayout = LinearLayout(bases, dimNames);
    llvm::errs() << "ctaLayer based on DPAS tile: " << ctaLayout << "\n";

    auto order = getOrderForDotOperand(opIdx, rank, /*kContig*/ true);

    unsigned inner = order[0];
    unsigned outer = order[1];
    llvm::errs() << "outer = " << outer << "\n";
    llvm::errs() << "inner = " << inner << "\n";

    // expand the DPAS tile to match the desired load size
    auto numDPASInstPerOuterDim =
        loadTileSize[outer] / ctaLayout.getOutDimSize(dimNames[outer]);
    auto numDPASInstPerInnerDim =
        loadTileSize[inner] / ctaLayout.getOutDimSize(dimNames[inner]);
    llvm::errs() << "numDPASInstPerOuterDim = " << numDPASInstPerOuterDim
                 << "\n";
    llvm::errs() << "numDPASInstPerInnerDim = " << numDPASInstPerInnerDim
                 << "\n";

    ctaLayout *= LinearLayout::identity1D(numDPASInstPerOuterDim, kRegister,
                                          dimNames[outer]);
    ctaLayout *= LinearLayout::identity1D(numDPASInstPerInnerDim, kRegister,
                                          dimNames[inner]);

    // replicate the tile
    auto numReps = layout.getNumReps();
    assert(numReps.size() == 2);
    llvm::errs() << "numReps: ";
    printVector(numReps);
    llvm::errs() << "\n";

    ctaLayout *=
        LinearLayout::identity1D(numReps[outer], kRegister, dimNames[outer]);
    ctaLayout *= LinearLayout::identity1D(
        numReps[inner] / numDPASInstPerInnerDim, kRegister, dimNames[inner]);
    llvm::errs() << "ctaLayout after replicating: " << ctaLayout << "\n";

    // Apply warp layout.
    ctaLayout *=
        broadcastedDotOperandLayout(ctx, layout.getWarpsPerCTA(), warpOrder,
                                    rank - 1, kWarp)
            .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));
    llvm::errs() << "ctaLayout after applying warp layout: " << ctaLayout
                 << "\n";

    return combineCtaCgaWithShape(ctaLayout, layout.getCTALayout(), blockShape);
  } else if (opIdx == 1) {
    LinearLayout::BasesT bases;

    // start with the DPAS tile
    int opsPerChannel = 2; // TODO: how to get this?
    auto regBases = DPASRegBasesB(
        opsPerChannel, DpasEncodingAttr::DPASCapability::repeatCount,
        layout.getThreadsPerWarp(),
        DpasEncodingAttr::DPASCapability::systolicDepth);

    llvm::errs() << "regBases\n";
    printBases(regBases);

    bases[kRegister] = regBases;

    auto laneBases =
        DPASLaneBasesB(opsPerChannel, layout.getThreadsPerWarp(),
                       DpasEncodingAttr::DPASCapability::systolicDepth);

    llvm::errs() << "laneBases:\n";
    printBases(laneBases);

    bases[kLane] = laneBases;

    auto ctaLayout = LinearLayout(bases, dimNames);
    llvm::errs() << "ctaLayer based on DPAS tile: " << ctaLayout << "\n";

    auto order = getOrderForDotOperand(opIdx, rank, /*kContig*/ true);

    unsigned inner = order[0];
    unsigned outer = order[1];
    llvm::errs() << "outer = " << outer << "\n";
    llvm::errs() << "inner = " << inner << "\n";

    // expand the DPAS tile to match the desired load size
    auto numDPASInstPerOuterDim =
        loadTileSize[outer] / ctaLayout.getOutDimSize(dimNames[outer]);
    auto numDPASInstPerInnerDim =
        loadTileSize[inner] / ctaLayout.getOutDimSize(dimNames[inner]);
    llvm::errs() << "numDPASInstPerOuterDim = " << numDPASInstPerOuterDim
                 << "\n";
    llvm::errs() << "numDPASInstPerInnerDim = " << numDPASInstPerInnerDim
                 << "\n";

    ctaLayout *= LinearLayout::identity1D(numDPASInstPerOuterDim, kRegister,
                                          dimNames[outer]);
    ctaLayout *= LinearLayout::identity1D(numDPASInstPerInnerDim, kRegister,
                                          dimNames[inner]);

    // replicate the tile
    auto numReps = layout.getNumReps();
    assert(numReps.size() == 2);
    llvm::errs() << "numReps: ";
    printVector(numReps);
    llvm::errs() << "\n";

    ctaLayout *=
        LinearLayout::identity1D(numReps[outer], kRegister, dimNames[outer]);
    ctaLayout *= LinearLayout::identity1D(
        numReps[inner] / numDPASInstPerInnerDim, kRegister, dimNames[inner]);
    llvm::errs() << "ctaLayout after replicating: " << ctaLayout << "\n";

    // Apply warp layout.
    ctaLayout *=
        broadcastedDotOperandLayout(ctx, layout.getWarpsPerCTA(), warpOrder,
                                    rank - 2, kWarp)
            .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));
    llvm::errs() << "ctaLayout after applying warp layout: " << ctaLayout
                 << "\n";

    return combineCtaCgaWithShape(ctaLayout, layout.getCTALayout(), blockShape);
  }
  llvm_unreachable("unhandle Op Idx");
}

} // namespace mlir::triton::gpu
