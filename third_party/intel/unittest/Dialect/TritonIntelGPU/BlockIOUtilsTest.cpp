// Tests for BlockIOUtils: validate2DBlockLoadTile, pickBlockIOLoadEncoding and
// matchRowLimitMask.
//
// PR #7487 changed computeTransposeShuffleMapping from a simple width
// comparison to a linear-layout comparison, enabling column-major B matrix
// loads with sub-group-size=32.  These tests verify:
//   1. tpw=32, f16/opsPerChan=2, column-major B is now accepted.
//   2. tpw=16, f16/opsPerChan=2, column-major B is still accepted (regression).
//
// matchRowLimitMask (issue #8565) is tested directly here because one
// branch -- a mask degenerate along both dims -- has no observable effect in
// the MaterializeBlockPointer/prefetch lit tests: the generic axis-info
// fallback permits block IO anyway, masking whether matchRowLimitMask itself
// rejected it.

#include "intel/include/Dialect/TritonIntelGPU/Transforms/BlockIOUtils.h"
#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Signals.h"
#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;
using mlir::triton::LinearLayout;

namespace {

class BlockIOUtilsTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctx.getOrLoadDialect<TritonIntelGPUDialect>();
  }

  DpasEncodingAttr makeDpas(ArrayRef<unsigned> warpsPerCTA,
                            unsigned threadsPerWarp, unsigned opsPerChan) {
    return DpasEncodingAttr::get(&ctx, /*repeatCount=*/8, /*systolicDepth=*/8,
                                 /*executionSize=*/16, opsPerChan, warpsPerCTA,
                                 /*repCluster=*/{1, 1}, threadsPerWarp,
                                 std::nullopt);
  }

  // A tensor type carrying the layout Coalesce gives a descriptor load
  // (pickDescriptorLoadStoreLayout): up to 128 bits per thread along the
  // last dim, row-major, one CTA. Pointers count as 64 bits.
  RankedTensorType makeCoalescedType(ArrayRef<int64_t> shape, Type elemTy,
                                     unsigned numWarps,
                                     unsigned threadsPerWarp) {
    unsigned rank = shape.size();
    unsigned bits = elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 64;
    int64_t elemsPerThread = std::max<int64_t>(
        ShapedType::getNumElements(shape) / (numWarps * threadsPerWarp), 1);
    SmallVector<unsigned> sizePerThread(rank, 1);
    sizePerThread.back() = std::min<int64_t>(elemsPerThread, 128 / bits);
    auto enc = BlockedEncodingAttr::get(
        &ctx, shape, sizePerThread, getMatrixOrder(rank, /*rowMajor=*/true),
        numWarps, threadsPerWarp, CGAEncodingAttr::get1CTALayout(&ctx, rank));
    return RankedTensorType::get(shape, elemTy, enc);
  }

  // The property pickBlockIOLoadEncoding promises for its result.
  static bool lowersTo2DBlockLoad(RankedTensorType type,
                                  BlockedEncodingAttr enc) {
    return validate2DBlockLoadTile(enc.toLinearLayout(type.getShape()),
                                   /*memContiguousDim=*/1,
                                   type.getElementTypeBitWidth(),
                                   type.cloneWithEncoding(enc));
  }

protected:
  MLIRContext ctx;

  // Rebuild `ll` with its "register" basis vectors reordered by `perm`
  // (perm[i] = source index for the i-th output basis). The resulting layout
  // holds the same data in each lane, with the registers renumbered -- the
  // scenario issue #7806 is about. Out-dim sizes and surjectivity are
  // preserved.
  LinearLayout permuteRegisterBases(const LinearLayout &ll,
                                    ArrayRef<unsigned> perm) {
    StringAttr kRegister = StringAttr::get(&ctx, "register");
    LinearLayout::BasesT bases = ll.getBases();
    auto it = bases.find(kRegister);
    assert(it != bases.end() && "layout has no register dim");
    std::vector<std::vector<int32_t>> &regBases = it->second;
    assert(perm.size() == regBases.size() && "permutation size mismatch");
    std::vector<std::vector<int32_t>> reordered;
    reordered.reserve(regBases.size());
    for (unsigned src : perm)
      reordered.push_back(regBases[src]);
    regBases = std::move(reordered);

    SmallVector<std::pair<StringAttr, int32_t>> outDims;
    for (StringAttr d : ll.getOutDimNames())
      outDims.push_back({d, ll.getOutDimSize(d)});
    return LinearLayout(std::move(bases), outDims, ll.isSurjective());
  }

  // Convenience: reverse the register bases.
  LinearLayout reverseRegisterBases(const LinearLayout &ll) {
    StringAttr kRegister = StringAttr::get(&ctx, "register");
    unsigned n = ll.getBases().find(kRegister)->second.size();
    SmallVector<unsigned> perm(n);
    for (unsigned i = 0; i < n; ++i)
      perm[i] = n - 1 - i;
    return permuteRegisterBases(ll, perm);
  }

  unsigned numRegisterBases(const LinearLayout &ll) {
    StringAttr kRegister = StringAttr::get(&ctx, "register");
    return ll.getBases().find(kRegister)->second.size();
  }

  // The physical mapping a single 2D-block-load message realizes:
  // (delivery-order register, lane) -> tensor coordinate. `regPackedBases`
  // defines regMapping (delivery -> tensor register index) exactly as the LLVM
  // lowering builds it (LoadStoreOpToLLVM.cpp buildBlock2DLoadConfig);
  // composing it with the layout's register/lane sublayout yields the delivered
  // coordinate. Only the first message's registers are kept: the remaining
  // "leftover" bases are independent whole-tile repetitions whose relative
  // order is a benign load-scheduling choice. Two content-equivalent layouts
  // must realize the same mapping regardless of how their register bases are
  // listed.
  LinearLayout singleTileDeliveryMapping(const LinearLayout &ll,
                                         const BlockIOTileSizeInfo &info) {
    StringAttr kReg = StringAttr::get(&ctx, "register");
    StringAttr kLane = StringAttr::get(&ctx, "lane");
    const SetVector<unsigned> &regPackedBases = *info.regPackedBases;
    std::vector<std::vector<int32_t>> bases(regPackedBases.size());
    llvm::transform(regPackedBases, bases.begin(), [](unsigned b) {
      return std::vector<int32_t>{static_cast<int32_t>(b)};
    });
    int regSize = ll.getInDimSize(kReg);
    LinearLayout regMapping({{kReg, bases}}, {{kReg, regSize}},
                            /*requireSurjective=*/true);
    int tpw = ll.getInDimSize(kLane);
    int elemsPerTile = info.tileHeight * info.tileWidth *
                       info.numElemPerPackedVal * info.vBlocks;
    // Rounds up, like the lowering.
    int numElemsPerLoad = (elemsPerTile + tpw - 1) / tpw;
    LinearLayout tileMapping = regMapping.resizeInDim(kReg, numElemsPerLoad);
    LinearLayout ext =
        tileMapping * LinearLayout::identity1D(tpw, kLane, kLane);
    LinearLayout sub =
        ll.sublayout({kReg, kLane}, llvm::to_vector(ll.getOutDimNames()));
    return ext.compose(sub);
  }
};

// issue #7806: the 2D-block-I/O tile geometry must not depend on the order in
// which a layout happens to list its register basis vectors. A DPAS operand-A
// f16/opsPerChan=2 layout (the case confirmed in the ticket) sized to
// tileHeight=32 must still size to tileHeight=32 when its register bases are
// reversed -- otherwise the same data silently needs ~16x more load messages.
TEST_F(BlockIOUtilsTest, RegisterBaseOrderInvariance_DpasOperandA_F16) {
  auto dpas = makeDpas(/*warpsPerCTA=*/{1, 1}, /*threadsPerWarp=*/16,
                       /*opsPerChan=*/2);
  auto dot = DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, dpas, /*kWidth=*/1);
  SmallVector<int64_t> shape = {64, 32}; // M=64, K=32
  auto ll = cast<DistributedEncodingTrait>(dot).toLinearLayout(shape);

  // row-major A: memContiguousDim = rank-1 = 1 (K direction).
  auto canonical = getBlockIOLoadTileSize(ll, /*memContiguousDim=*/1,
                                          /*elemSizeInBits=*/16,
                                          /*maskAxisInfo=*/nullptr,
                                          /*oneMatrixPerLoadForBT=*/false);
  ASSERT_TRUE(canonical.isValid());

  auto reversed = getBlockIOLoadTileSize(reverseRegisterBases(ll),
                                         /*memContiguousDim=*/1,
                                         /*elemSizeInBits=*/16, nullptr, false);
  ASSERT_TRUE(reversed.isValid());

  EXPECT_EQ(reversed.tileHeight, canonical.tileHeight);
  EXPECT_EQ(reversed.tileWidth, canonical.tileWidth);
  EXPECT_EQ(reversed.vBlocks, canonical.vBlocks);
  EXPECT_EQ(reversed.numElemPerPackedVal, canonical.numElemPerPackedVal);
}

// Beyond geometry, the register mapping itself must be order-independent: the
// physical (delivery-register, lane) -> tensor mapping of a single 2D block
// load message must match the canonical one for *any* permutation of the
// register bases, so a reordered layout is not just sized like the canonical
// one but also loaded into the same registers.
TEST_F(BlockIOUtilsTest,
       RegisterBaseOrderInvariance_RegMappingEquivalent_DpasOperandA_F16) {
  auto dpas = makeDpas(/*warpsPerCTA=*/{1, 1}, /*threadsPerWarp=*/16,
                       /*opsPerChan=*/2);
  auto dot = DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, dpas, /*kWidth=*/1);
  SmallVector<int64_t> shape = {64, 32}; // M=64, K=32
  auto ll = cast<DistributedEncodingTrait>(dot).toLinearLayout(shape);

  auto canonical =
      getBlockIOLoadTileSize(ll, /*memContiguousDim=*/1,
                             /*elemSizeInBits=*/16, nullptr, false);
  ASSERT_TRUE(canonical.isValid());
  ASSERT_TRUE(canonical.regPackedBases.has_value());
  LinearLayout ref = singleTileDeliveryMapping(ll, canonical);

  unsigned n = numRegisterBases(ll);
  SmallVector<SmallVector<unsigned>> perms;
  { // reversed
    SmallVector<unsigned> p(n);
    for (unsigned i = 0; i < n; ++i)
      p[i] = n - 1 - i;
    perms.push_back(p);
  }
  { // rotate by 3
    SmallVector<unsigned> p(n);
    for (unsigned i = 0; i < n; ++i)
      p[i] = (i + 3) % n;
    perms.push_back(p);
  }
  { // swap adjacent pairs
    SmallVector<unsigned> p(n);
    for (unsigned i = 0; i < n; ++i)
      p[i] = i;
    for (unsigned i = 0; i + 1 < n; i += 2)
      std::swap(p[i], p[i + 1]);
    perms.push_back(p);
  }

  for (const auto &perm : perms) {
    LinearLayout permuted = permuteRegisterBases(ll, perm);
    auto info = getBlockIOLoadTileSize(permuted, /*memContiguousDim=*/1,
                                       /*elemSizeInBits=*/16, nullptr, false);
    ASSERT_TRUE(info.isValid());
    EXPECT_EQ(info.tileHeight, canonical.tileHeight);
    EXPECT_EQ(info.tileWidth, canonical.tileWidth);
    EXPECT_EQ(info.vBlocks, canonical.vBlocks);
    EXPECT_EQ(info.numElemPerPackedVal, canonical.numElemPerPackedVal);
    ASSERT_TRUE(info.regPackedBases.has_value());
    EXPECT_EQ(singleTileDeliveryMapping(permuted, info), ref);
  }
}

// The same order-independence must hold for the transpose (column-major B)
// path, which is a distinct set of growth loops (transpose col-grow + row grow
// bounded by MAX_BITS_WIDTH). Uses the known-valid column-major B config.
TEST_F(BlockIOUtilsTest,
       RegisterBaseOrderInvariance_RegMappingEquivalent_ColumnMajorB_F16) {
  auto dpas = makeDpas(/*warpsPerCTA=*/{4, 2}, /*threadsPerWarp=*/16,
                       /*opsPerChan=*/2);
  auto dot = DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, dpas, /*kWidth=*/2);
  SmallVector<int64_t> shape = {32, 64}; // K=32, N=64
  auto ll = cast<DistributedEncodingTrait>(dot).toLinearLayout(shape);

  // column_major: memContiguousDim = rank-2 = 0 (K direction) -> transpose.
  auto canonical =
      getBlockIOLoadTileSize(ll, /*memContiguousDim=*/0,
                             /*elemSizeInBits=*/16, nullptr, false);
  ASSERT_TRUE(canonical.isValid());
  ASSERT_TRUE(canonical.transpose);
  ASSERT_TRUE(canonical.regPackedBases.has_value());
  LinearLayout ref = singleTileDeliveryMapping(ll, canonical);

  unsigned n = numRegisterBases(ll);
  SmallVector<SmallVector<unsigned>> perms;
  { // reversed
    SmallVector<unsigned> p(n);
    for (unsigned i = 0; i < n; ++i)
      p[i] = n - 1 - i;
    perms.push_back(p);
  }
  { // swap adjacent pairs
    SmallVector<unsigned> p(n);
    for (unsigned i = 0; i < n; ++i)
      p[i] = i;
    for (unsigned i = 0; i + 1 < n; i += 2)
      std::swap(p[i], p[i + 1]);
    perms.push_back(p);
  }

  for (const auto &perm : perms) {
    LinearLayout permuted = permuteRegisterBases(ll, perm);
    auto info = getBlockIOLoadTileSize(permuted, /*memContiguousDim=*/0,
                                       /*elemSizeInBits=*/16, nullptr, false);
    ASSERT_TRUE(info.isValid());
    EXPECT_EQ(info.tileHeight, canonical.tileHeight);
    EXPECT_EQ(info.tileWidth, canonical.tileWidth);
    EXPECT_EQ(info.vBlocks, canonical.vBlocks);
    EXPECT_EQ(info.numElemPerPackedVal, canonical.numElemPerPackedVal);
    ASSERT_TRUE(info.regPackedBases.has_value());
    EXPECT_EQ(singleTileDeliveryMapping(permuted, info), ref);
  }
}

// Rank-3 order-independence (issue #7806). For a rank-3 dot operand A the outer
// (batch) dimension is folded into the base pointer, not tiled; the row
// dimension must be the inner M dimension. The register bases of a rank-3 DPAS
// layout list the batch-dim bases too, so an order-sensitive row-dimension
// choice can wrongly pick the batch dim (collapsing tileHeight). Verify the
// expected geometry and that it is invariant to register-base order.
TEST_F(BlockIOUtilsTest, RegisterBaseOrderInvariance_Rank3DpasOperandA_F16) {
  auto dpas =
      DpasEncodingAttr::get(&ctx, /*repeatCount=*/8, /*systolicDepth=*/8,
                            /*executionSize=*/16, /*opsPerChan=*/2,
                            /*warpsPerCTA=*/{1, 4, 2},
                            /*repCluster=*/{1, 1, 1},
                            /*threadsPerWarp=*/16, std::nullopt);
  auto dot = DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, dpas, /*kWidth=*/1);
  SmallVector<int64_t> shape = {4, 64, 32}; // batch=4, M=64, K=32
  auto ll = cast<DistributedEncodingTrait>(dot).toLinearLayout(shape);

  // row-major: memContiguousDim = rank-1 = 2 (K direction).
  auto canonical =
      getBlockIOLoadTileSize(ll, /*memContiguousDim=*/2,
                             /*elemSizeInBits=*/16, nullptr, false);
  ASSERT_TRUE(canonical.isValid());
  // The row dimension is the inner M dim (1), never the batch dim (0).
  EXPECT_EQ(canonical.rowDim, 1);
  EXPECT_EQ(canonical.colDim, 2);
  EXPECT_EQ(canonical.tileHeight, 8);
  EXPECT_EQ(canonical.tileWidth, 16);
  EXPECT_EQ(canonical.vBlocks, 2);
  LinearLayout ref = singleTileDeliveryMapping(ll, canonical);

  LinearLayout reversed = reverseRegisterBases(ll);
  auto info = getBlockIOLoadTileSize(reversed, /*memContiguousDim=*/2,
                                     /*elemSizeInBits=*/16, nullptr, false);
  ASSERT_TRUE(info.isValid());
  EXPECT_EQ(info.rowDim, canonical.rowDim);
  EXPECT_EQ(info.colDim, canonical.colDim);
  EXPECT_EQ(info.tileHeight, canonical.tileHeight);
  EXPECT_EQ(info.tileWidth, canonical.tileWidth);
  EXPECT_EQ(info.vBlocks, canonical.vBlocks);
  ASSERT_TRUE(info.regPackedBases.has_value());
  EXPECT_EQ(singleTileDeliveryMapping(reversed, info), ref);
}

// A transposed tile reads its rows along the memory contiguous dimension, so
// the row dimension must be that dimension -- the check at the end of
// getBlockIOTileSize rejects any other. Discovering the row dimension by
// looking for a unit-stride register base can pick a different dimension (the
// memory contiguous one may own only wider strides), which turned a tile the
// transpose path can express into a scatter fallback. Rank 3 is needed to
// expose it: at rank 2 there is only one candidate dimension.
TEST_F(BlockIOUtilsTest, TransposeRowDimIsMemContiguousDim_Rank3Blocked) {
  auto cga = CGAEncodingAttr::get1CTALayout(&ctx, 3);
  SmallVector<int64_t> shape = {32, 64, 64};

  // The lanes run along dim 2, so dim 2 is the column dim; memContiguousDim 0
  // makes this a transpose. Dim 1 owns the unit-stride register base and dim 0
  // only wider ones, so a unit-stride search picks dim 1 and the tile is
  // rejected.
  auto blocked = BlockedEncodingAttr::get(&ctx, /*sizePerThread=*/{1, 1, 1},
                                          /*threadsPerWarp=*/{1, 1, 16},
                                          /*warpsPerCTA=*/{1, 1, 1},
                                          /*order=*/{0, 1, 2}, cga);
  auto ll = cast<DistributedEncodingTrait>(blocked).toLinearLayout(shape);
  auto info = getBlockIOLoadTileSize(ll, /*memContiguousDim=*/0,
                                     /*elemSizeInBits=*/32, nullptr, false);
  ASSERT_TRUE(info.isValid());
  EXPECT_TRUE(info.transpose);
  EXPECT_EQ(info.rowDim, 0);
  EXPECT_EQ(info.tileHeight, 32);
  EXPECT_EQ(info.tileWidth, 8);

  // Same, with the canonical row-major order and a memory contiguous dim that
  // owns no register base at all: the tile is still valid, one column wide.
  auto blocked2 = BlockedEncodingAttr::get(&ctx, /*sizePerThread=*/{1, 1, 1},
                                           /*threadsPerWarp=*/{1, 1, 16},
                                           /*warpsPerCTA=*/{1, 2, 2},
                                           /*order=*/{2, 1, 0}, cga);
  auto ll2 = cast<DistributedEncodingTrait>(blocked2).toLinearLayout(shape);
  auto info2 = getBlockIOLoadTileSize(ll2, /*memContiguousDim=*/1,
                                      /*elemSizeInBits=*/32, nullptr, false);
  ASSERT_TRUE(info2.isValid());
  EXPECT_TRUE(info2.transpose);
  EXPECT_EQ(info2.rowDim, 1);
  EXPECT_EQ(info2.tileHeight, 16);
  EXPECT_EQ(info2.tileWidth, 1);
}

// When the lanes leave the row dimension open, it is the dimension whose
// register bases give the tallest tile, not the innermost one that has a
// unit-stride register base. Here dim 1 has one (height 2) and dim 0 has five
// (height 32): preferring the inner dim needs 16x the messages.
TEST_F(BlockIOUtilsTest, RowDimIsTallest_Rank3Blocked) {
  auto cga = CGAEncodingAttr::get1CTALayout(&ctx, 3);
  SmallVector<int64_t> shape = {32, 2, 128};
  auto blocked = BlockedEncodingAttr::get(&ctx, /*sizePerThread=*/{1, 1, 2},
                                          /*threadsPerWarp=*/{1, 1, 16},
                                          /*warpsPerCTA=*/{1, 1, 4},
                                          /*order=*/{2, 0, 1}, cga);
  auto ll = cast<DistributedEncodingTrait>(blocked).toLinearLayout(shape);
  auto info = getBlockIOLoadTileSize(ll, /*memContiguousDim=*/2,
                                     /*elemSizeInBits=*/16, nullptr, false);
  ASSERT_TRUE(info.isValid());
  EXPECT_FALSE(info.transpose);
  EXPECT_EQ(info.rowDim, 0);
  EXPECT_EQ(info.colDim, 2);
  EXPECT_EQ(info.tileHeight, 32);

  // The choice is by geometry, so listing the register bases differently does
  // not change it.
  auto reversed = getBlockIOLoadTileSize(reverseRegisterBases(ll),
                                         /*memContiguousDim=*/2,
                                         /*elemSizeInBits=*/16, nullptr, false);
  ASSERT_TRUE(reversed.isValid());
  EXPECT_EQ(reversed.rowDim, 0);
  EXPECT_EQ(reversed.tileHeight, 32);

  // A descriptor load can only tile its inner two dims, so it gets dim 1.
  auto inner = getBlockIOLoadTileSize(ll, /*memContiguousDim=*/2,
                                      /*elemSizeInBits=*/16, nullptr, false,
                                      /*innerRowDimOnly=*/true);
  ASSERT_TRUE(inner.isValid());
  EXPECT_EQ(inner.rowDim, 1);
  EXPECT_EQ(inner.tileHeight, 2);
}

// innerRowDimOnly keeps the row dimension off the batch dim even when the
// inner dim cannot grow at all. The warps split dim 1, so its only register
// base has stride 2, and the batch dim owns the unit-stride one.
TEST_F(BlockIOUtilsTest, InnerRowDimOnly_Rank3Blocked) {
  auto cga = CGAEncodingAttr::get1CTALayout(&ctx, 3);
  SmallVector<int64_t> shape = {2, 4, 16};
  auto blocked = BlockedEncodingAttr::get(&ctx, /*sizePerThread=*/{1, 1, 1},
                                          /*threadsPerWarp=*/{1, 1, 16},
                                          /*warpsPerCTA=*/{1, 2, 1},
                                          /*order=*/{2, 1, 0}, cga);
  auto ll = cast<DistributedEncodingTrait>(blocked).toLinearLayout(shape);
  auto f16 = Float16Type::get(&ctx);
  auto tensorType = RankedTensorType::get(shape, f16, blocked);

  // A pointer load can tile across the batch dim.
  auto info = getBlockIOLoadTileSize(ll, /*memContiguousDim=*/2,
                                     /*elemSizeInBits=*/16, nullptr, false);
  ASSERT_TRUE(info.isValid());
  EXPECT_EQ(info.rowDim, 0);
  EXPECT_EQ(info.tileHeight, 2);

  auto inner = getBlockIOLoadTileSize(ll, /*memContiguousDim=*/2,
                                      /*elemSizeInBits=*/16, nullptr, false,
                                      /*innerRowDimOnly=*/true);
  ASSERT_TRUE(inner.isValid());
  EXPECT_EQ(inner.rowDim, 1);
  EXPECT_EQ(inner.tileHeight, 1);
  EXPECT_TRUE(validate2DBlockLoadTile(ll, /*memContiguousDim=*/2,
                                      /*elemSizeInBits=*/16, tensorType,
                                      /*oneMatrixPerLoadForBT=*/false,
                                      /*maskAxisInfo=*/nullptr,
                                      /*innerRowDimOnly=*/true));
}

// The mask bounds how tall each candidate row dim can grow, so it takes part
// in the choice. Same layout as RowDimIsTallest_Rank3Blocked, where dim 0
// wins without a mask.
TEST_F(BlockIOUtilsTest, RowDimIsTallestUnderMask_Rank3Blocked) {
  auto cga = CGAEncodingAttr::get1CTALayout(&ctx, 3);
  SmallVector<int64_t> shape = {32, 2, 128};
  auto blocked = BlockedEncodingAttr::get(&ctx, /*sizePerThread=*/{1, 1, 2},
                                          /*threadsPerWarp=*/{1, 1, 16},
                                          /*warpsPerCTA=*/{1, 1, 4},
                                          /*order=*/{2, 0, 1}, cga);
  auto ll = cast<DistributedEncodingTrait>(blocked).toLinearLayout(shape);

  // A mask constant over only 2 rows of dim 0 caps it at dim 1's height, and
  // on a tie the inner dim wins.
  AxisInfo capped(/*contiguity=*/{1, 1, 1}, /*divisibility=*/{1, 1, 1},
                  /*constancy=*/{2, 2, 128});
  auto info = getBlockIOLoadTileSize(ll, /*memContiguousDim=*/2,
                                     /*elemSizeInBits=*/16, &capped, false);
  ASSERT_TRUE(info.isValid());
  EXPECT_EQ(info.rowDim, 1);
  EXPECT_EQ(info.tileHeight, 2);

  // A dim whose mask constancy is not a power of 2 cannot be the row dim, so
  // it is passed over rather than rejecting the whole tile. Here no other dim
  // grows either, and the tile is one row tall.
  AxisInfo nonPow2(/*contiguity=*/{1, 1, 1}, /*divisibility=*/{1, 1, 1},
                   /*constancy=*/{3, 1, 128});
  auto info2 = getBlockIOLoadTileSize(ll, /*memContiguousDim=*/2,
                                      /*elemSizeInBits=*/16, &nonPow2, false);
  ASSERT_TRUE(info2.isValid());
  EXPECT_EQ(info2.rowDim, 1);
  EXPECT_EQ(info2.tileHeight, 1);
}

// The cost model sizes a descriptor load's tile on the inner two dims, as its
// lowering does, while a pointer load of the same type tiles across the batch
// dim. Layout as in InnerRowDimOnly_Rank3Blocked.
TEST_F(BlockIOUtilsTest, EstimateLoadHWCost_DescriptorTilesInnerDims) {
  ctx.getOrLoadDialect<TritonDialect>();
  const char *ir = R"mlir(
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 16], warpsPerCTA = [1, 2, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @f(%desc: !tt.tensordesc<2x4x16xf16>, %ptrs: tensor<2x4x16x!tt.ptr<f16>, #blocked>, %i: i32) {
    %0 = tt.descriptor_load %desc[%i, %i, %i] {ttig.block_io = "row_major"} : !tt.tensordesc<2x4x16xf16> -> tensor<2x4x16xf16, #blocked>
    %1 = tt.load %ptrs {ttig.block_io = "row_major"} : tensor<2x4x16x!tt.ptr<f16>, #blocked>
    tt.return
  }
})mlir";
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  ASSERT_TRUE(module);
  DescriptorLoadOp descLoad;
  LoadOp ptrLoad;
  module->walk([&](DescriptorLoadOp op) { descLoad = op; });
  module->walk([&](LoadOp op) { ptrLoad = op; });
  ASSERT_TRUE(descLoad && ptrLoad);

  // The 4 rows of dim 1, one per message.
  EXPECT_EQ(
      estimateLoadHWCost(cast<RankedTensorType>(descLoad.getType()), descLoad),
      4u);
  // The 2 rows of the batch dim in one message.
  EXPECT_EQ(
      estimateLoadHWCost(cast<RankedTensorType>(ptrLoad.getType()), ptrLoad),
      1u);
}

// Test that validate2DBlockLoadTile accepts a column-major B matrix load with
// sub-group-size=32 and f16/opsPerChan=2.
//
// Old code: computeTransposeShuffleMapping failed because
//   numPackedVals(=2) > 1 && dpasInstShapeB()[1](=16) != threadsPerWarp(=32)
// New code: linear-layout comparison succeeds for this configuration.
TEST_F(BlockIOUtilsTest, ColumnMajorB_Tpw32_F16_Accepted) {
  auto dpas = makeDpas(/*warpsPerCTA=*/{2, 2}, /*threadsPerWarp=*/32,
                       /*opsPerChan=*/2);
  auto dot = DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, dpas, /*kWidth=*/2);
  auto f16 = Float16Type::get(&ctx);
  SmallVector<int64_t> shape = {32, 64}; // K=32, N=64
  auto tensorType = RankedTensorType::get(shape, f16, dot);
  auto ll = cast<DistributedEncodingTrait>(dot).toLinearLayout(shape);
  // column_major: memContiguousDim = rank-2 = 0 (K direction)
  EXPECT_TRUE(validate2DBlockLoadTile(ll, /*memContiguousDim=*/0,
                                      /*elemSizeInBits=*/16, tensorType));
}

// Regression test: validate2DBlockLoadTile must still accept a column-major B
// matrix load with sub-group-size=16 and f16/opsPerChan=2 after PR #7487.
TEST_F(BlockIOUtilsTest, ColumnMajorB_Tpw16_F16_Accepted) {
  auto dpas = makeDpas(/*warpsPerCTA=*/{4, 2}, /*threadsPerWarp=*/16,
                       /*opsPerChan=*/2);
  auto dot = DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, dpas, /*kWidth=*/2);
  auto f16 = Float16Type::get(&ctx);
  SmallVector<int64_t> shape = {32, 64}; // K=32, N=64
  auto tensorType = RankedTensorType::get(shape, f16, dot);
  auto ll = cast<DistributedEncodingTrait>(dot).toLinearLayout(shape);
  // column_major: memContiguousDim = rank-2 = 0 (K direction)
  EXPECT_TRUE(validate2DBlockLoadTile(ll, /*memContiguousDim=*/0,
                                      /*elemSizeInBits=*/16, tensorType));
}

// pickBlockIOLoadEncoding: a blocked layout for a non-dot descriptor load
// that 2D block reads can serve. Each input carries the Coalesce layout.

// reducedDim 0: warps split the columns, so the reduction stays in a warp.
// This is the layout measured for issue #8458.
TEST_F(BlockIOUtilsTest, PickF32Tpw32Warps4ReduceRows) {
  auto type = makeCoalescedType({16, 256}, Float32Type::get(&ctx),
                                /*numWarps=*/4, /*threadsPerWarp=*/32);
  auto enc = pickBlockIOLoadEncoding(type, /*numWarps=*/4,
                                     /*threadsPerWarp=*/32, /*reducedDim=*/0);
  ASSERT_TRUE(enc.has_value());
  EXPECT_EQ(enc->getSizePerThread(), ArrayRef<unsigned>({1, 1}));
  EXPECT_EQ(enc->getThreadsPerWarp(), ArrayRef<unsigned>({2, 16}));
  EXPECT_EQ(enc->getWarpsPerCTA(), ArrayRef<unsigned>({1, 4}));
  EXPECT_EQ(enc->getOrder(), ArrayRef<unsigned>({1, 0}));
  EXPECT_EQ(enc->getCGALayout(), getCGALayout(type.getEncoding()));
  EXPECT_TRUE(lowersTo2DBlockLoad(type, *enc));
}

// 16 lanes x f32 already fill the 64-byte row, so lanes take one row.
TEST_F(BlockIOUtilsTest, PickF32Tpw16Warps1ReduceRows) {
  auto type = makeCoalescedType({16, 256}, Float32Type::get(&ctx),
                                /*numWarps=*/1, /*threadsPerWarp=*/16);
  auto enc = pickBlockIOLoadEncoding(type, /*numWarps=*/1,
                                     /*threadsPerWarp=*/16, /*reducedDim=*/0);
  ASSERT_TRUE(enc.has_value());
  EXPECT_EQ(enc->getSizePerThread(), ArrayRef<unsigned>({1, 1}));
  EXPECT_EQ(enc->getThreadsPerWarp(), ArrayRef<unsigned>({1, 16}));
  EXPECT_EQ(enc->getWarpsPerCTA(), ArrayRef<unsigned>({1, 1}));
  EXPECT_EQ(enc->getOrder(), ArrayRef<unsigned>({1, 0}));
  EXPECT_EQ(enc->getCGALayout(), getCGALayout(type.getEncoding()));
  EXPECT_TRUE(lowersTo2DBlockLoad(type, *enc));
}

// reducedDim 1: warps split the rows, so the reduction stays in a warp.
TEST_F(BlockIOUtilsTest, PickF32ReduceCols) {
  auto type = makeCoalescedType({64, 64}, Float32Type::get(&ctx),
                                /*numWarps=*/4, /*threadsPerWarp=*/32);
  auto enc = pickBlockIOLoadEncoding(type, /*numWarps=*/4,
                                     /*threadsPerWarp=*/32, /*reducedDim=*/1);
  ASSERT_TRUE(enc.has_value());
  EXPECT_EQ(enc->getSizePerThread(), ArrayRef<unsigned>({1, 1}));
  EXPECT_EQ(enc->getThreadsPerWarp(), ArrayRef<unsigned>({2, 16}));
  EXPECT_EQ(enc->getWarpsPerCTA(), ArrayRef<unsigned>({4, 1}));
  EXPECT_EQ(enc->getOrder(), ArrayRef<unsigned>({1, 0}));
  EXPECT_EQ(enc->getCGALayout(), getCGALayout(type.getEncoding()));
  EXPECT_TRUE(lowersTo2DBlockLoad(type, *enc));
}

// No reduction: warps split the columns. 32 lanes x f16 fill a 64-byte row.
TEST_F(BlockIOUtilsTest, PickF16Tpw32NoReduce) {
  auto type = makeCoalescedType({16, 256}, Float16Type::get(&ctx),
                                /*numWarps=*/4, /*threadsPerWarp=*/32);
  auto enc =
      pickBlockIOLoadEncoding(type, /*numWarps=*/4, /*threadsPerWarp=*/32,
                              /*reducedDim=*/std::nullopt);
  ASSERT_TRUE(enc.has_value());
  EXPECT_EQ(enc->getSizePerThread(), ArrayRef<unsigned>({1, 1}));
  EXPECT_EQ(enc->getThreadsPerWarp(), ArrayRef<unsigned>({1, 32}));
  EXPECT_EQ(enc->getWarpsPerCTA(), ArrayRef<unsigned>({1, 4}));
  EXPECT_EQ(enc->getOrder(), ArrayRef<unsigned>({1, 0}));
  EXPECT_EQ(enc->getCGALayout(), getCGALayout(type.getEncoding()));
  EXPECT_TRUE(lowersTo2DBlockLoad(type, *enc));
}

// The rejections hold for every reducedDim.
constexpr std::optional<unsigned> kReducedDims[] = {std::nullopt, 0u, 1u};

std::string reducedDimName(std::optional<unsigned> reducedDim) {
  return reducedDim ? std::to_string(*reducedDim) : "none";
}

// Only rank-2 loads are picked. At 1 warp x 16 lanes, the first two dims
// (2x16) pass every geometry check, so only the rank check rejects it. At the
// neighbours' 4 warps x 32 lanes, the replication guard would reject 2x16 too
// (see PickRejectsTooSmall) and mask the rank check, hence 1 warp x 16 lanes.
TEST_F(BlockIOUtilsTest, PickRejectsRank3) {
  auto type = makeCoalescedType({2, 16, 256}, Float32Type::get(&ctx),
                                /*numWarps=*/1, /*threadsPerWarp=*/16);
  for (std::optional<unsigned> reducedDim : kReducedDims)
    EXPECT_FALSE(pickBlockIOLoadEncoding(type, /*numWarps=*/1,
                                         /*threadsPerWarp=*/16, reducedDim)
                     .has_value())
        << "reducedDim=" << reducedDimName(reducedDim);
}

// 2x16 f32 is a single 2x16 lane tile, so 4 warps would replicate it.
TEST_F(BlockIOUtilsTest, PickRejectsTooSmall) {
  auto type = makeCoalescedType({2, 16}, Float32Type::get(&ctx),
                                /*numWarps=*/4, /*threadsPerWarp=*/32);
  for (std::optional<unsigned> reducedDim : kReducedDims)
    EXPECT_FALSE(pickBlockIOLoadEncoding(type, /*numWarps=*/4,
                                         /*threadsPerWarp=*/32, reducedDim)
                     .has_value())
        << "reducedDim=" << reducedDimName(reducedDim);
}

// i1 and i4 are below 8 bits; !tt.ptr has no int/float bit width.
TEST_F(BlockIOUtilsTest, PickRejectsSubByteAndPointer) {
  std::pair<const char *, Type> elemTypes[] = {
      {"i1", IntegerType::get(&ctx, 1)},
      {"i4", IntegerType::get(&ctx, 4)},
      {"!tt.ptr<f32>", triton::PointerType::get(Float32Type::get(&ctx))}};
  for (auto [name, elemTy] : elemTypes) {
    auto type = makeCoalescedType({16, 256}, elemTy, /*numWarps=*/4,
                                  /*threadsPerWarp=*/32);
    for (std::optional<unsigned> reducedDim : kReducedDims)
      EXPECT_FALSE(pickBlockIOLoadEncoding(type, /*numWarps=*/4,
                                           /*threadsPerWarp=*/32, reducedDim)
                       .has_value())
          << name << ", reducedDim=" << reducedDimName(reducedDim);
  }
}

// Builds the `(row_base + arange(0, N)) < bound` row-limit mask IR by hand
// and calls matchRowLimitMask directly on the constructed Value.
class MatchRowLimitMaskTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<TritonDialect>();
    module = ModuleOp::create(UnknownLoc::get(&ctx));
    builder.setInsertionPointToStart(module->getBody());
  }

  Location loc() { return builder.getUnknownLoc(); }

  Value constI(int64_t value, unsigned width) {
    return arith::ConstantIntOp::create(builder, loc(),
                                        builder.getIntegerType(width), value);
  }

  // tensor<N x i32> via tt.make_range(0, N).
  Value arange(int64_t n) {
    auto ty = RankedTensorType::get({n}, builder.getI32Type());
    return MakeRangeOp::create(builder, loc(), ty, 0, static_cast<uint32_t>(n));
  }

  Value expandDims(Value v, unsigned axis, ArrayRef<int64_t> resultShape) {
    auto srcTy = cast<RankedTensorType>(v.getType());
    auto resTy = RankedTensorType::get(resultShape, srcTy.getElementType());
    return ExpandDimsOp::create(builder, loc(), resTy, v, axis);
  }

  Value broadcastTo(Value v, ArrayRef<int64_t> resultShape) {
    auto srcTy = cast<RankedTensorType>(v.getType());
    auto resTy = RankedTensorType::get(resultShape, srcTy.getElementType());
    return BroadcastOp::create(builder, loc(), resTy, v);
  }

  Value splatTo(Value scalar, ArrayRef<int64_t> shape) {
    auto resTy = RankedTensorType::get(shape, scalar.getType());
    return SplatOp::create(builder, loc(), resTy, scalar);
  }

  Value cmpSlt(Value lhs, Value rhs) {
    return arith::CmpIOp::create(builder, loc(), arith::CmpIPredicate::slt, lhs,
                                 rhs);
  }

  Value cmpUlt(Value lhs, Value rhs) {
    return arith::CmpIOp::create(builder, loc(), arith::CmpIPredicate::ult, lhs,
                                 rhs);
  }

  Value cmpSgt(Value lhs, Value rhs) {
    return arith::CmpIOp::create(builder, loc(), arith::CmpIPredicate::sgt, lhs,
                                 rhs);
  }

  Value addI32(Value lhs, Value rhs) {
    return arith::AddIOp::create(builder, loc(), lhs, rhs);
  }

  Value mulI32(Value lhs, Value rhs) {
    return arith::MulIOp::create(builder, loc(), lhs, rhs);
  }

  Value denseBoolConst(ArrayRef<int64_t> shape, bool value) {
    auto ty = RankedTensorType::get(shape, builder.getI1Type());
    return arith::ConstantOp::create(
        builder, loc(), ty,
        DenseElementsAttr::get(ty, builder.getBoolAttr(value)));
  }

  Value denseIntSplat(ArrayRef<int64_t> shape, int64_t value) {
    auto ty = RankedTensorType::get(shape, builder.getI32Type());
    return arith::ConstantOp::create(
        builder, loc(), ty,
        DenseElementsAttr::get(ty, builder.getI32IntegerAttr(value)));
  }

  Value denseFloatSplat(ArrayRef<int64_t> shape, double value) {
    auto ty = RankedTensorType::get(shape, builder.getF32Type());
    return arith::ConstantOp::create(
        builder, loc(), ty,
        DenseElementsAttr::get(ty, builder.getF32FloatAttr(value)));
  }

  // The canonical row-limit mask: `(rowBaseScalar + arange(0, boundedExtent))
  // < boundScalar`, expanded along `boundedDim` and broadcast to
  // `tensorShape`.
  Value buildRowLimitMask(Value rowIndex1D, unsigned boundedDim,
                          ArrayRef<int64_t> tensorShape, Value boundScalar) {
    int64_t boundedExtent = tensorShape[boundedDim];
    SmallVector<int64_t, 2> narrowShape = {1, 1};
    narrowShape[boundedDim] = boundedExtent;
    Value rowIndex2D =
        expandDims(rowIndex1D, /*axis=*/1 - boundedDim, narrowShape);
    Value boundSplat2D = splatTo(boundScalar, narrowShape);
    Value cmp = cmpSlt(rowIndex2D, boundSplat2D);
    return broadcastTo(cmp, tensorShape);
  }

  // Expands a 1D tensor along `varyDim` up to `narrowShape`'s rank, with
  // size-1 dims elsewhere.
  Value expandToShape(Value v, unsigned varyDim,
                      ArrayRef<int64_t> narrowShape) {
    for (unsigned d = 0; d < varyDim; ++d) {
      auto curShape = cast<RankedTensorType>(v.getType()).getShape();
      SmallVector<int64_t> newShape{1};
      newShape.append(curShape.begin(), curShape.end());
      v = expandDims(v, /*axis=*/0, newShape);
    }
    for (unsigned d = varyDim + 1; d < narrowShape.size(); ++d) {
      auto curShape = cast<RankedTensorType>(v.getType()).getShape();
      SmallVector<int64_t> newShape(curShape.begin(), curShape.end());
      newShape.push_back(1);
      v = expandDims(v, /*axis=*/newShape.size() - 1, newShape);
    }
    return v;
  }

  // Generalizes `buildRowLimitMask` to any rank: varies only along
  // `varyDim`, degenerate elsewhere, broadcast to `tensorShape`. Used for
  // both the row-limit comparison and a batch-only predicate ANDed with it.
  Value buildBoundedMask(Value index1D, unsigned varyDim,
                         ArrayRef<int64_t> tensorShape, Value boundScalar) {
    SmallVector<int64_t> narrowShape(tensorShape.size(), 1);
    narrowShape[varyDim] = tensorShape[varyDim];
    Value indexND = expandToShape(index1D, varyDim, narrowShape);
    Value boundSplatND = splatTo(boundScalar, narrowShape);
    Value cmp = cmpSlt(indexND, boundSplatND);
    return broadcastTo(cmp, tensorShape);
  }

  Value andI(Value lhs, Value rhs) {
    return arith::AndIOp::create(builder, loc(), lhs, rhs);
  }

protected:
  MLIRContext ctx;
  OpBuilder builder{&ctx};
  OwningOpRef<ModuleOp> module;
};

TEST_F(MatchRowLimitMaskTest, BareArangeHasNullRowBase) {
  // `arange(0, N) < bound`, with no explicit base (implicit zero).
  SmallVector<int64_t, 2> shape = {256, 32};
  Value bound = constI(100, 32);
  Value mask = buildRowLimitMask(arange(256), /*boundedDim=*/0, shape, bound);

  auto result = matchRowLimitMask(mask, /*other=*/Value(), shape,
                                  /*contiguousDim=*/1,
                                  /*pitchDimIsBroadcast=*/false);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->rowBase);
  EXPECT_EQ(result->bound, bound);
}

// pitchDimIsBroadcast=true rejects outright, before inspecting the mask.
TEST_F(MatchRowLimitMaskTest, PitchDimBroadcastRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value mask =
      buildRowLimitMask(arange(256), /*boundedDim=*/0, shape, constI(100, 32));
  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/true)
                   .has_value());
}

// Same mask, but contiguousDim=0 makes dim 0 the (wrong) contiguous dim.
TEST_F(MatchRowLimitMaskTest, BoundOnContiguousDimRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value mask =
      buildRowLimitMask(arange(256), /*boundedDim=*/0, shape, constI(100, 32));
  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/0,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// Any value that isn't a zero splat (or absent) fails isZeroFillOther. Uses a
// dense splat (not a bare scalar constant) so the integer branch of
// isZeroFillOther is actually exercised, rather than being rejected earlier
// by the splat-attribute check.
TEST_F(MatchRowLimitMaskTest, NonZeroFillOtherRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value mask =
      buildRowLimitMask(arange(256), /*boundedDim=*/0, shape, constI(100, 32));
  Value nonZeroOther = denseIntSplat(shape, /*value=*/1);
  EXPECT_FALSE(matchRowLimitMask(mask, nonZeroOther, shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// `-0.0` is a zero-valued float splat, but isZeroFillOther's sign check
// must still reject it: the HW boundary check only ever produces +0.0, and
// -0.0 would flip the sign bit of masked-out elements.
TEST_F(MatchRowLimitMaskTest, NegativeZeroFillOtherRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value mask =
      buildRowLimitMask(arange(256), /*boundedDim=*/0, shape, constI(100, 32));
  Value negZeroOther = denseFloatSplat(shape, -0.0);
  EXPECT_FALSE(matchRowLimitMask(mask, negZeroOther, shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// mask isn't a tt.broadcast result at all.
TEST_F(MatchRowLimitMaskTest, MaskNotFromBroadcastRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value mask = denseBoolConst(shape, /*value=*/true);
  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// A <1x1> scalar predicate broadcast to the full tile: neither dim kept
// full extent, so boundedDim can't be determined.
TEST_F(MatchRowLimitMaskTest, BothDimsDegenerateBoundedDimUndetermined) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value a = splatTo(constI(3, 32), {1, 1});
  Value b = splatTo(constI(5, 32), {1, 1});
  Value cmp = cmpSlt(a, b);
  Value mask = broadcastTo(cmp, shape);

  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// ult instead of slt.
TEST_F(MatchRowLimitMaskTest, UnsignedPredicateRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowIndex2D = expandDims(arange(256), /*axis=*/1, {256, 1});
  Value boundSplat2D = splatTo(constI(100, 32), {256, 1});
  Value cmp = cmpUlt(rowIndex2D, boundSplat2D);
  Value mask = broadcastTo(cmp, shape);

  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// A per-row bound that isn't a uniform tt.splat scalar.
TEST_F(MatchRowLimitMaskTest, NonSplatBoundRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowIndex2D = expandDims(arange(256), /*axis=*/1, {256, 1});
  Value nonSplatBound2D = expandDims(arange(256), /*axis=*/1, {256, 1});
  Value cmp = cmpSlt(rowIndex2D, nonSplatBound2D);
  Value mask = broadcastTo(cmp, shape);

  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// 2 * arange is neither a bare arange nor base + arange.
TEST_F(MatchRowLimitMaskTest, NonArangeIndexRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value scaled = mulI32(arange(256), splatTo(constI(2, 32), {256}));
  Value mask =
      buildRowLimitMask(scaled, /*boundedDim=*/0, shape, constI(100, 32));

  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// `sext(arange(0, N)) < bound64`, with no explicit base: matches with a null
// `rowBase` and the i64 bound.
TEST_F(MatchRowLimitMaskTest, I64BareArangeHasNullRowBase) {
  SmallVector<int64_t, 2> shape = {256, 32};
  auto i64TensorTy = RankedTensorType::get({256}, builder.getI64Type());
  Value idx64 =
      arith::ExtSIOp::create(builder, loc(), i64TensorTy, arange(256));
  Value bound64 = constI(100, 64);
  Value mask = buildRowLimitMask(idx64, /*boundedDim=*/0, shape, bound64);

  auto result = matchRowLimitMask(mask, /*other=*/Value(), shape,
                                  /*contiguousDim=*/1,
                                  /*pitchDimIsBroadcast=*/false);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->rowBase);
  EXPECT_EQ(result->bound, bound64);
}

// An i64 row index (issue #8565: Helion computes it with `.to(tl.int64)`):
// `(rowBase64 + sext(arange)) < bound64` matches, with i64 `rowBase`/`bound`.
TEST_F(MatchRowLimitMaskTest, I64RowIndexAccepted) {
  SmallVector<int64_t, 2> shape = {256, 32};
  auto i64TensorTy = RankedTensorType::get({256}, builder.getI64Type());
  Value idx64 =
      arith::ExtSIOp::create(builder, loc(), i64TensorTy, arange(256));
  Value rowBase64 = constI(7, 64);
  Value rows64 = addI32(splatTo(rowBase64, {256}), idx64);
  Value bound64 = constI(100, 64);
  Value mask = buildRowLimitMask(rows64, /*boundedDim=*/0, shape, bound64);

  auto result = matchRowLimitMask(mask, /*other=*/Value(), shape,
                                  /*contiguousDim=*/1,
                                  /*pitchDimIsBroadcast=*/false);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBase, rowBase64);
  EXPECT_EQ(result->bound, bound64);
}

// The shape Helion's grouped GEMM emits: an i64 base summed from
// sign-extended i32 scalars, and an i32 bound sign-extended before the splat.
TEST_F(MatchRowLimitMaskTest, I64RowIndexFromExtendedI32Accepted) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Type i64Ty = builder.getI64Type();
  auto i64TensorTy = RankedTensorType::get({256}, i64Ty);
  Value groupStart =
      arith::ExtSIOp::create(builder, loc(), i64Ty, constI(4096, 32));
  Value tileOffset =
      arith::ExtSIOp::create(builder, loc(), i64Ty, constI(256, 32));
  Value rowBase64 = addI32(groupStart, tileOffset);
  Value idx64 =
      arith::ExtSIOp::create(builder, loc(), i64TensorTy, arange(256));
  Value rows64 = addI32(splatTo(rowBase64, {256}), idx64);
  Value bound64 =
      arith::ExtSIOp::create(builder, loc(), i64Ty, constI(8192, 32));
  Value mask = buildRowLimitMask(rows64, /*boundedDim=*/0, shape, bound64);

  auto result = matchRowLimitMask(mask, /*other=*/Value(), shape,
                                  /*contiguousDim=*/1,
                                  /*pitchDimIsBroadcast=*/false);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBase, rowBase64);
  EXPECT_EQ(result->bound, bound64);
}

// An extension between an i32 base and an i64 compare is rejected: the
// matcher looks through it, so the height would be computed from the i32 base
// with the wrong extension (base 0x80000000 under `extui` is +2^31, but
// computeRowLimitHeight sign-extends it to -2^31 and returns the full tile).
TEST_F(MatchRowLimitMaskTest, I32BaseExtendedToI64CompareRejected) {
  SmallVector<int64_t, 2> shape = {64, 32};
  Type i64Ty = builder.getI64Type();
  auto i64TensorTy = RankedTensorType::get({64}, i64Ty);
  Value bound64 = constI(2147483664, 64);

  // extui(base32 + arange) < bound64, extension around the add.
  {
    Value rows32 = addI32(splatTo(constI(INT32_MIN, 32), {64}), arange(64));
    Value rows64 = arith::ExtUIOp::create(builder, loc(), i64TensorTy, rows32);
    Value mask = buildRowLimitMask(rows64, /*boundedDim=*/0, shape, bound64);
    EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                   /*contiguousDim=*/1,
                                   /*pitchDimIsBroadcast=*/false)
                     .has_value());
  }
  // extui(splat(base32)) + sext(arange) < bound64, extension on the splat.
  {
    Value base64 = arith::ExtUIOp::create(builder, loc(), i64TensorTy,
                                          splatTo(constI(INT32_MIN, 32), {64}));
    Value idx64 =
        arith::ExtSIOp::create(builder, loc(), i64TensorTy, arange(64));
    Value mask = buildRowLimitMask(addI32(base64, idx64), /*boundedDim=*/0,
                                   shape, bound64);
    EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                   /*contiguousDim=*/1,
                                   /*pitchDimIsBroadcast=*/false)
                     .has_value());
  }
  // extsi(base32 + arange) < bound64: conservatively rejected too.
  {
    Value rows32 = addI32(splatTo(constI(7, 32), {64}), arange(64));
    Value rows64 = arith::ExtSIOp::create(builder, loc(), i64TensorTy, rows32);
    Value mask =
        buildRowLimitMask(rows64, /*boundedDim=*/0, shape, constI(100, 64));
    EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                   /*contiguousDim=*/1,
                                   /*pitchDimIsBroadcast=*/false)
                     .has_value());
  }
}

// A rank-3 tensor (batch dim 0, pitch dim 1, contiguous dim 2) with just the
// row-limit comparison, no batch predicate: generalizes the rank-2 case,
// `extraPred` stays null.
TEST_F(MatchRowLimitMaskTest, RankThreeRowLimitOnlyAccepted) {
  SmallVector<int64_t, 3> shape = {4, 256, 32}; // [batch, M, K]
  Value bound = constI(100, 32);
  Value mask = buildBoundedMask(arange(256), /*varyDim=*/1, shape, bound);

  auto result = matchRowLimitMask(mask, /*other=*/Value(), shape,
                                  /*contiguousDim=*/2,
                                  /*pitchDimIsBroadcast=*/false);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->boundedDim, 1u);
  EXPECT_EQ(result->bound, bound);
  EXPECT_FALSE(result->extraPred);
}

// The mask also ANDs in a batch-index bound (degenerate along the pitch and
// contiguous dims), in either operand order: the row-limit half still
// becomes `base_height`, and the batch half comes back as `extraPred`.
TEST_F(MatchRowLimitMaskTest, RankThreeBatchPredicateKeptAsExtraPred) {
  SmallVector<int64_t, 3> shape = {4, 256, 32}; // [batch, M, K]
  Value rowBound = constI(100, 32);
  Value rowLimit =
      buildBoundedMask(arange(256), /*varyDim=*/1, shape, rowBound);
  Value batchBound = constI(3, 32);
  Value batchPred =
      buildBoundedMask(arange(4), /*varyDim=*/0, shape, batchBound);

  for (Value mask : {andI(rowLimit, batchPred), andI(batchPred, rowLimit)}) {
    auto result = matchRowLimitMask(mask, /*other=*/Value(), shape,
                                    /*contiguousDim=*/2,
                                    /*pitchDimIsBroadcast=*/false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->boundedDim, 1u);
    EXPECT_EQ(result->bound, rowBound);
    EXPECT_EQ(result->extraPred, batchPred);
  }
}

// `tt-reorder-broadcast` can sink the AND below the final broadcast:
// broadcast(and(broadcast(row), broadcast(batch))) with the AND on [B, M, 1].
// The matcher looks through it, and `extraPred` keeps that narrow shape; the
// IR is not rewritten.
TEST_F(MatchRowLimitMaskTest, RankThreeAndSunkBelowBroadcastAccepted) {
  SmallVector<int64_t, 3> shape = {4, 256, 32};      // [batch, M, K]
  SmallVector<int64_t, 3> narrowShape = {4, 256, 1}; // AND's shape
  Value rowBound = constI(100, 32);
  Value rowLimit =
      buildBoundedMask(arange(256), /*varyDim=*/1, narrowShape, rowBound);
  Value batchPred =
      buildBoundedMask(arange(4), /*varyDim=*/0, narrowShape, constI(3, 32));

  for (Value narrowAnd :
       {andI(rowLimit, batchPred), andI(batchPred, rowLimit)}) {
    Value mask = broadcastTo(narrowAnd, shape);
    auto result = matchRowLimitMask(mask, /*other=*/Value(), shape,
                                    /*contiguousDim=*/2,
                                    /*pitchDimIsBroadcast=*/false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->boundedDim, 1u);
    EXPECT_EQ(result->bound, rowBound);
    EXPECT_EQ(result->extraPred, batchPred);
  }
}

// A sunk AND whose second operand varies along the pitch dim is not a batch
// predicate, so it is rejected just like the un-sunk form.
TEST_F(MatchRowLimitMaskTest, SunkAndWithNonBatchOperandRejected) {
  SmallVector<int64_t, 3> shape = {4, 256, 32};
  SmallVector<int64_t, 3> narrowShape = {4, 256, 1};
  Value rowLimit = buildBoundedMask(arange(256), /*varyDim=*/1, narrowShape,
                                    constI(100, 32));
  Value otherRow =
      buildBoundedMask(arange(256), /*varyDim=*/1, narrowShape, constI(50, 32));
  Value mask = broadcastTo(andI(rowLimit, otherRow), shape);
  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/2,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// A second AND operand varying along the contiguous (K) dim isn't a pure
// batch-dim predicate -- it can change within a single hardware message --
// so the whole AND is rejected rather than misidentified.
TEST_F(MatchRowLimitMaskTest, ExtraPredVaryingAlongContiguousDimRejected) {
  SmallVector<int64_t, 3> shape = {4, 256, 32}; // [batch, M, K]
  Value rowLimit =
      buildBoundedMask(arange(256), /*varyDim=*/1, shape, constI(100, 32));
  Value colPred =
      buildBoundedMask(arange(32), /*varyDim=*/2, shape, constI(16, 32));
  Value mask = andI(rowLimit, colPred);

  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/2,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// computeRowLimitHeight: `clamp(bound - rowBase, 0, rowExtent)` as an i32.
// i32 inputs (or no rowBase) are widened to i64 and clamped with signed
// min/max; i64 inputs, whose difference can overflow, use
// `(bound <=s rowBase) ? 0 : minui(bound - rowBase, rowExtent)`.
class ComputeRowLimitHeightTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    module = ModuleOp::create(UnknownLoc::get(&ctx));
    builder.setInsertionPointToStart(module->getBody());
  }

  Location loc() { return builder.getUnknownLoc(); }

  Value constI32(int64_t value) {
    return arith::ConstantIntOp::create(builder, loc(), builder.getI32Type(),
                                        value);
  }

protected:
  MLIRContext ctx;
  OpBuilder builder{&ctx};
  OwningOpRef<ModuleOp> module;
};

// No arith.subi: the widened bound feeds the clamp directly.
TEST_F(ComputeRowLimitHeightTest, NullRowBaseSkipsSubtraction) {
  Value bound = constI32(100);
  Value height = computeRowLimitHeight(builder, loc(), bound, /*rowBase=*/{},
                                       /*rowExtent=*/64);
  ASSERT_TRUE(height.getType().isInteger(32));

  auto trunc = height.getDefiningOp<arith::TruncIOp>();
  ASSERT_TRUE(trunc);
  auto minOp = trunc.getIn().getDefiningOp<arith::MinSIOp>();
  ASSERT_TRUE(minOp);
  auto maxOp = minOp.getLhs().getDefiningOp<arith::MaxSIOp>();
  ASSERT_TRUE(maxOp);
  EXPECT_TRUE(isa<arith::ExtSIOp>(maxOp.getLhs().getDefiningOp()));
}

// An i64 bound with no rowBase can't overflow: the bound feeds the signed
// clamp directly, with no subtraction and no sign extension.
TEST_F(ComputeRowLimitHeightTest, I64BoundNullRowBase) {
  Value bound =
      arith::ConstantIntOp::create(builder, loc(), builder.getI64Type(), 100);
  Value height = computeRowLimitHeight(builder, loc(), bound, /*rowBase=*/{},
                                       /*rowExtent=*/64);
  ASSERT_TRUE(height.getType().isInteger(32));

  auto trunc = height.getDefiningOp<arith::TruncIOp>();
  ASSERT_TRUE(trunc);
  auto minOp = trunc.getIn().getDefiningOp<arith::MinSIOp>();
  ASSERT_TRUE(minOp);
  auto maxOp = minOp.getLhs().getDefiningOp<arith::MaxSIOp>();
  ASSERT_TRUE(maxOp);
  EXPECT_EQ(maxOp.getLhs(), bound);
}

// i64 inputs can overflow `bound - rowBase`, so the height is
// `(bound <=s rowBase) ? 0 : minui(bound - rowBase, rowExtent)`, with no
// sign extension of the already-i64 operands.
TEST_F(ComputeRowLimitHeightTest, I64InputsUseOverflowSafeClamp) {
  Value bound =
      arith::ConstantIntOp::create(builder, loc(), builder.getI64Type(), 100);
  Value rowBase =
      arith::ConstantIntOp::create(builder, loc(), builder.getI64Type(), 7);
  Value height = computeRowLimitHeight(builder, loc(), bound, rowBase,
                                       /*rowExtent=*/64);
  ASSERT_TRUE(height.getType().isInteger(32));

  auto trunc = height.getDefiningOp<arith::TruncIOp>();
  ASSERT_TRUE(trunc);
  auto select = trunc.getIn().getDefiningOp<arith::SelectOp>();
  ASSERT_TRUE(select);
  auto cmp = select.getCondition().getDefiningOp<arith::CmpIOp>();
  ASSERT_TRUE(cmp);
  EXPECT_EQ(cmp.getPredicate(), arith::CmpIPredicate::sle);
  EXPECT_EQ(cmp.getLhs(), bound);
  EXPECT_EQ(cmp.getRhs(), rowBase);
  auto minOp = select.getFalseValue().getDefiningOp<arith::MinUIOp>();
  ASSERT_TRUE(minOp);
  auto sub = minOp.getLhs().getDefiningOp<arith::SubIOp>();
  ASSERT_TRUE(sub);
  EXPECT_EQ(sub.getLhs(), bound);
  EXPECT_EQ(sub.getRhs(), rowBase);
}

// matchBoundaryCheckMask, reusing MatchRowLimitMaskTest's builders.
class MatchBoundaryCheckMaskTest : public MatchRowLimitMaskTest {
protected:
  std::optional<BoundaryMaskInfo> match(Value mask, ArrayRef<int64_t> shape,
                                        Value other = Value(),
                                        bool allowConstBounds = false) {
    // The mask's own op stands in for the load as the point the index base's
    // sign is asked at.
    return matchBoundaryCheckMask(mask, other, shape, /*contiguousDim=*/1,
                                  /*pitchDimIsBroadcast=*/false,
                                  mask.getDefiningOp(), allowConstBounds);
  }

  // `match` with constant bounds allowed.
  std::optional<BoundaryMaskInfo> matchConst(Value mask,
                                             ArrayRef<int64_t> shape) {
    return match(mask, shape, Value(), /*allowConstBounds=*/true);
  }

  Value cmpSge(Value lhs, Value rhs) {
    return arith::CmpIOp::create(builder, loc(), arith::CmpIPredicate::sge, lhs,
                                 rhs);
  }

  // `arange(0, extent) >= 0` along `varyDim`, broadcast to `tensorShape`.
  Value buildLowerBound(unsigned varyDim, ArrayRef<int64_t> tensorShape) {
    SmallVector<int64_t> narrowShape(tensorShape.size(), 1);
    narrowShape[varyDim] = tensorShape[varyDim];
    Value indexND =
        expandToShape(arange(tensorShape[varyDim]), varyDim, narrowShape);
    Value zero = splatTo(constI(0, 32), narrowShape);
    return broadcastTo(cmpSge(indexND, zero), tensorShape);
  }

  // `index < dense<bound>` along `varyDim`, the constant of the index's type,
  // broadcast to `tensorShape`; `ult` compares unsigned.
  Value buildConstBoundedMask(Value index1D, unsigned varyDim,
                              ArrayRef<int64_t> tensorShape, int64_t bound,
                              bool ult = false) {
    SmallVector<int64_t> narrowShape(tensorShape.size(), 1);
    narrowShape[varyDim] = tensorShape[varyDim];
    Value indexND = expandToShape(index1D, varyDim, narrowShape);
    auto ty = cast<RankedTensorType>(indexND.getType());
    IntegerAttr attr = builder.getIntegerAttr(ty.getElementType(), bound);
    Value boundND = arith::ConstantOp::create(builder, loc(), ty,
                                              DenseElementsAttr::get(ty, attr));
    auto pred = ult ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
    Value cmp = arith::CmpIOp::create(builder, loc(), pred, indexND, boundND);
    return broadcastTo(cmp, tensorShape);
  }
};

TEST_F(MatchBoundaryCheckMaskTest, BothDimsBoundedAccepted) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowBound = constI(100, 32), colBound = constI(24, 32);
  Value colBase = constI(64, 32);
  Value rowMask = buildBoundedMask(arange(256), /*varyDim=*/0, shape, rowBound);
  Value colIndex = addI32(splatTo(colBase, {32}), arange(32));
  Value colMask = buildBoundedMask(colIndex, /*varyDim=*/1, shape, colBound);

  auto result = match(andI(rowMask, colMask), shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, OpFoldResult(rowBound));
  EXPECT_FALSE(result->rowBase); // bare arange
  EXPECT_EQ(result->colBound, OpFoldResult(colBound));
  EXPECT_EQ(result->colBase, colBase);
  EXPECT_TRUE(result->extraPreds.empty());
}

TEST_F(MatchBoundaryCheckMaskTest, ColBoundAloneAccepted) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value colBound = constI(24, 32);
  Value colMask = buildBoundedMask(arange(32), /*varyDim=*/1, shape, colBound);

  auto result = match(colMask, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->rowBound);
  EXPECT_EQ(result->colBound, OpFoldResult(colBound));
}

// Both surfaces start at the tensor, so a column before it is read like any
// other: an upper bound alone agrees with the mask whatever the base's sign.
TEST_F(MatchBoundaryCheckMaskTest, ColBoundWithSignedBaseAccepted) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value colBase = constI(-8, 32);
  Value colIndex = addI32(splatTo(colBase, {32}), arange(32));
  Value colMask =
      buildBoundedMask(colIndex, /*varyDim=*/1, shape, constI(24, 32));

  auto result = match(colMask, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->colBase, colBase);
}

// ... but a `>= 0` conjunct can then only be dropped where it is vacuous.
TEST_F(MatchBoundaryCheckMaskTest, ColLowerBoundWithSignedBaseRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value colIndex = addI32(splatTo(constI(-8, 32), {32}), arange(32));
  SmallVector<int64_t, 2> narrow = {1, 32};
  Value index2D = expandDims(colIndex, /*axis=*/0, narrow);
  Value lower =
      broadcastTo(cmpSge(index2D, splatTo(constI(0, 32), narrow)), shape);
  Value upper =
      buildBoundedMask(colIndex, /*varyDim=*/1, shape, constI(24, 32));
  EXPECT_FALSE(match(andI(lower, upper), shape).has_value());
}

// Unlike matchRowLimitMask, which keeps at most one, every conjunct uniform
// along both surface dims is kept, splats included: a batch-dim predicate and
// a scalar flag can both reach one mask.
TEST_F(MatchBoundaryCheckMaskTest, SeveralUniformConjunctsKept) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowBound = constI(100, 32);
  Value rowMask = buildBoundedMask(arange(256), /*varyDim=*/0, shape, rowBound);
  Value flagA = splatTo(constI(1, 1), shape);
  Value flagB = splatTo(constI(1, 1), shape);

  auto result = match(andI(flagA, andI(rowMask, flagB)), shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, OpFoldResult(rowBound));
  ASSERT_EQ(result->extraPreds.size(), 2u);
  EXPECT_EQ(result->extraPreds[0], flagA);
  EXPECT_EQ(result->extraPreds[1], flagB);
}

// Two pitch-dim bounds are two clamps, and base_height holds only one.
TEST_F(MatchBoundaryCheckMaskTest, SecondRowBoundRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value first =
      buildBoundedMask(arange(256), /*varyDim=*/0, shape, constI(100, 32));
  Value second =
      buildBoundedMask(arange(256), /*varyDim=*/0, shape, constI(90, 32));
  EXPECT_FALSE(match(andI(first, second), shape).has_value());
}

// Only a zero `other` matches the hardware's zero fill.
TEST_F(MatchBoundaryCheckMaskTest, NonZeroFillOtherRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value mask =
      buildBoundedMask(arange(256), /*varyDim=*/0, shape, constI(100, 32));
  EXPECT_FALSE(
      match(mask, shape, denseIntSplat(shape, /*value=*/1)).has_value());
}

// base_height cannot express a bound when every row is the same row.
TEST_F(MatchBoundaryCheckMaskTest, PitchDimBroadcastRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value mask =
      buildBoundedMask(arange(256), /*varyDim=*/0, shape, constI(100, 32));
  EXPECT_FALSE(matchBoundaryCheckMask(mask, /*other=*/Value(), shape,
                                      /*contiguousDim=*/1,
                                      /*pitchDimIsBroadcast=*/true,
                                      mask.getDefiningOp())
                   .has_value());
}

// An AND reused on both sides at every level is a DAG of 2^depth paths; it is
// walked once per value, and the repeated conjuncts collapse into one each.
TEST_F(MatchBoundaryCheckMaskTest, SharedAndVisitedOnce) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowBound = constI(100, 32);
  Value flag = splatTo(constI(1, 1), shape);
  Value mask =
      andI(buildBoundedMask(arange(256), /*varyDim=*/0, shape, rowBound), flag);
  for (int i = 0; i < 64; ++i)
    mask = andI(mask, mask);

  auto result = match(mask, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, OpFoldResult(rowBound));
  ASSERT_EQ(result->extraPreds.size(), 1u);
  EXPECT_EQ(result->extraPreds.front(), flag);
}

// `bound > index` is the same bound with the index on the right (the form
// `K > offs_k` takes in the IR).
TEST_F(MatchBoundaryCheckMaskTest, SwappedComparisonAccepted) {
  SmallVector<int64_t, 2> shape = {256, 32};
  SmallVector<int64_t, 2> narrow = {256, 1};
  Value rowBound = constI(100, 32);
  Value index2D = expandDims(arange(256), /*axis=*/1, narrow);
  Value mask = broadcastTo(cmpSgt(splatTo(rowBound, narrow), index2D), shape);

  auto result = match(mask, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, OpFoldResult(rowBound));
}

// `index > bound` is a lower bound, which base_height cannot express.
TEST_F(MatchBoundaryCheckMaskTest, IndexGreaterThanBoundRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  SmallVector<int64_t, 2> narrow = {256, 1};
  Value index2D = expandDims(arange(256), /*axis=*/1, narrow);
  Value bound2D = splatTo(constI(100, 32), narrow);
  Value mask = broadcastTo(cmpSgt(index2D, bound2D), shape);
  EXPECT_FALSE(match(mask, shape).has_value());
}

// A bare `arange` starts at zero, so a lower bound next to the upper bound is
// vacuous and dropped.
TEST_F(MatchBoundaryCheckMaskTest, LowerBoundDroppedNextToUpperBound) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowBound = constI(100, 32);
  Value rowMask = andI(buildLowerBound(/*varyDim=*/0, shape),
                       buildBoundedMask(arange(256), 0, shape, rowBound));

  auto result = match(rowMask, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, OpFoldResult(rowBound));
  EXPECT_TRUE(result->extraPreds.empty());
}

// Alone it is the whole mask, and dropping it would make the load unmasked.
TEST_F(MatchBoundaryCheckMaskTest, LowerBoundAloneRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  EXPECT_FALSE(match(buildLowerBound(/*varyDim=*/0, shape), shape).has_value());
}

// `index >= 8` excludes rows the surface would read, so it is not vacuous.
TEST_F(MatchBoundaryCheckMaskTest, NonZeroLowerBoundRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  SmallVector<int64_t, 2> narrow = {256, 1};
  Value index2D = expandDims(arange(256), /*axis=*/1, narrow);
  Value lower =
      broadcastTo(cmpSge(index2D, splatTo(constI(8, 32), narrow)), shape);
  Value upper =
      buildBoundedMask(arange(256), /*varyDim=*/0, shape, constI(100, 32));
  EXPECT_FALSE(match(andI(lower, upper), shape).has_value());
}

// A lower and an upper bound that start the index at different places are not
// the two halves of one boundary check.
TEST_F(MatchBoundaryCheckMaskTest, MismatchedBasesOnOneDimRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value based = addI32(splatTo(constI(8, 32), {256}), arange(256));
  Value upper = buildBoundedMask(based, /*varyDim=*/0, shape, constI(100, 32));
  EXPECT_FALSE(match(andI(buildLowerBound(/*varyDim=*/0, shape), upper), shape)
                   .has_value());
}

// `0 <= index` is the lower bound with the index on the right.
TEST_F(MatchBoundaryCheckMaskTest, SwappedLowerBoundAccepted) {
  SmallVector<int64_t, 2> shape = {256, 32};
  SmallVector<int64_t, 2> narrow = {256, 1};
  Value rowBound = constI(100, 32);
  Value index2D = expandDims(arange(256), /*axis=*/1, narrow);
  Value lower = broadcastTo(
      arith::CmpIOp::create(builder, loc(), arith::CmpIPredicate::sle,
                            splatTo(constI(0, 32), narrow), index2D),
      shape);
  Value upper = buildBoundedMask(arange(256), /*varyDim=*/0, shape, rowBound);

  auto result = match(andI(lower, upper), shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, OpFoldResult(rowBound));
}

// matchRowLimitMask has no load to ask the sign at, so it drops a lower
// bound only for a bare `arange`, even where the load-aware match proves the
// base non-negative.
TEST_F(MatchBoundaryCheckMaskTest, LowerBoundWithBaseNeedsLoadOp) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowIndex = addI32(splatTo(constI(8, 32), {256}), arange(256));
  SmallVector<int64_t, 2> narrow = {256, 1};
  Value index2D = expandDims(rowIndex, /*axis=*/1, narrow);
  Value lower =
      broadcastTo(cmpSge(index2D, splatTo(constI(0, 32), narrow)), shape);
  Value upper =
      buildBoundedMask(rowIndex, /*varyDim=*/0, shape, constI(100, 32));
  Value mask = andI(lower, upper);

  EXPECT_TRUE(match(mask, shape).has_value());
  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// A splat constant bound is matched only with `allowConstBounds`.
TEST_F(MatchBoundaryCheckMaskTest, ConstBoundNeedsOptIn) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value mask = buildConstBoundedMask(arange(256), /*varyDim=*/0, shape, 100);
  EXPECT_FALSE(match(mask, shape).has_value());
}

// Each constant bound comes back as an IntegerAttr of the compare's type.
TEST_F(MatchBoundaryCheckMaskTest, ConstBoundsAccepted) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value colBase = constI(64, 32);
  Value colIndex = addI32(splatTo(colBase, {32}), arange(32));
  Value mask =
      andI(buildConstBoundedMask(arange(256), /*varyDim=*/0, shape, 100),
           buildConstBoundedMask(colIndex, /*varyDim=*/1, shape, 24));

  auto result = matchConst(mask, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, OpFoldResult(builder.getI32IntegerAttr(100)));
  EXPECT_FALSE(result->rowBase); // bare arange
  EXPECT_EQ(result->colBound, OpFoldResult(builder.getI32IntegerAttr(24)));
  EXPECT_EQ(result->colBase, colBase);
  EXPECT_TRUE(result->extraPreds.empty());
}

// A scalar bound stays a Value next to a constant one.
TEST_F(MatchBoundaryCheckMaskTest, ScalarAndConstBoundsKeepTheirForms) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowBound = constI(100, 32);
  Value mask =
      andI(buildBoundedMask(arange(256), /*varyDim=*/0, shape, rowBound),
           buildConstBoundedMask(arange(32), /*varyDim=*/1, shape, 24));

  auto result = matchConst(mask, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, OpFoldResult(rowBound));
  EXPECT_EQ(result->colBound, OpFoldResult(builder.getI32IntegerAttr(24)));
}

// One upper bound per dim also when one is a constant, in either order.
TEST_F(MatchBoundaryCheckMaskTest, SecondRowBoundRejectedWhenConst) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value scalar =
      buildBoundedMask(arange(256), /*varyDim=*/0, shape, constI(100, 32));
  Value constant = buildConstBoundedMask(arange(256), /*varyDim=*/0, shape, 90);
  EXPECT_FALSE(matchConst(andI(scalar, constant), shape).has_value());
  EXPECT_FALSE(matchConst(andI(constant, scalar), shape).has_value());
}

// A constant bound has to be a splat.
TEST_F(MatchBoundaryCheckMaskTest, NonSplatConstBoundRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  SmallVector<int64_t, 2> narrow = {1, 32};
  SmallVector<int32_t> limits(32, 24);
  limits.back() = 25;
  auto ty = RankedTensorType::get(narrow, builder.getI32Type());
  auto attr = DenseElementsAttr::get(ty, ArrayRef<int32_t>(limits));
  Value bound = arith::ConstantOp::create(builder, loc(), ty, attr);
  Value index2D = expandDims(arange(32), /*axis=*/0, narrow);
  Value mask = broadcastTo(cmpSlt(index2D, bound), shape);
  EXPECT_FALSE(matchConst(mask, shape).has_value());
}

// An i64 constant bound keeps its width.
TEST_F(MatchBoundaryCheckMaskTest, I64ConstBoundKeepsWidth) {
  SmallVector<int64_t, 2> shape = {256, 32};
  auto i64TensorTy = RankedTensorType::get({256}, builder.getI64Type());
  Value idx64 =
      arith::ExtSIOp::create(builder, loc(), i64TensorTy, arange(256));
  int64_t bound = int64_t(1) << 40;
  Value mask = buildConstBoundedMask(idx64, /*varyDim=*/0, shape, bound);

  auto result = matchConst(mask, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, OpFoldResult(builder.getI64IntegerAttr(bound)));
}

// matchStoreBoundaryClamp on the 64x128 store of #8813: rows
// `pid_m * rowMul + arange(64)` below `rowBound` and columns
// `pid_n * colMul + arange(numCols)` below `colBound`, both splat constants.
class MatchStoreBoundaryClampTest : public MatchBoundaryCheckMaskTest {
protected:
  struct Spec {
    std::optional<int64_t> rowBound = 8450, colBound = 504;
    int64_t rowMul = 64, numCols = 128, colMul = numCols;
    bool rowShl = false;              // row base `pid_m << 6` instead
    bool rowAndZero = false;          // adds `pid_n & 0` to the row base
    bool f32 = false;                 // f16 otherwise
    unsigned intBits = 0;             // iN elements if nonzero, else f16/f32
    bool colUlt = false;              // unsigned column compare
    bool flag = false;                // ANDs in a splatted i1 argument
    std::optional<int32_t> alignment; // ttig.2d_block_io_base_alignment
    unsigned contiguousDim = 1;
  };

  // Builds the store in a new function and matches it.
  StoreBoundaryClamp matchStore(const Spec &s, std::optional<int64_t> pitch) {
    SmallVector<int64_t, 2> shape = {64, s.numCols};
    Type elemTy = s.f32 ? builder.getF32Type() : builder.getF16Type();
    if (s.intBits)
      elemTy = builder.getIntegerType(s.intBits);
    Type i32 = builder.getI32Type();
    SmallVector<Type> args = {
        RankedTensorType::get(shape, triton::PointerType::get(elemTy)),
        RankedTensorType::get(shape, elemTy), i32, i32, builder.getI1Type()};
    builder.setInsertionPointToEnd(module->getBody());
    auto funcOp = triton::FuncOp::create(builder, loc(),
                                         "store" + std::to_string(numFuncs++),
                                         builder.getFunctionType(args, {}));
    builder.setInsertionPointToStart(funcOp.addEntryBlock());
    auto ret = triton::ReturnOp::create(builder, loc(), ValueRange{});
    builder.setInsertionPoint(ret);

    Value pidM = funcOp.getArgument(2), pidN = funcOp.getArgument(3);
    rowBase = mulI32(pidM, constI(s.rowMul, 32));
    if (s.rowShl)
      rowBase = arith::ShLIOp::create(builder, loc(), pidM, constI(6, 32));
    if (s.rowAndZero)
      rowBase = addI32(rowBase, andI(pidN, constI(0, 32)));
    colBase = mulI32(pidN, constI(s.colMul, 32));
    Value rows = addI32(splatTo(rowBase, {64}), arange(64));
    Value cols = addI32(splatTo(colBase, {s.numCols}), arange(s.numCols));
    Value mask;
    auto andWith = [&](Value c) { mask = mask ? andI(mask, c) : c; };
    if (s.rowBound)
      andWith(buildConstBoundedMask(rows, /*varyDim=*/0, shape, *s.rowBound));
    if (s.colBound)
      andWith(buildConstBoundedMask(cols, /*varyDim=*/1, shape, *s.colBound,
                                    s.colUlt));
    if (s.flag)
      andWith(splatTo(funcOp.getArgument(4), shape));
    auto store = triton::StoreOp::create(builder, loc(), funcOp.getArgument(0),
                                         funcOp.getArgument(1), mask);

    StringRef attrName =
        TritonIntelGPUDialect::get2DBlockIOBaseAlignmentAttrName();
    if (s.alignment)
      (*module)->setAttr(attrName, builder.getI32IntegerAttr(*s.alignment));
    else
      (*module)->removeAttr(attrName);
    triton::intel::ModuleAxisInfoAnalysis axisInfo(*module);
    return matchStoreBoundaryClamp(store, s.contiguousDim, pitch, axisInfo);
  }

  // The dims `matchStore` clamps: "row+col", "row", "col" or "none".
  std::string clampedDims(const Spec &s, std::optional<int64_t> pitch = 1008) {
    StoreBoundaryClamp c = matchStore(s, pitch);
    if (c.row && c.col)
      return "row+col";
    return c.row ? "row" : c.col ? "col" : "none";
  }

  // Sets `name` to `value`, or unsets it for a null `value`.
  static void setEnv(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
      setenv(name, value, /*overwrite=*/1);
    else
      unsetenv(name);
#endif
  }

  // Each test starts with the kill switch unset; TearDown restores it.
  void SetUp() override {
    MatchBoundaryCheckMaskTest::SetUp();
    if (const char *old = std::getenv(kKillSwitch))
      savedKillSwitch = old;
    setEnv(kKillSwitch, nullptr);
  }
  void TearDown() override {
    setEnv(kKillSwitch, savedKillSwitch ? savedKillSwitch->c_str() : nullptr);
  }

  static constexpr const char *kKillSwitch =
      "TRITON_INTEL_DISABLE_STORE_BOUNDARY_CLAMP";
  std::optional<std::string> savedKillSwitch;
  Value rowBase, colBase; // of the last `matchStore`
  int numFuncs = 0;
};

// The #8813 store mask clamps each dim at its constant bound.
TEST_F(MatchStoreBoundaryClampTest, KernelMaskClampsBothDims) {
  StoreBoundaryClamp c = matchStore({}, /*pitch=*/1008);
  ASSERT_TRUE(c.row.has_value());
  ASSERT_TRUE(c.col.has_value());
  EXPECT_EQ(c.row->base, rowBase);
  EXPECT_EQ(c.row->bound, OpFoldResult(builder.getI32IntegerAttr(8450)));
  EXPECT_EQ(c.row->extent, 64);
  EXPECT_EQ(c.col->base, colBase);
  EXPECT_EQ(c.col->bound, OpFoldResult(builder.getI32IntegerAttr(504)));
  EXPECT_EQ(c.col->extent, 128);
}

// A uniform extra conjunct refuses every dim.
TEST_F(MatchStoreBoundaryClampTest, UniformConjunctRefused) {
  EXPECT_EQ(clampedDims({.flag = true}), "none");
}

// An unsigned compare is no boundary check, so nothing clamps.
TEST_F(MatchStoreBoundaryClampTest, UnsignedCompareRefused) {
  EXPECT_EQ(clampedDims({.colUlt = true}), "none");
}

// A base not provably a multiple of PowerOf2Ceil(extent) may wrap.
TEST_F(MatchStoreBoundaryClampTest, UnalignedBaseRefused) {
  EXPECT_EQ(clampedDims({.rowMul = 32}), "col");
  EXPECT_EQ(clampedDims({.colMul = 64}), "row");
}

// Proving the row base a multiple of 64 needs both provers: only AxisInfo
// admits `<< 6`, only isDivisible `& 0`.
TEST_F(MatchStoreBoundaryClampTest, RowBaseNeedsBothProvers) {
  EXPECT_EQ(clampedDims({.rowShl = true}), "col");
  EXPECT_EQ(clampedDims({.rowAndZero = true}), "col");
}

// 505 f16 columns are no whole number of 4-byte granules; f32 ones are.
TEST_F(MatchStoreBoundaryClampTest, OddColumnBoundNeedsGranule) {
  EXPECT_EQ(clampedDims({.colBound = 505}), "row");
  EXPECT_EQ(clampedDims({.colBound = 505, .f32 = true}), "row+col");
}

// An i1 width has no exact byte count, so even a bound of 512, a whole
// number of 4-byte granules of i1, is refused.
TEST_F(MatchStoreBoundaryClampTest, SubByteColumnRefused) {
  EXPECT_EQ(clampedDims({.colBound = 512, .intBits = 1}), "row");
}

// The column base must start on a 4-byte granule: two i8 columns at
// `pid_n * 2` may start mid-granule, which the base's PowerOf2Ceil(extent)
// proof alone admits.
TEST_F(MatchStoreBoundaryClampTest, UnalignedColumnBaseRefused) {
  EXPECT_EQ(clampedDims({.numCols = 2, .intBits = 8}), "row");
}

// With compensation, a 128-column f16 tile needs a pitch of 256 + 63 bytes
// on every dim; an unknown pitch or a contiguous dim past the rank refuses.
TEST_F(MatchStoreBoundaryClampTest, PitchMustCoverCompensatedTile) {
  EXPECT_EQ(clampedDims({.colBound = 120}, /*pitch=*/240), "none");
  Spec rowOnly{.colBound = std::nullopt};
  EXPECT_EQ(clampedDims(rowOnly, /*pitch=*/256), "none");
  rowOnly.alignment = 64;
  EXPECT_EQ(clampedDims(rowOnly, /*pitch=*/256), "none");
  EXPECT_EQ(clampedDims({.colBound = 120}, /*pitch=*/320), "row+col");
  EXPECT_EQ(clampedDims({}, /*pitch=*/std::nullopt), "none");
  EXPECT_EQ(clampedDims({.contiguousDim = 2}), "none");
}

// Without compensation (base alignment 4), a pitch of 256 bytes suffices.
TEST_F(MatchStoreBoundaryClampTest, TilePitchSufficesWithoutCompensation) {
  EXPECT_EQ(clampedDims({.alignment = 4}, /*pitch=*/256), "row+col");
}

// The pitch floor is 64 bytes: a 32-byte row with compensation needs 127.
TEST_F(MatchStoreBoundaryClampTest, NarrowRowNeedsPitchFloor) {
  EXPECT_EQ(clampedDims({.numCols = 16}, /*pitch=*/112), "none");
  EXPECT_EQ(clampedDims({.numCols = 16}, /*pitch=*/128), "row+col");
}

// The minimum pitch is exact in bytes: 128 f32 columns need 512 + 63.
TEST_F(MatchStoreBoundaryClampTest, CompensationIs63Bytes) {
  EXPECT_EQ(clampedDims({.f32 = true}, /*pitch=*/574), "none");
  EXPECT_EQ(clampedDims({.f32 = true}, /*pitch=*/575), "row+col");
}

// The kill switch refuses every dim.
TEST_F(MatchStoreBoundaryClampTest, KillSwitchRefused) {
  setEnv(kKillSwitch, "1");
  EXPECT_EQ(clampedDims({}), "none");
}

// The pitch policy shared by MaterializeBlockPointer and LowerTo2DBlockLoad.
// The runtime cases are not reachable from lit: a runtime row stride does not
// pass MaterializeBlockPointer's alignment check there.
TEST(IsBlockPitchSupportedTest, ConstantPitch) {
  EXPECT_TRUE(isBlockPitchSupported(64, false, 0));
  EXPECT_TRUE(isBlockPitchSupported(int64_t(1) << 24, false, 0));
  EXPECT_FALSE(isBlockPitchSupported(32, false, 1024)); // < 64
  EXPECT_FALSE(isBlockPitchSupported(72, false, 1024)); // not 16-aligned
  EXPECT_FALSE(isBlockPitchSupported((int64_t(1) << 24) + 16, false, 1024));
}

TEST(IsBlockPitchSupportedTest, RuntimePitch) {
  EXPECT_TRUE(isBlockPitchSupported(std::nullopt, true, 64));
  EXPECT_FALSE(isBlockPitchSupported(std::nullopt, true, 32));   // narrow row
  EXPECT_FALSE(isBlockPitchSupported(std::nullopt, false, 128)); // no stride
}

} // namespace
