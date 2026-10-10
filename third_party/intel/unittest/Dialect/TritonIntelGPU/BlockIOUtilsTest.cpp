// Tests for validate2DBlockLoadTile in BlockIOUtils.
//
// PR #7487 changed computeTransposeShuffleMapping from a simple width
// comparison to a linear-layout comparison, enabling column-major B matrix
// loads with sub-group-size=32.  These tests verify:
//   1. tpw=32, f16/opsPerChan=2, column-major B is now accepted.
//   2. tpw=16, f16/opsPerChan=2, column-major B is still accepted (regression).

#include "intel/include/Dialect/TritonIntelGPU/Transforms/BlockIOUtils.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Signals.h"
#include <algorithm>
#include <gtest/gtest.h>

using namespace mlir;
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

protected:
  MLIRContext ctx;

  // Rebuild `ll` with its "register" basis vectors reordered by `perm`
  // (perm[i] = source index for the i-th output basis). The resulting layout
  // describes the exact same register/lane->tensor mapping, just with the
  // register bases listed in a different ("swizzled") order -- the scenario
  // issue #7806 is about. Out-dim sizes and surjectivity are preserved.
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
    int numElemsPerLoad = (info.tileHeight * info.tileWidth *
                           info.numElemPerPackedVal * info.vBlocks) /
                          tpw;
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
  auto f16 = Float16Type::get(&ctx);
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
// register bases. This guards the latent correctness risk the ticket flags:
// an order-derived regMapping that no longer matches the tile it was sized for.
TEST_F(BlockIOUtilsTest,
       RegisterBaseOrderInvariance_RegMappingEquivalent_DpasOperandA_F16) {
  auto dpas = makeDpas(/*warpsPerCTA=*/{1, 1}, /*threadsPerWarp=*/16,
                       /*opsPerChan=*/2);
  auto dot = DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, dpas, /*kWidth=*/1);
  auto f16 = Float16Type::get(&ctx);
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
  auto f16 = Float16Type::get(&ctx);
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
  auto f16 = Float16Type::get(&ctx);
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

// Solution B guardrail (issue #7806): a well-formed tile's register mapping is
// consistent with its geometry, while a tile whose declared dimensions do not
// match the delivered data is rejected (so the op falls back to scatter).
TEST_F(BlockIOUtilsTest, TileRegMappingConsistencyGuardrail) {
  auto dpas = makeDpas(/*warpsPerCTA=*/{1, 1}, /*threadsPerWarp=*/16,
                       /*opsPerChan=*/2);
  auto dot = DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, dpas, /*kWidth=*/1);
  auto f16 = Float16Type::get(&ctx);
  SmallVector<int64_t> shape = {64, 32}; // M=64, K=32
  auto ll = cast<DistributedEncodingTrait>(dot).toLinearLayout(shape);
  auto info = getBlockIOLoadTileSize(ll, /*memContiguousDim=*/1,
                                     /*elemSizeInBits=*/16, nullptr, false);
  ASSERT_TRUE(info.isValid());
  ASSERT_NE(info.rowDim, info.colDim);

  // A well-formed tile is consistent.
  EXPECT_TRUE(isTileRegMappingConsistent(ll, info));

  // A tile that claims both of its dimensions are the same one is not: the
  // delivered data still varies the real row dimension, which now falls outside
  // the declared {rowDim, colDim}, so the guardrail rejects it.
  BlockIOTileSizeInfo corrupted(info.tileHeight, info.tileWidth,
                                info.numElemPerPackedVal, info.vBlocks,
                                /*rowDim=*/info.colDim, /*colDim=*/info.colDim,
                                info.transpose, info.vnni, info.regPackedBases);
  EXPECT_FALSE(isTileRegMappingConsistent(ll, corrupted));
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

} // namespace
