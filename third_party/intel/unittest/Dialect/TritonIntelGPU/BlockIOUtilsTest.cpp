// Tests for validate2DBlockLoadTile and matchRowLimitMask in BlockIOUtils.
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
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Signals.h"
#include <algorithm>
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
TEST_F(MatchRowLimitMaskTest, NonArangeLhsRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value scaled = mulI32(arange(256), splatTo(constI(2, 32), {256}));
  Value mask =
      buildRowLimitMask(scaled, /*boundedDim=*/0, shape, constI(100, 32));

  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
}

// sext to i64 still matches the arange, but base_height requires i32.
TEST_F(MatchRowLimitMaskTest, NonI32BoundRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value idx64 = arith::ExtSIOp::create(
      builder, loc(), RankedTensorType::get({256}, builder.getI64Type()),
      arange(256));
  Value idx64_2D = expandDims(idx64, /*axis=*/1, {256, 1});
  Value bound64 = constI(100, 64);
  Value boundSplat2D = splatTo(bound64, {256, 1});
  Value cmp = cmpSlt(idx64_2D, boundSplat2D);
  Value mask = broadcastTo(cmp, shape);

  EXPECT_FALSE(matchRowLimitMask(mask, /*other=*/Value(), shape,
                                 /*contiguousDim=*/1,
                                 /*pitchDimIsBroadcast=*/false)
                   .has_value());
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

// computeRowLimitHeight: `clamp(bound - rowBase, 0, rowExtent)`, widened to
// i64 before subtracting to avoid overflow.
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

} // namespace
