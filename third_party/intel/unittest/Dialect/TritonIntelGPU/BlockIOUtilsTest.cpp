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
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

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
};

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

// Any value that isn't a zero splat (or absent) fails isZeroFillOther.
TEST_F(MatchRowLimitMaskTest, NonZeroFillOtherRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value mask =
      buildRowLimitMask(arange(256), /*boundedDim=*/0, shape, constI(100, 32));
  Value nonZeroOther = constI(1, 32);
  EXPECT_FALSE(matchRowLimitMask(mask, nonZeroOther, shape,
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
