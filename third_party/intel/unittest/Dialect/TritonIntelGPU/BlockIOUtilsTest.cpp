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
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
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
                                        Value other = Value()) {
    // The mask's own op stands in for the load: the sign query only needs a
    // use point, and these tests carry no `llvm.intr.assume`.
    return matchBoundaryCheckMask(mask, other, shape, /*contiguousDim=*/1,
                                  /*pitchDimIsBroadcast=*/false,
                                  mask.getDefiningOp());
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
};

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
  EXPECT_EQ(result->rowBound, rowBound);
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
  EXPECT_EQ(result->rowBound, rowBound);
  ASSERT_EQ(result->extraPreds.size(), 1u);
  EXPECT_EQ(result->extraPreds.front(), flag);
}

// `bound > index` is the same bound with the index on the right, as
// `K > offs_k` reaches the IR.
TEST_F(MatchBoundaryCheckMaskTest, SwappedComparisonAccepted) {
  SmallVector<int64_t, 2> shape = {256, 32};
  SmallVector<int64_t, 2> narrow = {256, 1};
  Value rowBound = constI(100, 32);
  Value index2D = expandDims(arange(256), /*axis=*/1, narrow);
  Value upper = broadcastTo(
      arith::CmpIOp::create(builder, loc(), arith::CmpIPredicate::sgt,
                            splatTo(rowBound, narrow), index2D),
      shape);

  auto result = match(upper, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, rowBound);
}

// A lower bound next to an upper bound on the same dim is dropped: it only
// fires for a negative index, which the match assumes away.
TEST_F(MatchBoundaryCheckMaskTest, LowerBoundDroppedNextToUpperBound) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowBound = constI(100, 32);
  Value rowMask = andI(buildLowerBound(/*varyDim=*/0, shape),
                       buildBoundedMask(arange(256), 0, shape, rowBound));

  auto result = match(rowMask, shape);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->rowBound, rowBound);
  EXPECT_TRUE(result->extraPreds.empty());
}

// Alone it is the whole mask, and dropping it would make the load unmasked.
TEST_F(MatchBoundaryCheckMaskTest, LowerBoundAloneRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  EXPECT_FALSE(match(buildLowerBound(/*varyDim=*/0, shape), shape).has_value());
}

// Two bounds on one dim that start the index at different places are not the
// two halves of one boundary check.
TEST_F(MatchBoundaryCheckMaskTest, MismatchedBasesOnOneDimRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value based = addI32(splatTo(constI(8, 32), {256}), arange(256));
  Value withBase = buildBoundedMask(based, 0, shape, constI(100, 32));
  Value withoutBase = buildBoundedMask(arange(256), 0, shape, constI(90, 32));
  EXPECT_FALSE(match(andI(withBase, withoutBase), shape).has_value());
}

// A negative row is read like any other, so a row bound with a `>= 0`
// conjunct is only absorbed when the base cannot be negative.
TEST_F(MatchBoundaryCheckMaskTest, RowLowerBoundWithSignedBaseRejected) {
  SmallVector<int64_t, 2> shape = {256, 32};
  Value rowIndex = addI32(splatTo(constI(-8, 32), {256}), arange(256));
  SmallVector<int64_t, 2> narrow = {256, 1};
  Value index2D = expandDims(rowIndex, /*axis=*/1, narrow);
  Value lower =
      broadcastTo(cmpSge(index2D, splatTo(constI(0, 32), narrow)), shape);
  Value upper =
      buildBoundedMask(rowIndex, /*varyDim=*/0, shape, constI(100, 32));
  EXPECT_FALSE(match(andI(lower, upper), shape).has_value());
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
  EXPECT_EQ(result->rowBound, rowBound);
}

// matchRowLimitMask has no load to ask the sign at, so it drops a lower
// bound only for a bare `arange`, even where the load-aware match proves the
// base non-negative.
TEST_F(MatchBoundaryCheckMaskTest, LowerBoundNeedsLoadOpForSignedBase) {
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
