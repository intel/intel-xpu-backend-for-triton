// Tests for BlockIOUtils: validate2DBlockLoadTile, pickBlockIOLoadEncoding.
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
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

using namespace mlir;
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

TEST_F(BlockIOUtilsTest, PickF32Tpw32Warps4ReduceRowsD128) {
  auto type = makeCoalescedType({16, 128}, Float32Type::get(&ctx),
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

// Only rank-2 loads are picked.
TEST_F(BlockIOUtilsTest, PickRejectsRank3) {
  auto type = makeCoalescedType({2, 16, 256}, Float32Type::get(&ctx),
                                /*numWarps=*/4, /*threadsPerWarp=*/32);
  for (std::optional<unsigned> reducedDim : kReducedDims)
    EXPECT_FALSE(pickBlockIOLoadEncoding(type, /*numWarps=*/4,
                                         /*threadsPerWarp=*/32, reducedDim)
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

} // namespace
