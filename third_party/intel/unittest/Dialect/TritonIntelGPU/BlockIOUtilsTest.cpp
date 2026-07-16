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
  LinearLayout ll = cast<DistributedEncodingTrait>(dot).toLinearLayout(shape);
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
  LinearLayout ll = cast<DistributedEncodingTrait>(dot).toLinearLayout(shape);
  // column_major: memContiguousDim = rank-2 = 0 (K direction)
  EXPECT_TRUE(validate2DBlockLoadTile(ll, /*memContiguousDim=*/0,
                                      /*elemSizeInBits=*/16, tensorType));
}

} // namespace
