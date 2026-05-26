#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

namespace {

class UtilityTest : public ::testing::Test {
public:
  void SetUp() override {
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<func::FuncDialect>();
    ctx.getOrLoadDialect<TritonDialect>();
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctx.getOrLoadDialect<TritonIntelGPUDialect>();
    builder = std::make_unique<OpBuilder>(&ctx);
  }

  DpasEncodingAttr makeDpas(unsigned repeatCount = 8,
                             unsigned systolicDepth = 8,
                             unsigned executionSize = 16,
                             unsigned opsPerChannel = 4,
                             ArrayRef<unsigned> warpsPerCTA = {1, 1},
                             ArrayRef<unsigned> repCluster = {1, 1},
                             unsigned threadsPerWarp = 16) {
    return DpasEncodingAttr::get(&ctx, repeatCount, systolicDepth,
                                 executionSize, opsPerChannel, warpsPerCTA,
                                 repCluster, threadsPerWarp, std::nullopt);
  }

protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
};

// ===----------------------------------------------------------------------===//
// Tests for cvtIsSubGroupReinterpret
// ===----------------------------------------------------------------------===//

// Verify that converting a 32x32xf8E5M2 tensor from mma (dpas) layout to
// dot_op A layout with kWidth=2 is detected as a sub-group reinterpret.
TEST_F(UtilityTest, SubGroupReinterpret_MmaToDotA_F8E5M2) {
  // #ttig.dpas<{repeatCount=8, systolicDepth=8, executionSize=16, opsPerChan=4,
  //             threadsPerWarp=16, warpsPerCTA=[1,1], repCluster=[1,1]}>
  DpasEncodingAttr mmaLayout = makeDpas();
  // #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
  auto dotALayout =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, mmaLayout, /*kWidth=*/2);

  Type f8E5M2 = Float8E5M2Type::get(&ctx);
  auto srcTy = RankedTensorType::get({32, 32}, f8E5M2, mmaLayout);
  auto dstTy = RankedTensorType::get({32, 32}, f8E5M2, dotALayout);

  EXPECT_TRUE(cvtIsSubGroupReinterpret(srcTy, dstTy));
}

// Verify that a mma -> mma conversion (same layout) is NOT a sub-group
// reinterpret.
TEST_F(UtilityTest, SubGroupReinterpret_SameLayout_NotReinterpret) {
  DpasEncodingAttr mmaLayout = makeDpas();

  Type f8E5M2 = Float8E5M2Type::get(&ctx);
  auto srcTy = RankedTensorType::get({32, 32}, f8E5M2, mmaLayout);
  auto dstTy = RankedTensorType::get({32, 32}, f8E5M2, mmaLayout);

  EXPECT_FALSE(cvtIsSubGroupReinterpret(srcTy, dstTy));
}

// Verify that a blocked -> blocked shuffle conversion is NOT a sub-group
// reinterpret.
TEST_F(UtilityTest, SubGroupReinterpret_ShuffleConversion_NotReinterpret) {
  // A 16x16 layout transposition (shuffle case) should not be detected as
  // a reinterpret.
  SmallVector<unsigned> sizePerThread0 = {1, 16};
  SmallVector<unsigned> threadsPerWarp0 = {16, 1};
  SmallVector<unsigned> warpsPerCTA0 = {1, 1};
  SmallVector<unsigned> order = {0, 1};
  auto cgaLayout = CGAEncodingAttr::get1CTALayout(&ctx, /*rank=*/2);
  auto blocked0 = BlockedEncodingAttr::get(&ctx, sizePerThread0, threadsPerWarp0,
                                           warpsPerCTA0, order, cgaLayout);

  SmallVector<unsigned> sizePerThread1 = {16, 1};
  SmallVector<unsigned> threadsPerWarp1 = {1, 16};
  SmallVector<unsigned> warpsPerCTA1 = {1, 1};
  auto blocked1 = BlockedEncodingAttr::get(&ctx, sizePerThread1, threadsPerWarp1,
                                           warpsPerCTA1, order, cgaLayout);

  auto srcTy =
      RankedTensorType::get({16, 16}, builder->getF16Type(), blocked0);
  auto dstTy =
      RankedTensorType::get({16, 16}, builder->getF16Type(), blocked1);

  EXPECT_FALSE(cvtIsSubGroupReinterpret(srcTy, dstTy));
}

} // namespace

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
