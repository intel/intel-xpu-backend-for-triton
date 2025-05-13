#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"

#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir::triton::gpu::intel {

namespace {

class LinearLayoutConversionsTest : public ::testing::Test {
public:
  void SetUp() { ctx.loadDialect<TritonGPUDialect, TritonIntelGPUDialect>(); }

  Subgroup2DBlockEncodingAttr sdb(ArrayRef<unsigned> instrShape,
                                  unsigned numBlocks, unsigned kWidth,
                                  ArrayRef<unsigned> warpsPerCTA,
                                  ArrayRef<int64_t> blockShape,
                                  unsigned opIdx) {
    auto dpasLayout = DpasEncodingAttr::get(
        &ctx, /*repeatCount=*/8, /*systolicDepth=*/8, /*executionSize=*/16,
        /*opsPerChan=*/2, warpsPerCTA, /*repCluster=*/{4, 2},
        /*threadsPerWarp=*/16);

    auto dpasReps = dpasLayout.getDPASRepetitions(blockShape, opIdx);
    llvm::errs() << "dpas reps size = " << dpasReps.size() << "\n";
    assert(dpasReps.size() == 3);
    llvm::errs() << "dpas reps: ";
    for (size_t i = 0; i < dpasReps.size(); i++) {
      llvm::errs() << dpasReps[i] << " ";
    }
    llvm::errs() << "\n";
    auto dpasRepsUnsigned = SmallVector<unsigned>{
        static_cast<unsigned>(dpasReps[1]), static_cast<unsigned>(dpasReps[2])};
    // TODO: could put the getOrderForDotOperand in the builder?
    return Subgroup2DBlockEncodingAttr::get(
        &ctx, warpsPerCTA,
        CTALayoutAttr::get(&ctx, dpasLayout.getCTAsPerCGA(),
                           dpasLayout.getCTASplitNum(),
                           dpasLayout.getCTAOrder()),
        instrShape, numBlocks, dpasRepsUnsigned,
        getOrderForDotOperand(opIdx, /*rank*/ 2, /*kContig*/ true), kWidth,
        dpasLayout.getThreadsPerWarp());
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

// TODO: is this valid, and how should this layout be created?
TEST_F(LinearLayoutConversionsTest, DISABLED_FP16_32x32x1_M256_N32_K32_A) {

  // Layout for A operand, warpsPerCTA is (8, 4). We have one tile per warp.
  // The load should be 32 by 16 with 2 blocks --> 32 by 32
  // There is one load per warp.

  auto layout = subgroup2DBlockToLinearLayout(
      /*blockShape*/ {256, 32},
      sdb(/*instrShape*/ {32, 32}, /*numBlocks*/ 1, /*kWidth*/ 2,
          /*warpsPerCTA*/ {8, 4},
          /*blockShape*/ {256, 32}, /*opIdx*/ 0),
      /*kWidth*/ 2);
  llvm::errs() << "layout from conversion: " << layout << "\n";
  EXPECT_EQ(
      layout,
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 16}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
           {S("warp"), {{0, 0}, {0, 0}, {32, 0}, {64, 0}, {128, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, FP16_32x16x2_M256_N32_K32_A) {
  // Layout for A operand, warpsPerCTA is (8, 4). We have one tile per warp.
  // The load should be 32 by 16 with 2 blocks --> 32 by 32
  // There is one load per warp.
  auto sdbEncoding = sdb(/*instrShape*/ {32, 16}, /*numBlocks*/ 2, /*kWidth*/ 2,
                         /*warpsPerCTA*/ {8, 4},
                         /*blockShape*/ {256, 32}, /*opIdx*/ 0);
  llvm::errs() << "sdp: " << sdbEncoding << "\n";

  auto layout = subgroup2DBlockToLinearLayout(
      /*blockShape*/ {256, 32}, sdbEncoding,
      /*kWidth*/ 2);
  llvm::errs() << "layout from conversion: " << layout << "\n";
  EXPECT_EQ(
      layout,
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 16}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
           {S("warp"), {{0, 0}, {0, 0}, {32, 0}, {64, 0}, {128, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, FP16_32x16x2_M256_N32_K32_B) {
  // Layout for A operand, warpsPerCTA is (8, 4). We have one tile per warp.
  // The load should be 32 by 16 with 2 blocks --> 32 by 32
  // There are two loads per warp.

  auto sdbEncoding = sdb(/*instrShape*/ {32, 16}, /*numBlocks*/ 2, /*kWidth*/ 2,
                         /*warpsPerCTA*/ {8, 4}, /*blockShape*/ {32, 256},
                         /*opIdx*/ 1);
  llvm::errs() << "sdp: " << sdbEncoding << "\n";

  auto layout = subgroup2DBlockToLinearLayout(
      /*shape*/ {32, 256}, sdbEncoding,
      /*kWidth*/ 2);
  llvm::errs() << "layout from conversion: " << layout << "\n";
  EXPECT_EQ(layout,
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 16}, {0, 128}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 32}, {0, 64}, {0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

} // anonymous namespace
} // namespace mlir::triton::gpu::intel

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
