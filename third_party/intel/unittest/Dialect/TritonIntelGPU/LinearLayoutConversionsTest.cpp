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

  // Create a Subgroup2DBlockEncoding layout based on a DPAS layout
  Subgroup2DBlockEncodingAttr
  sdb(ArrayRef<unsigned> instrShape, unsigned numBlocks, unsigned kWidth,
      ArrayRef<unsigned> warpsPerCTA, ArrayRef<unsigned> repCluster,
      ArrayRef<int64_t> blockShape, unsigned opsPerChannel, unsigned opIdx) {
    auto dpasLayout = DpasEncodingAttr::get(
        &ctx, /*repeatCount=*/8, /*systolicDepth=*/8, /*executionSize=*/16,
        opsPerChannel, warpsPerCTA, repCluster,
        /*threadsPerWarp=*/16);

    // TODO: could put the getOrderForDotOperand in the builder?
    auto layout = Subgroup2DBlockEncodingAttr::get(
        &ctx, warpsPerCTA,
        CTALayoutAttr::get(
            &ctx, dpasLayout.getCTAsPerCGA(), // TODO: add to DpasLayout?
            dpasLayout.getCTASplitNum(), dpasLayout.getCTAOrder()),
        instrShape, numBlocks,
        getOrderForDotOperand(opIdx, /*rank*/ 2, /*kContig*/ true), kWidth,
        dpasLayout.getThreadsPerWarp());
    return layout;
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearLayoutConversionsTest, FP32_32x8x2_M256_N128_K32_A) {
  EXPECT_EQ(
      subgroup2DBlockToLinearLayout(
          /*blockShape*/ {256, 32},
          sdb(/*instrShape*/ {32, 8}, /*numBlocks*/ 2, /*kWidth*/ 4,
              /*warpsPerCTA*/ {8, 4}, /*repCluster*/ {4, 1},
              /*blockShape*/ {256, 32}, /*opsPerChannel*/ 1, /*opIdx*/ 0),
          /*kWidth*/ 4),
      LinearLayout(
          {{S("register"), {{2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}, {0, 16}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {1, 0}}},
           {S("warp"), {{0, 0}, {0, 0}, {32, 0}, {64, 0}, {128, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, FP32_32x16x1_M256_N128_K32_B) {
  EXPECT_EQ(
      subgroup2DBlockToLinearLayout(
          /*blockShape*/ {32, 128},
          sdb(/*instrShape*/ {32, 16}, /*numBlocks*/ 1, /*kWidth*/ 4,
              /*warpsPerCTA*/ {8, 4}, /*repCluster*/ {4, 1},
              /*blockShape*/ {32, 128}, /*opsPerChannel*/ 1, /*opIdx*/ 1),
          /*kWidth*/ 4),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 64}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
           {S("warp"), {{0, 16}, {0, 32}, {0, 0}, {0, 0}, {0, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, FP16_32x32x1_M256_N32_K32_A) {
  EXPECT_EQ(
      subgroup2DBlockToLinearLayout(
          /*blockShape*/ {256, 32},
          sdb(/*instrShape*/ {32, 32}, /*numBlocks*/ 1, /*kWidth*/ 2,
              /*warpsPerCTA*/ {8, 4}, /*repCluster*/ {4, 2},
              /*blockShape*/ {256, 32}, /*opsPerChannel*/ 2, /*opIdx*/ 0),
          /*kWidth*/ 2),
      LinearLayout(
          {{S("register"), {{0, 1}, {1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
           {S("lane"), {{0, 2}, {0, 4}, {0, 8}, {0, 16}}},
           {S("warp"), {{0, 0}, {0, 0}, {32, 0}, {64, 0}, {128, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, FP16_32x16x2_M256_N32_K32_A) {
  EXPECT_EQ(
      subgroup2DBlockToLinearLayout(
          /*blockShape*/ {256, 32},
          sdb(/*instrShape*/ {32, 16}, /*numBlocks*/ 2, /*kWidth*/ 2,
              /*warpsPerCTA*/ {8, 4}, /*repCluster*/ {4, 2},
              /*blockShape*/ {256, 32}, /*opsPerChannel*/ 2, /*opIdx*/ 0),
          /*kWidth*/ 2),
      LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 16}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
           {S("warp"), {{0, 0}, {0, 0}, {32, 0}, {64, 0}, {128, 0}}},
           {S("block"), {}}},
          {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, FP16_32x16x2_M256_N32_K32_B) {
  EXPECT_EQ(subgroup2DBlockToLinearLayout(
                /*shape*/ {32, 256},
                sdb(/*instrShape*/ {32, 16}, /*numBlocks*/ 2, /*kWidth*/ 2,
                    /*warpsPerCTA*/ {8, 4}, /*repCluster*/ {4, 2},
                    /*blockShape*/ {32, 256}, /*opsPerChannel*/ 2,
                    /*opIdx*/ 1),
                /*kWidth*/ 2),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 16}, {0, 128}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 32}, {0, 64}, {0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, FP16_16x16x2_M256_N32_K32_B) {
  EXPECT_EQ(subgroup2DBlockToLinearLayout(
                /*shape*/ {32, 256},
                sdb(/*instrShape*/ {16, 16}, /*numBlocks*/ 2, /*kWidth*/ 2,
                    /*warpsPerCTA*/ {8, 4}, /*repCluster*/ {4, 2},
                    /*blockShape*/ {32, 256}, /*opsPerChannel*/ 2,
                    /*opIdx*/ 1),
                /*kWidth*/ 2),
            LinearLayout(
                {{S("register"),
                  {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 16}, {16, 0}, {0, 128}}},
                 {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                 {S("warp"), {{0, 32}, {0, 64}, {0, 0}, {0, 0}, {0, 0}}},
                 {S("block"), {}}},
                {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, I8_16x32x1_M64_N128_K32_A) {
  EXPECT_EQ(
      subgroup2DBlockToLinearLayout(
          /*shape*/ {64, 32},
          sdb(/*instrShape*/ {16, 32}, /*numBlocks*/ 1, /*kWidth*/ 1,
              /*warpsPerCTA*/ {4, 8}, /*repCluster*/ {2, 1},
              /*blockShape*/ {64, 32}, /*opsPerChannel*/ 4,
              /*opIdx*/ 0),
          /*kWidth*/ 1),
      LinearLayout({{S("register"), {{0, 1}, {1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("lane"), {{0, 2}, {0, 4}, {0, 8}, {0, 16}}},
                    {S("warp"), {{0, 0}, {0, 0}, {0, 0}, {16, 0}, {32, 0}}},
                    {S("block"), {}}},
                   {S("dim0"), S("dim1")}));
}

TEST_F(LinearLayoutConversionsTest, I8_32x32x1_M64_N128_K32_B) {
  EXPECT_EQ(
      subgroup2DBlockToLinearLayout(
          /*shape*/ {32, 128},
          sdb(/*instrShape*/ {32, 16}, /*numBlocks*/ 1, /*kWidth*/ 1,
              /*warpsPerCTA*/ {4, 8}, /*repCluster*/ {2, 1},
              /*blockShape*/ {32, 128}, /*opsPerChannel*/ 4,
              /*opIdx*/ 1),
          /*kWidth*/ 1),
      LinearLayout({{S("register"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("warp"), {{0, 16}, {0, 32}, {0, 64}, {0, 0}, {0, 0}}},
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
