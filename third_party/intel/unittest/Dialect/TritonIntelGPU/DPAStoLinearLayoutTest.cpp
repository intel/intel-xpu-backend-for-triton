#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
std::ostream &operator<<(std::ostream &os, StringAttr str) {
  os << str.str();
  return os;
}
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::gpu::intel;

namespace mlir::triton::gpu {
namespace {

class DPAStoLinearLayoutTest : public ::testing::Test {
public:
  void SetUp() { ctx.getOrLoadDialect<TritonIntelGPUDialect>(); }

  DpasEncodingAttr dpas(ArrayRef<unsigned> warps, unsigned repeatCount,
                        unsigned systolicDepth, unsigned executionSize,
                        unsigned opsPerChannel, ArrayRef<unsigned> repCluster,
                        unsigned threadsPerWarp) {
    return DpasEncodingAttr::get(&ctx, repeatCount, systolicDepth,
                                 executionSize, opsPerChannel, warps,
                                 repCluster, threadsPerWarp);
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(DPAStoLinearLayoutTest, DPAS_perInst) {
  // Default: Operand C
  EXPECT_EQ(DPAStoLinearLayout({8, 16}, dpas({1, 1}, 8, 8, 16, 2, {1, 1}, 32)),
            LinearLayout(
                {
                    {S("register"), {{2, 0}, {4, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(DPAStoLinearLayout({8, 16}, dpas({1, 1}, 8, 8, 16, 1, {1, 1}, 16)),
            LinearLayout(
                {
                    {S("register"), {{1, 0}, {2, 0}, {4, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  // Test Operand A (opIdx=0)
  EXPECT_EQ(
      DPAStoLinearLayout({8, 32}, dpas({1, 1}, 8, 8, 16, 4, {1, 1}, 32), 0),
      LinearLayout(
          {
              {S("register"), {{0, 1}, {2, 0}, {4, 0}}},
              {S("lane"), {{0, 2}, {0, 4}, {0, 8}, {0, 16}, {1, 0}}},
              {S("warp"), {}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      DPAStoLinearLayout({8, 16}, dpas({1, 1}, 8, 8, 16, 2, {1, 1}, 32), 0),
      LinearLayout(
          {
              {S("register"), {{2, 0}, {4, 0}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
              {S("warp"), {}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      DPAStoLinearLayout({8, 8}, dpas({1, 1}, 8, 8, 16, 1, {1, 1}, 32), 0),
      LinearLayout(
          {
              {S("register"), {{4, 0}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {1, 0}, {2, 0}}},
              {S("warp"), {}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  // Test Operand B (opIdx=1)
  EXPECT_EQ(
      DPAStoLinearLayout({32, 16}, dpas({1, 1}, 8, 8, 16, 4, {1, 1}, 32), 1),
      LinearLayout(
          {
              {S("register"), {{1, 0}, {2, 0}, {8, 0}, {16, 0}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}}},
              {S("warp"), {}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      DPAStoLinearLayout({16, 16}, dpas({1, 1}, 8, 8, 16, 2, {1, 1}, 32), 1),
      LinearLayout(
          {
              {S("register"), {{1, 0}, {4, 0}, {8, 0}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {2, 0}}},
              {S("warp"), {}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      DPAStoLinearLayout({8, 16}, dpas({1, 1}, 8, 8, 16, 1, {1, 1}, 32), 1),
      LinearLayout(
          {
              {S("register"), {{2, 0}, {4, 0}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
              {S("warp"), {}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
}

TEST_F(DPAStoLinearLayoutTest, DPAS_withRepCluster) {
  EXPECT_EQ(DPAStoLinearLayout({32, 32}, dpas({1, 1}, 8, 8, 16, 2, {4, 2}, 16)),
            LinearLayout(
                {
                    {S("register"),
                     {{1, 0}, {2, 0}, {4, 0}, {0, 16}, {8, 0}, {16, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  // Test Operand A (opIdx=0)
  EXPECT_EQ(
      DPAStoLinearLayout({32, 16}, dpas({1, 1}, 8, 8, 16, 2, {4, 2}, 32), 0),
      LinearLayout(
          {
              {S("register"), {{2, 0}, {4, 0}, {8, 0}, {16, 0}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
              {S("warp"), {}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  // Test Operand B (opIdx=1)
  EXPECT_EQ(
      DPAStoLinearLayout({16, 32}, dpas({1, 1}, 8, 8, 16, 2, {4, 2}, 32), 1),
      LinearLayout(
          {
              {S("register"), {{1, 0}, {4, 0}, {8, 0}, {0, 16}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {2, 0}}},
              {S("warp"), {}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(DPAStoLinearLayout({32, 32}, dpas({1, 1}, 8, 8, 16, 1, {4, 2}, 16)),
            LinearLayout(
                {
                    {S("register"),
                     {{1, 0}, {2, 0}, {4, 0}, {0, 16}, {8, 0}, {16, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("warp"), {}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(DPAStoLinearLayoutTest, DPAS_withWarp) {
  EXPECT_EQ(DPAStoLinearLayout({32, 32}, dpas({4, 1}, 8, 8, 16, 2, {1, 2}, 16)),
            LinearLayout(
                {
                    {S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 16}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("warp"), {{8, 0}, {16, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(DPAStoLinearLayout({64, 64}, dpas({2, 2}, 8, 8, 16, 1, {4, 2}, 32)),
            LinearLayout(
                {
                    {S("register"), {{2, 0}, {4, 0}, {0, 16}, {8, 0}, {16, 0}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                    {S("warp"), {{0, 32}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
}

TEST_F(DPAStoLinearLayoutTest, DPAS_withWarpOperandA) {
  EXPECT_EQ(
      DPAStoLinearLayout({64, 64}, dpas({2, 2}, 8, 8, 16, 2, {4, 2}, 32), 0),
      LinearLayout(
          {
              {S("register"),
               {{2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 16}, {0, 32}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
              {S("warp"), {{0, 0}, {32, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
}

TEST_F(DPAStoLinearLayoutTest, DPAS_withWarpOperandB) {
  EXPECT_EQ(
      DPAStoLinearLayout({64, 64}, dpas({2, 2}, 8, 8, 16, 2, {4, 2}, 32), 1),
      LinearLayout(
          {
              {S("register"),
               {{1, 0}, {4, 0}, {8, 0}, {0, 16}, {16, 0}, {32, 0}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {2, 0}}},
              {S("warp"), {{0, 32}, {0, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
}

TEST_F(DPAStoLinearLayoutTest, DPAS_withDPASRepetitions) {
  EXPECT_EQ(DPAStoLinearLayout({64, 64}, dpas({2, 1}, 8, 8, 16, 2, {4, 2}, 32)),
            LinearLayout(
                {
                    {S("register"),
                     {{2, 0}, {4, 0}, {0, 16}, {8, 0}, {16, 0}, {0, 32}}},
                    {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                    {S("warp"), {{32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      DPAStoLinearLayout({128, 128}, dpas({2, 2}, 8, 8, 16, 2, {2, 2}, 32)),
      LinearLayout(
          {
              {S("register"),
               {{2, 0}, {4, 0}, {0, 16}, {8, 0}, {0, 64}, {32, 0}, {64, 0}}},
              {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
              {S("warp"), {{0, 32}, {16, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
}

} // anonymous namespace
} // namespace mlir::triton::gpu

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
