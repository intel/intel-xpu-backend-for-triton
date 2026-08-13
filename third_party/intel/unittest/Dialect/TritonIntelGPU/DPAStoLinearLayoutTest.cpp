#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
  void SetUp() {
    ctx.getOrLoadDialect<TritonIntelGPUDialect>();
    ctx.getOrLoadDialect<mlir::triton::gpu::TritonGPUDialect>();
  }

  DpasEncodingAttr dpas(ArrayRef<unsigned> warps, unsigned repeatCount,
                        unsigned systolicDepth, unsigned executionSize,
                        unsigned opsPerChannel, ArrayRef<unsigned> repCluster,
                        unsigned threadsPerWarp,
                        std::optional<unsigned> fp4KPack = std::nullopt) {
    return DpasEncodingAttr::get(&ctx, repeatCount, systolicDepth,
                                 executionSize, opsPerChannel, warps,
                                 repCluster, threadsPerWarp, fp4KPack);
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

TEST_F(DPAStoLinearLayoutTest, DPAS_withWarpOperandB_128x256x128) {
  // Confirms: for 128×256×128 MXFP4 on BMG, DPAStoLinearLayout for B operand
  // correctly produces {1,0} at register position 0 — the nibble-selector
  // basis. This means the register-spill problem does NOT originate here. The
  // {32,0} at position 0 in the actual #linear1 from TTGIR comes from
  // cvtDotOperand in DecomposeScaledBlocked using a blocked-encoding parent,
  // which produces a different layout than the DPAS-encoding parent.
  //
  // #mma = dpas{RC=8,SD=8,ES=16,opc=2,tPW=16,warps=[4,8],repCluster=[4,2]}
  EXPECT_EQ(
      DPAStoLinearLayout({128, 256}, dpas({4, 8}, 8, 8, 16, 2, {4, 2}, 16), 1)
          .getBases()
          .at(S("register"))[0],
      (std::vector<int32_t>{1, 0}))
      << "K-stride-1 should be at register position 0 for fp4_to_fp backward "
         "inference compatibility";
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

TEST_F(DPAStoLinearLayoutTest, DPAS_OperandScaleA) {
  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32), 3),
            LinearLayout(
                {
                    {S("register"), {{8, 0}, {16, 0}, {0, 1}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {0, 0}, {0, 0}}},
                    {S("warp"), {{0, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 4},
                dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32, std::make_optional(2)),
                3),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {16, 0}, {0, 2}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {0, 0}, {0, 0}}},
                    {S("warp"), {{0, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16), 3),
            LinearLayout(
                {
                    {S("register"), {{8, 0}, {16, 0}, {0, 1}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {0, 0}}},
                    {S("warp"), {{0, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 4},
                dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16, std::make_optional(2)),
                3),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {8, 0}, {16, 0}, {0, 2}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {0, 0}}},
                    {S("warp"), {{0, 0}, {32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {1, 128, 4}, dpas({1, 2, 2}, 8, 8, 16, 4, {1, 4, 2}, 16), 3),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 8, 0}, {0, 16, 0}, {0, 0, 1}, {0, 0, 2}, {0, 64, 0}}},
                    {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 0, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 32, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1"), S("dim2")}));

  EXPECT_EQ(
      BlockScaledDPAStoLinearLayout(
          {16, 128, 4},
          dpas({2, 1, 2}, 8, 8, 16, 4, {1, 4, 2}, 16, std::make_optional(2)),
          4),
      LinearLayout(
          {
              {S("register"),
               {{0, 0, 1},
                {0, 16, 0},
                {0, 0, 2},
                {0, 64, 0},
                {2, 0, 0},
                {4, 0, 0},
                {8, 0, 0}}},
              {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}}},
              {S("warp"), {{0, 32, 0}, {1, 0, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(DPAStoLinearLayoutTest, DPAS_OperandScaleB) {
  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32), 4),
            LinearLayout(
                {
                    {S("register"), {{16, 0}, {0, 1}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                    {S("warp"), {{32, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16), 4),
            LinearLayout(
                {
                    {S("register"), {{16, 0}, {0, 1}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("warp"), {{32, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 4},
                dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32, std::make_optional(2)),
                4),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {16, 0}, {0, 2}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                    {S("warp"), {{32, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 4},
                dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16, std::make_optional(2)),
                4),
            LinearLayout(
                {
                    {S("register"), {{0, 1}, {16, 0}, {0, 2}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("warp"), {{32, 0}, {0, 0}}},
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
