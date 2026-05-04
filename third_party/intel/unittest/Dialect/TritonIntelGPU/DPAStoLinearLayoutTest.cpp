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
  // Scale operands are warp-broadcast (see BlockScaledDPAStoLinearLayout): all
  // warp bases are zero. The tile shrinks along the M dimension as a result,
  // so the remaining dim0 coverage is supplied by additional register bases
  // instead of by warp identity bases.
  EXPECT_EQ(
      BlockScaledDPAStoLinearLayout({128, 2},
                                    dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32), 3),
      LinearLayout(
          {
              {S("register"), {{8, 0}, {16, 0}, {0, 1}, {32, 0}, {64, 0}}},
              {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {0, 0}, {0, 0}}},
              {S("warp"), {{0, 0}, {0, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 4},
                dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32, std::make_optional(2)),
                3),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1}, {8, 0}, {16, 0}, {0, 2}, {32, 0}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {0, 0}, {0, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(
      BlockScaledDPAStoLinearLayout({128, 2},
                                    dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16), 3),
      LinearLayout(
          {
              {S("register"), {{8, 0}, {16, 0}, {0, 1}, {32, 0}, {64, 0}}},
              {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {0, 0}}},
              {S("warp"), {{0, 0}, {0, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));

  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 4},
                dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16, std::make_optional(2)),
                3),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 1}, {8, 0}, {16, 0}, {0, 2}, {32, 0}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {0, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));

  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {1, 128, 4}, dpas({1, 2, 2}, 8, 8, 16, 4, {1, 4, 2}, 16), 3),
            LinearLayout(
                {
                    {S("register"),
                     {{0, 8, 0},
                      {0, 16, 0},
                      {0, 0, 1},
                      {0, 0, 2},
                      {0, 32, 0},
                      {0, 64, 0}}},
                    {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 0, 0}}},
                    {S("warp"), {{0, 0, 0}, {0, 0, 0}}},
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
                {0, 32, 0},
                {0, 64, 0},
                {1, 0, 0},
                {2, 0, 0},
                {4, 0, 0},
                {8, 0, 0}}},
              {S("lane"), {{0, 1, 0}, {0, 2, 0}, {0, 4, 0}, {0, 8, 0}}},
              {S("warp"), {{0, 0, 0}, {0, 0, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1"), S("dim2")}));
}

TEST_F(DPAStoLinearLayoutTest, DPAS_OperandScaleB) {
  // Scale operands are warp-broadcast (see BlockScaledDPAStoLinearLayout): all
  // warp bases are zero. An extra register basis covers the dim0 elements
  // that used to be distributed across warps.
  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32), 4),
            LinearLayout(
                {
                    {S("register"), {{16, 0}, {0, 1}, {32, 0}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(BlockScaledDPAStoLinearLayout(
                {128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16), 4),
            LinearLayout(
                {
                    {S("register"), {{16, 0}, {0, 1}, {32, 0}, {64, 0}}},
                    {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                    {S("warp"), {{0, 0}, {0, 0}}},
                    {S("block"), {}},
                },
                {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      BlockScaledDPAStoLinearLayout(
          {128, 4},
          dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32, std::make_optional(2)), 4),
      LinearLayout(
          {
              {S("register"), {{0, 1}, {16, 0}, {0, 2}, {32, 0}, {64, 0}}},
              {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 0}}},
              {S("warp"), {{0, 0}, {0, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
  EXPECT_EQ(
      BlockScaledDPAStoLinearLayout(
          {128, 4},
          dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16, std::make_optional(2)), 4),
      LinearLayout(
          {
              {S("register"), {{0, 1}, {16, 0}, {0, 2}, {32, 0}, {64, 0}}},
              {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
              {S("warp"), {{0, 0}, {0, 0}}},
              {S("block"), {}},
          },
          {S("dim0"), S("dim1")}));
}

TEST_F(DPAStoLinearLayoutTest, BlockScaledScaleOperandWarpBroadcast) {
  // Preparatory audit for SpatialReuseAnalysis (see
  // annotate-cc-p1-spatial-reuse.md): the scale-operand branch of
  // BlockScaledDPAStoLinearLayout must produce a warp-broadcast LinearLayout
  // (zero warp basis across all output dims). Scales are consumed by the same
  // DPAS-backed matmul as the A/B operands and benefit from the same
  // cross-subgroup L1 reuse; a non-zero warp basis here would force
  // AnnotateCacheControl to keep a use-site scan as a workaround.
  //
  // Coverage: opIdx in {3, 4} (scale A, scale B) x MXFP8 (block 16, no
  // fp4Kpack) / MXFP4 (block 32, fp4Kpack=2) x warp sizes 16 and 32 x 2D and
  // 3D shapes, mirroring the configurations already asserted in
  // DPAS_OperandScaleA / DPAS_OperandScaleB above.
  StringAttr kWarp = S("warp");

  struct Config {
    SmallVector<int64_t> shape;
    DpasEncodingAttr dpas;
    unsigned opIdx;
    StringRef name;
  };

  SmallVector<Config> configs = {
      // 2D, warp size 32, MXFP8, scale A / scale B
      {{128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32), 3, "mxfp8-tpw32-A"},
      {{128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32), 4, "mxfp8-tpw32-B"},
      // 2D, warp size 16, MXFP8, scale A / scale B
      {{128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16), 3, "mxfp8-tpw16-A"},
      {{128, 2}, dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16), 4, "mxfp8-tpw16-B"},
      // 2D, warp size 32, MXFP4, scale A / scale B
      {{128, 4},
       dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32, std::make_optional(2u)),
       3,
       "mxfp4-tpw32-A"},
      {{128, 4},
       dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 32, std::make_optional(2u)),
       4,
       "mxfp4-tpw32-B"},
      // 2D, warp size 16, MXFP4, scale A / scale B
      {{128, 4},
       dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16, std::make_optional(2u)),
       3,
       "mxfp4-tpw16-A"},
      {{128, 4},
       dpas({2, 2}, 8, 8, 16, 4, {4, 2}, 16, std::make_optional(2u)),
       4,
       "mxfp4-tpw16-B"},
      // 3D, warp size 16, MXFP8, scale A
      {{1, 128, 4},
       dpas({1, 2, 2}, 8, 8, 16, 4, {1, 4, 2}, 16),
       3,
       "mxfp8-3d-A"},
      // 3D, warp size 16, MXFP4, scale B (mirrors the existing
      // DPAS_OperandScaleA 3D case shape/dpas, flipped to opIdx=4)
      {{16, 128, 4},
       dpas({2, 1, 2}, 8, 8, 16, 4, {1, 4, 2}, 16, std::make_optional(2u)),
       4,
       "mxfp4-3d-B"},
  };

  for (const Config &cfg : configs) {
    LinearLayout ll =
        BlockScaledDPAStoLinearLayout(cfg.shape, cfg.dpas, cfg.opIdx);
    auto outDims = llvm::to_vector(ll.getOutDimNames());
    EXPECT_TRUE(ll.sublayoutIsZero({kWarp}, outDims))
        << "config=" << cfg.name.str() << " opIdx=" << cfg.opIdx;
  }
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
