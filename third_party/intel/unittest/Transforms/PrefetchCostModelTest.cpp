#include "intel/lib/TritonIntelGPUTransforms/Pipeliner/PrefetchCostModel.h"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

using namespace mlir::triton::gpu::intel::prefetchcost;

namespace {

// Expected seconds are model predictions printed to three significant digits.
constexpr double kTol = 0.03;
constexpr double kFlops = 2.0 * 256 * 256 * 32;
// Split-K tile: A steps 4x past its 64-byte runs, B steps across its runs.
constexpr OperandWalk kGap4A{256, 64, 256};
constexpr OperandWalk kB{32, 512, 0};
constexpr OperandWalk kContiguousA{256, 64, 64};
constexpr DeviceFacts kB580{.coreClockHz = 2.85e9, .memClockHz = 19.0e9};

Device b580() { return getDevice("bmg", kB580).value(); }

Device b580With(double coreClockHz, double memClockHz) {
  return getDevice("bmg", {coreClockHz, memClockHz}).value();
}

LoopShape loop(OperandWalk a, OperandWalk b, int64_t tripCount,
               double flops = kFlops, unsigned distance = 3) {
  return {{a, b}, flops, tripCount, distance};
}

double keep(const LoopShape &shape, const Device &dev = b580()) {
  return loopSeconds(shape, dev, std::nullopt);
}

double dropA(const LoopShape &shape, const Device &dev = b580()) {
  return loopSeconds(shape, dev, 0u);
}

// The first gap-4 trip count at which keeping is slower than dropping A by
// more than the 15% margin.
int64_t marginFlip(const Device &dev, unsigned distance = 3) {
  auto ratio = [&](int64_t n) {
    LoopShape shape = loop(kGap4A, kB, n, kFlops, distance);
    return keep(shape, dev) / dropA(shape, dev);
  };
  int64_t flip = 1;
  while (flip < 1000 && ratio(flip) <= 1.15)
    ++flip;
  return flip;
}

TEST(PrefetchCostModelTest, HugeTripCount_Finite) {
  LoopShape huge = loop(kGap4A, kB, int64_t{1} << 40);
  EXPECT_TRUE(std::isfinite(keep(huge)));
  EXPECT_EQ(chooseDrop(huge, b580()), 0u);
  double exactPerIter = keep(loop(kGap4A, kB, 65536)) / 65536;
  double closedPerIter = keep(loop(kGap4A, kB, 65537)) / 65537;
  EXPECT_NEAR(closedPerIter / exactPerIter, 1, 1e-3);
}

// Past the exact-sum limit every iteration costs the ceiling, so the closed
// form must not exceed the exact sum extended at that cost.
TEST(PrefetchCostModelTest, ClosedFormTail_UnderExactSum) {
  double last = keep(loop(kGap4A, kB, 65536));
  double perIter = last - keep(loop(kGap4A, kB, 65535));
  for (int64_t n : {int64_t{65537}, int64_t{1} << 20})
    EXPECT_LE(keep(loop(kGap4A, kB, n)),
              (last + (n - 65536) * perIter) * (1 + 1e-9))
        << "trips " << n;
}

TEST(PrefetchCostModelTest, TwoGapped_Keeps) {
  EXPECT_EQ(chooseDrop(loop(kGap4A, kGap4A, 224), b580()), std::nullopt);
}

// The onset is looked up by prefetch distance; a distance outside the table was
// never measured, so it keeps at every trip count.
TEST(PrefetchCostModelTest, OnsetByDistance_MovesFlip) {
  EXPECT_EQ(marginFlip(b580(), 1), 259);
  EXPECT_EQ(marginFlip(b580(), 2), 207);
  EXPECT_EQ(marginFlip(b580(), 3), 207);
  EXPECT_EQ(marginFlip(b580(), 4), 220);
  EXPECT_EQ(marginFlip(b580(), 5), 220);
  EXPECT_EQ(marginFlip(b580(), 6), 207);
  unsigned size = b580().arch.onsetItersByDistance.size();
  for (unsigned distance : {0u, size + 1, size + 4})
    for (int64_t n = 1; n <= int64_t{1} << 40; n *= 2)
      ASSERT_EQ(chooseDrop(loop(kGap4A, kB, n, kFlops, distance), b580()),
                std::nullopt)
          << "distance " << distance << ", trips " << n;
}

// std::array zero-fills a missing initializer; a zero onset only adds drops.
TEST(PrefetchCostModelTest, OnsetTable_AllPositive) {
  Device dev = b580();
  for (size_t i = 0; i < dev.arch.onsetItersByDistance.size(); ++i)
    EXPECT_GT(dev.arch.onsetItersByDistance[i], 0) << "distance " << i + 1;
}

TEST(PrefetchCostModelTest, GetDevice_NeedsArchRowAndPositiveFacts) {
  EXPECT_FALSE(getDevice("pvc", kB580).has_value());
  EXPECT_FALSE(getDevice("bmg", {0, 19.0e9}).has_value());
  EXPECT_FALSE(getDevice("bmg", {2.85e9, 0}).has_value());
}

// The cases below pin rules the cases above leave unexercised.
TEST(PrefetchCostModelTest, WideRow_FetchesRow) {
  EXPECT_EQ((OperandWalk{32, 512, 1024}.fetchedBytes(256)), 32 * 512);
}

TEST(PrefetchCostModelTest, Step0ShortRow_NotGapped) {
  OperandWalk walk{32, 128, 0};
  EXPECT_EQ(walk.fetchedBytes(256), 32 * 128);
  EXPECT_FALSE(walk.gapped(256));
}

// The most negative step has no signed magnitude; it still reads as gapped.
TEST(PrefetchCostModelTest, Int64MinStep_Gapped) {
  OperandWalk walk{256, 64, std::numeric_limits<int64_t>::min()};
  EXPECT_EQ(walk.fetchedBytes(256), 256 * 256);
  EXPECT_TRUE(walk.gapped(256));
}

// BK=64 rows are 128 B, under the granule, so its gap-4 walk is gapped too.
TEST(PrefetchCostModelTest, BK64_128BRowsGapped_DropsA) {
  OperandWalk a{256, 128, 512}, b{64, 512, 0};
  EXPECT_EQ(a.fetchedBytes(256) + b.fetchedBytes(256), 98304);
  LoopShape shape = loop(a, b, 224, 2 * kFlops);
  EXPECT_EQ(chooseDrop(shape, b580()), 0u);
  EXPECT_NEAR(keep(shape) / 224 / 2.13e-6, 1, kTol);
  EXPECT_NEAR(dropA(shape) / 224 / 1.79e-6, 1, kTol);
}

TEST(PrefetchCostModelTest, NonPositiveTripCount_Keeps) {
  EXPECT_EQ(chooseDrop(loop(kGap4A, kB, 0), b580()), std::nullopt);
  EXPECT_EQ(chooseDrop(loop(kGap4A, kB, -1), b580()), std::nullopt);
}

TEST(PrefetchCostModelTest, FillBoundUngapped_NoRamp) {
  LoopShape shape = loop(kContiguousA, kB, 224, kFlops / 2);
  EXPECT_NEAR(keep(shape) / 224 / 0.596e-6, 1, kTol);
}

// A slower core clock lowers the fill roof, which shrinks the gap-4 penalty,
// and slows the compute-bound loop.
TEST(PrefetchCostModelTest, SlowerCoreClock_ScalesRoofs) {
  Device dev = b580With(2.4e9, 19.0e9);
  EXPECT_EQ(chooseDrop(loop(kGap4A, kB, 224), dev), std::nullopt);
  EXPECT_EQ(chooseDrop(loop(kGap4A, kB, 256), dev), 0u);
  EXPECT_NEAR(marginFlip(dev), 229, 2);
  EXPECT_NEAR(keep(loop(kContiguousA, kB, 224), dev) / 224 / 1.02e-6, 1, kTol);
}

// A 40 GHz memory clock lowers the ratio of fill rate to DRAM share from 2.80
// to 1.33, which delays the crossover.
TEST(PrefetchCostModelTest, FasterMemoryClock_DelaysCrossover) {
  EXPECT_NEAR(marginFlip(b580With(2.85e9, 40e9)), 620, 2);
}

} // namespace
