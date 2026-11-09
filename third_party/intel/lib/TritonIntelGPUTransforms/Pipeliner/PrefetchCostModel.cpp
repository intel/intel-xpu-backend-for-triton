#include "PrefetchCostModel.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace mlir::triton::gpu::intel::prefetchcost {

// Xe2 discrete, fit on the B580 (issue #8652) with split-K GEMMs (256x256x32
// bf16, 1 WG/core) at PR #8520's demand-load order (A issued last).
// The onset, span and DRAM terms include slack relative to the B580 fits. They
// are not validated bounds for other Xe2 parts or for launches with few
// work-groups, so a regression is still possible there.
static const ArchConstants kXe2Discrete = {
    .granuleBytes = 256,         // measured: 64, 128 B rows waste; A/B 2.64x.
    .dpasFlopsPerClk = 1712,     // 84% of the 2048 peak; contiguous grid.
    .fillBytesPerClk = 19.3,     // L1 fill roof, from a fill-rate sweep.
    .dramBytesPerMemClk = 1.035, // bound: fit 0.828 + 25%, unshared ceiling.
    // bound: fit + 25%, rounded up to 10; fits by distance 136 / 102 / 104 /
    // 111 / 107 / 98. Distances beyond the table keep until measured.
    .onsetItersByDistance = {170, 130, 130, 140, 140, 130},
    .rampSpanIters = 650, // bound: fit 500 + 30%.
};

// The ramp sum is exact up to this trip count and its integral beyond, which
// bounds the compile-time cost.
static constexpr int64_t kMaxExactSumTrips = 65536;

std::optional<Device> getDevice(llvm::StringRef deviceArch,
                                const DeviceFacts &facts) {
  bool positive = facts.coreClockHz > 0 && facts.memClockHz > 0;
  if (deviceArch != "bmg" || !positive)
    return std::nullopt;
  // A missing initializer would be zero, which only adds drops.
  assert(llvm::all_of(kXe2Discrete.onsetItersByDistance,
                      [](double v) { return v > 0; }));
  return Device{kXe2Discrete, facts};
}

int64_t OperandWalk::fetchedBytes(unsigned granule) const {
  uint64_t step = llvm::AbsoluteValue(stepBytes); // total, even for INT64_MIN
  if (step <= static_cast<uint64_t>(rowBytes) || rowBytes >= granule)
    return rows * rowBytes;
  return rows * static_cast<int64_t>(std::min<uint64_t>(step, granule));
}

int64_t OperandWalk::usedBytes() const { return rows * rowBytes; }

bool OperandWalk::gapped(unsigned granule) const {
  return fetchedBytes(granule) > usedBytes();
}

static double rampSum(int64_t trips, double onset, double span) {
  if (trips <= kMaxExactSumTrips) {
    double sum = 0;
    for (int64_t i = 0; i < trips; ++i)
      sum += std::sqrt(std::clamp((i - onset) / span, 0.0, 1.0));
    return sum;
  }
  // The ramp never falls, so its integral to the last iteration bounds the sum
  // from below, which can only remove drops.
  double last = trips - 1;
  double x = std::clamp((last - onset) / span, 0.0, 1.0);
  return span * (2.0 / 3.0) * std::pow(x, 1.5) +
         std::max(0.0, last - onset - span);
}

double loopSeconds(const LoopShape &shape, const Device &dev,
                   std::optional<unsigned> drop) {
  assert(shape.tripCount > 0 && "tripCount must be positive");
  assert((!drop || *drop < shape.operands.size()) && "drop is not an operand");
  const ArchConstants &arch = dev.arch;
  assert(shape.prefetchDistance >= 1 &&
         shape.prefetchDistance <= arch.onsetItersByDistance.size() &&
         "prefetch distance is not in the onset table");
  double fetched = 0;
  bool anyGapped = false;
  for (const OperandWalk &op : shape.operands) {
    fetched += op.fetchedBytes(arch.granuleBytes);
    anyGapped |= op.gapped(arch.granuleBytes);
  }
  double trips = shape.tripCount;
  double coreClockHz = dev.facts.coreClockHz;
  double computeSec = shape.flopsPerIter / (arch.dpasFlopsPerClk * coreClockHz);
  double fillSec = fetched / (arch.fillBytesPerClk * coreClockHz);
  // Co-resident work-groups scale every term alike, so one is priced alone.
  double floorSec = std::max(computeSec, fillSec);
  // Dropping the prefetch is modeled as removing the ramp.
  if (drop || !anyGapped || fillSec <= computeSec)
    return trips * floorSec;
  // A fill-bound loop with a gapped prefetch loses in-wave panel sharing as
  // it runs, ramping to the unshared DRAM ceiling. The onset is an iteration
  // count fixed per prefetch distance: the measured onset did not grow when
  // fewer bytes were fetched.
  double ceilingSec =
      fetched / (arch.dramBytesPerMemClk * dev.facts.memClockHz);
  double rampedIters = rampSum(
      shape.tripCount, arch.onsetItersByDistance[shape.prefetchDistance - 1],
      arch.rampSpanIters);
  return trips * floorSec + (ceilingSec - floorSec) * rampedIters;
}

std::optional<unsigned> chooseDrop(const LoopShape &shape, const Device &dev,
                                   double margin) {
  if (shape.tripCount <= 0 || shape.prefetchDistance == 0 ||
      shape.prefetchDistance > dev.arch.onsetItersByDistance.size())
    return std::nullopt;
  auto isGapped = [&](const OperandWalk &op) {
    return op.gapped(dev.arch.granuleBytes);
  };
  // Dropping one of two or more gapped operands was never measured.
  if (llvm::count_if(shape.operands, isGapped) != 1)
    return std::nullopt;
  auto drop = static_cast<unsigned>(llvm::find_if(shape.operands, isGapped) -
                                    shape.operands.begin());
  if (loopSeconds(shape, dev, std::nullopt) >
      (1 + margin) * loopSeconds(shape, dev, drop))
    return drop;
  return std::nullopt;
}

} // namespace mlir::triton::gpu::intel::prefetchcost
