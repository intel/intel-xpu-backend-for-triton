#ifndef TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_PREFETCHCOSTMODEL_H
#define TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_PREFETCHCOSTMODEL_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <array>
#include <cstdint>
#include <optional>

/// Predicts whether dropping a gapped operand's prefetch speeds up a pipelined
/// dot loop; free of MLIR so the arithmetic is unit-testable.
namespace mlir::triton::gpu::intel::prefetchcost {

/// Per-architecture coefficients, per Xe core, named after the spec's symbols.
struct ArchConstants {
  unsigned granuleBytes;     // G
  double dpasFlopsPerClk;    // R_core per core clock, 16-bit DPAS
  double fillBytesPerClk;    // Phi_L1 per core clock
  double dramBytesPerMemClk; // DRAM share per memory clock
  std::array<double, 6> onsetItersByDistance; // i0 at prefetch distance 1..size
  double rampSpanIters;                       // L
};

/// What the compiler reads from the target device.
struct DeviceFacts {
  double coreClockHz;
  double memClockHz; // as the device reports it
};

struct Device {
  ArchConstants arch;
  DeviceFacts facts;
};

/// The device for `deviceArch` with `facts`; nullopt unless the arch has
/// constants (only "bmg" does) and every fact is positive.
std::optional<Device> getDevice(llvm::StringRef deviceArch,
                                const DeviceFacts &facts);

/// One operand's tile: `rows` contiguous runs of `rowBytes`, advanced along
/// the run by `stepBytes` each iteration (0 if the last index does not change).
struct OperandWalk {
  int64_t rows;
  int64_t rowBytes;
  int64_t stepBytes;
  /// Bytes pulled per iteration: a run shorter than `granule` whose step skips
  /// past it costs min(|step|, granule); any other run costs `rowBytes`.
  int64_t fetchedBytes(unsigned granule) const;
  int64_t usedBytes() const;
  /// True when the step skips past a run shorter than `granule`, so the fetch
  /// pulls bytes the walk never reads. A step-0 run never counts, however
  /// short: next to a B with 128 B step-0 rows, dropping a gapped A's prefetch
  /// was still measured faster.
  bool gapped(unsigned granule) const;
};

struct LoopShape {
  llvm::SmallVector<OperandWalk, 2> operands;
  double flopsPerIter;       // 16-bit DPAS-equivalent flops
  int64_t tripCount;         // > 0, except chooseDrop returns nullopt for <= 0
  unsigned prefetchDistance; // num_stages - 1
};

/// Predicted seconds for one work-group's loop with every prefetch kept (`drop`
/// is nullopt) or with operand `*drop`'s removed; chooseDrop uses their ratio.
/// The prefetch distance must index the onset table (1..its size).
double loopSeconds(const LoopShape &shape, const Device &dev,
                   std::optional<unsigned> drop);

/// Index of the operand whose prefetch to drop, or nullopt to keep all. Only a
/// loop with a positive trip count, a prefetch distance in the onset table
/// (distances beyond the table keep) and exactly one gapped operand qualifies,
/// and dropping must be faster by more than `margin`, the model's error band.
std::optional<unsigned> chooseDrop(const LoopShape &shape, const Device &dev,
                                   double margin = 0.15);

} // namespace mlir::triton::gpu::intel::prefetchcost

#endif // TRITON_TRITONINTELGPU_TRANSFORM_PIPELINE_PREFETCHCOSTMODEL_H
