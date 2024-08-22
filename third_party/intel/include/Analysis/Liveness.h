#ifndef TRITON_INTEL_ANALYSIS_LIVENESS_H
#define TRITON_INTEL_ANALYSIS_LIVENESS_H

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Value.h"
#include <map>

namespace mlir::triton::gpu::intel {

class LiveInterval {
  friend raw_ostream &operator<<(raw_ostream &, const LiveInterval &);

public:
  LiveInterval(Operation *start, Operation *end, Value val)
      : start(start), end(end), liveValue(val) {
    assert(start && end && "start and end operations must not be null");
  }

  Operation *getStart() const { return start; }
  Operation *getEnd() const { return end; }
  Value getValue() const { return liveValue; }

private:
  Operation *start = nullptr; // interval start (inclusive).
  Operation *end = nullptr;   // internal end (inclusive).
  Value liveValue;            // value associated with the live interval.
};

class LivenessAnalysis : public mlir::Liveness {
public:
  LivenessAnalysis(Operation *op);

  /// Returns the number of overlapping live intervals at program point \p op
  /// given the \p livenessInfo of the block containing \p op.
  unsigned
  numOverlappingLiveIntervals(Operation *op,
                              const LivenessBlockInfo &livenessInfo) const;

  /// Returns the max number of overlapping live intervals in the given basic
  /// block.
  /// Note: the given block must be contained by the root operation this
  /// analysis was created for.
  unsigned maxNumOverlappingLiveIntervals(Block *block) const;

  /// Returns the live values at the given program point.
  ValueSetT getLiveValues(Operation *op) const;

  /// Given a basic block \p block, populate the live intervals for the block.
  void computeLiveIntervals(Block &block);

  void printLiveIntervals(raw_ostream &os) const;

private:
  using LiveIntervals = SmallVector<LiveInterval>;
  std::map<Block *, LiveIntervals> blockToLiveIntervals;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_LIVENESS_H