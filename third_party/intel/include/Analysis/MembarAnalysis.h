#ifndef TRITON_INTEL_ANALYSIS_MEMBAR_ANALYSIS_H
#define TRITON_INTEL_ANALYSIS_MEMBAR_ANALYSIS_H

#include "triton/Analysis/Function.h"
#include "triton/Analysis/Membar/SmemModel.h"
#include <utility>

namespace mlir::triton::intel {

/// Legacy forward-flow state at the current program point and at function
/// boundaries.
struct MembarInfo {
  /// Effects since the most recent whole-region synchronization.
  BlockInfo pending;

  /// Effects reachable from the function entry block before the first
  /// whole-region synchronization.
  /// It keeps incrementing during the iterative algorithm until
  ///  we note all paths to a basic block has synchronized.
  BlockInfo entryBlockInfo;

  /// Whether every path from the function entry block to the current program
  /// point has synchronized the whole execution region.
  bool allPathsFromEntrySynced = false;

  MembarInfo &join(const MembarInfo &other) {
    pending.join(other.pending);
    entryBlockInfo.join(other.entryBlockInfo);
    allPathsFromEntrySynced &= other.allPathsFromEntrySynced;
    return *this;
  }

  void addBlockInfo(const BlockInfo &blockInfo) {
    if (!allPathsFromEntrySynced)
      entryBlockInfo.join(blockInfo);
    pending.join(blockInfo);
  }

  void sync() {
    pending.sync();
    allPathsFromEntrySynced = true;
  }

  void applyCallSummary(const MembarInfo &callee) {
    if (!allPathsFromEntrySynced)
      entryBlockInfo.join(callee.entryBlockInfo);
    if (callee.allPathsFromEntrySynced)
      sync();
    pending.join(callee.pending);
  }

  template <typename Transform> void transformSlices(Transform transform) {
    pending.transformSlices(transform);
    entryBlockInfo.transformSlices(transform);
  }

  bool operator==(const MembarInfo &) const = default;
};

//===----------------------------------------------------------------------===//
// Shared Memory Barrier Analysis
//===----------------------------------------------------------------------===//

/// Copy of the greedy analysis that AMD keeps in
/// third_party/amd/include/Analysis/Membar.h. Barriers from the Membar cut
/// solver can make IGC run out of scratch space.
/// TODO: Use the cut solver and remove this copy (#9016).
class MembarAnalysis : public triton::PostOrderFunctionAnalysis<MembarInfo> {
public:
  /// Creates a new Membar analysis that generates the shared memory barrier
  /// in the following circumstances:
  /// - RAW: If a shared memory write is followed by a shared memory read, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAR: If a shared memory read is followed by a shared memory write, and
  /// their addresses are intersected, a barrier is inserted.
  /// - WAW: Successive writes to intersecting addresses require ordering.
  /// RAR needs no barrier because no write is performed.
  /// Temporary storage of operations such as Reduce contributes an incoming
  /// scratch write and an outgoing scratch read.
  MembarAnalysis(Allocation &allocation, MembarFilterFn filter,
                 triton::BufferRegionAnalysis &regions,
                 CompletionScopeFn completionScope = nullptr)
      : allocation(allocation), regions(regions), filter(std::move(filter)),
        shared(allocation, regions,
               triton::membar::SMemModel::AccessMode::AllSharedAccesses,
               completionScope) {}

private:
  void updateSuccessor(Operation *terminator, Block *successor,
                       MembarInfo *membarInfo) override;

  void updateExitState(MembarInfo *membarInfo) override;

  void update(Operation *operation, MembarInfo *membarInfo, FuncMapT *funcMap,
              OpBuilder *builder) override;

  void syncIfNeeded(Operation *operation, const BlockInfo &effects,
                    MembarInfo *membarInfo, OpBuilder *builder);

  Allocation &allocation;
  BufferRegionAnalysis &regions;
  MembarFilterFn filter;
  triton::membar::SMemModel shared;
};

/// Inserts shared-memory and operation rendezvous barriers across a module,
/// before async-completion analyses consume the synchronization points.
/// Function summaries retain entry-prefix and pending exit states for calls.
class ModuleMembarAnalysis {
public:
  ModuleMembarAnalysis(ModuleAllocation &moduleAllocation,
                       MembarFilterFn filter = nullptr)
      : moduleAllocation(moduleAllocation), filter(std::move(filter)) {}

  void run() {
    // Geometry and interval slices use the same completed allocation, including
    // backend scratch sizes and shared-memory partition offsets.
    auto solver = createDataFlowSolver();
    auto *regions = solver->load<triton::BufferRegionAnalysis>(
        triton::BufferRegionAnalysis::Mode::AllMemory, &moduleAllocation);
    if (failed(solver->initializeAndRun(moduleAllocation.getModuleOp())))
      llvm::report_fatal_error("failed to analyze allocated buffer regions");
    runAnalysis<MembarAnalysis>(*regions);
  }

  template <typename AnalysisT>
  void runAnalysis(triton::BufferRegionAnalysis &regions) {
    AnalysisT::runModule(
        moduleAllocation.getModuleOp(), [&](FunctionOpInterface function) {
          auto &allocation = *moduleAllocation.getFuncData(function);
          return AnalysisT(allocation, filter, regions);
        });
  }

private:
  ModuleAllocation &moduleAllocation;
  MembarFilterFn filter;
};

} // namespace mlir::triton::intel

#endif // TRITON_INTEL_ANALYSIS_MEMBAR_ANALYSIS_H
