// This file declares the RegisterPressureAnalysis class, which computes
// per-thread register pressure in bytes using liveness information and
// per-thread element distribution from distributed encodings.

#ifndef TRITON_INTEL_ANALYSIS_REGISTER_PRESSURE_H
#define TRITON_INTEL_ANALYSIS_REGISTER_PRESSURE_H

#include "intel/include/Analysis/Liveness.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::gpu::intel {

/// Options controlling register pressure analysis fidelity.
struct RegisterPressureOptions {
  /// If true, exclude rematerializable values (cheap constants and similar
  /// ops) from pressure computation. These can be regenerated cheaply rather
  /// than held in registers.
  bool excludeRematerializable = true;

  /// If true, ensure loop-carried values (iter args in scf.for) are counted
  /// across the entire loop body. The base liveness analysis already handles
  /// this via block live-in, so this option is reserved for potential future
  /// refinements but currently has no effect.
  bool countLoopCarried = true;
};

/// Analysis that computes per-thread GRF register pressure in bytes.
///
/// This analysis builds on LivenessAnalysis and weights each live value by its
/// per-thread size in bytes. For distributed tensors, the size is computed
/// using the encoding's element distribution. For scalars, the size is the
/// element bitwidth in bytes.
///
/// The unit is **per-lane bytes** ("thread" = one SIMD lane, Triton's usual
/// convention), NOT the per-hardware-thread unit `getGRFBytesPerThread`
/// returns. Use `getPerLaneGRFBudgetInBytes` to compare against a GRF budget.
class RegisterPressureAnalysis {
public:
  /// Construct the analysis for the given root operation.
  explicit RegisterPressureAnalysis(Operation *op,
                                    RegisterPressureOptions opts = {});

  /// Returns the per-thread register pressure in bytes at the given operation,
  /// accounting for all live values at that program point.
  unsigned pressureAt(Operation *op) const;

  /// Returns the peak per-thread register pressure in bytes within the given
  /// block.
  unsigned peakPressure(Block *block) const;

  /// Returns the peak per-thread register pressure in bytes within the given
  /// loop, considering all blocks in the loop body region.
  unsigned peakPressure(LoopLikeOpInterface loop) const;

  /// Returns the per-thread register pressure in bytes from the values live-in
  /// to the given block (i.e. defined outside and used inside). Honors
  /// excludeRematerializable. Returns 0 for a block with no liveness info
  /// (e.g. unreachable).
  unsigned liveInPressure(Block *block) const;

  /// Returns true if `value` is live-in to `block`. Convenience accessor so
  /// consumers need not build their own liveness analysis.
  bool isLiveIn(Block *block, Value value) const;

  /// Returns the per-hardware-thread GRF budget in bytes for the given GRF
  /// mode (one hardware thread executes a whole subgroup/warp of lanes sharing
  /// one register file).
  ///
  /// Explicit sizes ("128", "256", "512") map to the exact per-hardware-thread
  /// budget. For "default" and "auto" the compiler chooses the GRF size at JIT
  /// time, so this function conservatively returns the smallest (128-register)
  /// budget to avoid exceeding the hardware limit on configurations that
  /// ultimately compile with fewer registers.
  static unsigned getGRFBytesPerHardwareThread(StringRef grfMode);

  /// Returns `getGRFBytesPerHardwareThread(grfMode) / threads-per-warp`: the
  /// per-lane figure to compare against this analysis's (per-lane) output.
  /// When the module's ttg.threads-per-warp attribute is absent,
  /// getThreadsPerWarp returns 32 as a default, so the budget is divided by 32
  /// (128 bytes/lane at default GRF mode) rather than the DPAS-typical 16
  /// (256 bytes/lane).
  static unsigned getPerLaneGRFBudgetInBytes(StringRef grfMode, ModuleOp mod);

  /// Returns the per-thread size in bytes for the given type.
  ///
  /// For RankedTensorType: computes getTotalElemsPerThread * (bitwidth/8).
  /// For scalar int/float types: returns bitwidth/8.
  /// For other types: returns 0.
  static unsigned getPerThreadSizeInBytes(Type type);

  /// Print the peak pressure per block to the given stream.
  void print(raw_ostream &os) const;

private:
  /// Returns true if the defining op of \p value is rematerializable (cheap to
  /// regenerate on demand, such as constants or simple range ops).
  bool isRematerializable(Value value) const;

  LivenessAnalysis liveness;
  RegisterPressureOptions options;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_REGISTER_PRESSURE_H
