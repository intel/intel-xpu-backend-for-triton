// This file declares the RegisterPressureAnalysis class, which computes
// per-thread register pressure in bytes using liveness information and
// per-thread element distribution from distributed encodings.

#ifndef TRITON_INTEL_ANALYSIS_REGISTER_PRESSURE_H
#define TRITON_INTEL_ANALYSIS_REGISTER_PRESSURE_H

#include "intel/include/Analysis/Liveness.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::gpu::intel {

/// Options controlling register pressure analysis fidelity.
struct RegisterPressureOptions {
  /// If true, exclude rematerializable values (cheap constants and similar
  /// ops) from pressure computation. These can be regenerated cheaply rather
  /// than held in registers.
  bool excludeRematerializable = true;
};

/// Analysis that computes per-thread GRF register pressure in bytes.
///
/// This analysis builds on LivenessAnalysis and weights each live value by its
/// per-thread size in bytes. For distributed tensors, the size is computed
/// using the encoding's element distribution. For scalars, the size is the
/// element size in bytes.
///
/// The canonical unit is **per-thread bytes**, matching how GRF budget is
/// expressed (e.g., 4096 bytes for 128-GRF mode).
///
/// Pointers are modeled as 64-bit addresses: TTGIR-level `tt.ptr` values live
/// in the global address space, and shared local memory is modeled as
/// `ttg.memdesc` rather than `tt.ptr`, so no address-space-specific pointer
/// sizing is needed.
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

  /// Returns the per-thread GRF budget in bytes for the given GRF mode.
  ///
  /// Explicit sizes ("128", "256", "512") map to the exact per-thread budget.
  /// For "default" and "auto" the compiler chooses the GRF size at JIT time,
  /// so this function conservatively returns the smallest (128-register)
  /// budget to avoid exceeding the hardware limit on configurations that
  /// ultimately compile with fewer registers.
  static unsigned getGRFBytesPerThread(StringRef grfMode);

  /// Returns the per-thread size in bytes for the given type.
  ///
  /// For RankedTensorType: computes getTotalElemsPerThread times the element
  /// size in bytes.
  /// For scalar types: returns the element size in bytes.
  /// For any other type (e.g. `tt.tensordesc`, `ttg.memdesc`), and for a tensor
  /// whose encoding is not a DistributedEncodingTrait (e.g. TTIR before layout
  /// assignment) or whose element type has no register footprint: returns 0.
  ///
  /// The element size in bytes is `ceil(bitwidth/8)` for int and float types,
  /// so that sub-byte types (i1, fp8, fp4) are not counted as free, and 8 for
  /// pointer types.
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
