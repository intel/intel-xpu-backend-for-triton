// This file declares the RegisterPressureAnalysis class, which computes
// per-thread register pressure in bytes using liveness information and
// per-thread element distribution from distributed encodings.

#ifndef TRITON_INTEL_ANALYSIS_REGISTER_PRESSURE_H
#define TRITON_INTEL_ANALYSIS_REGISTER_PRESSURE_H

#include "intel/include/Analysis/Liveness.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
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
/// convention), NOT the per-hardware-thread unit `getGRFBytesPerHardwareThread`
/// returns. Use `getPerLaneGRFBudgetInBytes` to compare against a GRF budget.
///
/// The analysis is region-aware in two distinct ways. `peakPressure(Block *)`
/// accounts for values live inside any region nested under the block (an
/// `scf.if` body, an inner loop, ...); `pressureAt` and its siblings
/// (`pressureAt(Operation *, Value)`, `pressureBefore`) account for values
/// that merely live *through* the region-holding ops enclosing the queried
/// point, without being touched inside them -- all three share one
/// underlying live-through computation, so they cannot silently disagree
/// about what is live at a nested point. None of the three descend into the
/// queried op's own regions -- use `peakPressure` for that. (The live-in
/// accessors below -- `liveInPressure`, `isLiveIn`, `liveInContribution` --
/// are block-local by design and do not descend into or climb past nested
/// regions either; see their own doc comments.) See their individual doc
/// comments for the exact rules. `peakPressure` and `pressureAt` are not free
/// to call: see the `LiveValuesCache`/`AncestorLiveThroughCache` comment below
/// for what they cost and why.
class RegisterPressureAnalysis {
private:
  /// Memoized results of `LivenessBlockInfo::currentlyLiveValues()`, keyed by
  /// the program point queried. MLIR documents that query as expensive and does
  /// not cache it, while reporting pressure over a nest of blocks queries the
  /// same few points over and over: each op enclosing a block is queried once
  /// per operation that block's walk visits, and each nested op is walked again
  /// on behalf of every block enclosing it.
  ///
  /// A cache is owned by (and lives no longer than) a single top-level query,
  /// never by the analysis: consumers run inside passes, which must not carry
  /// mutable state across invocations. Every public entry point creates one and
  /// threads it through the whole traversal it performs, so that within one
  /// such query the live set of any given program point is computed exactly
  /// once.
  ///
  /// NOTE ON COST: even with this cache, `peakPressure`/`pressureAt` are real,
  /// non-trivial per-candidate costs, not free predicates -- each query still
  /// builds and returns a value set, and `peakPressure(Block *)` invokes one
  /// such query per op in the (possibly deeply nested) block it walks. A prior
  /// hardware A/B of this analysis measured roughly 3x longer compile times
  /// across six real kernel regimes, severe enough that the test DUT went into
  /// swap during the run. Any future cost-model work that calls these
  /// primitives per candidate (e.g. to rank several possible transforms) must
  /// account for that cost explicitly rather than assume it is negligible.
  using LiveValuesCache = DenseMap<Operation *, Liveness::ValueSetT>;

  /// Memoized live-through contribution of an ancestor (a region-holding op
  /// enclosing the point of interest) and everything enclosing *that*
  /// ancestor, up to (but not including) the analysis root. Each entry is the
  /// union of the ancestor's own filtered live values (its own results are
  /// dropped -- see `getLiveThroughAncestorSet`'s implementation for why) with
  /// the entry for its own parent.
  ///
  /// This exists so that querying pressure at N sibling (or cousin) ops that
  /// share an ancestor chain costs that chain's union exactly once in total,
  /// not once per query: without it, `pressureAt` would re-walk and re-union
  /// the same ancestor chain from scratch for every op `peakPressure(Block *)`
  /// visits, an O(ops visited x nesting depth) cost that grows with how deep a
  /// kernel nests `scf.if`/`scf.for`. With it, computing a given ancestor's
  /// entry costs only that ancestor's own live set once (its parent's entry is
  /// reused, not recomputed), and each descendant query becomes a single union
  /// of its own live set with the already-computed ancestor entry.
  using AncestorLiveThroughCache = DenseMap<Operation *, Liveness::ValueSetT>;

public:
  /// Construct the analysis for the given root operation.
  explicit RegisterPressureAnalysis(Operation *op,
                                    RegisterPressureOptions opts = {});

  /// Returns the per-thread register pressure in bytes at the given operation,
  /// accounting for all live values at that program point.
  ///
  /// For an op nested inside one or more regions, this includes values that are
  /// live *through* every enclosing region-holding op (defined before it and
  /// used after it): they occupy a register for the whole duration of that op
  /// even though they are neither defined nor used in the nested block. Each
  /// such value is counted once, no matter how many nesting levels it spans.
  unsigned pressureAt(Operation *op) const;

  /// Returns the peak per-thread register pressure in bytes within the given
  /// block, including operations nested in the regions of ops in the block (an
  /// scf.if body, an inner loop body, ...). A block therefore never reports
  /// less pressure than any block nested inside it.
  unsigned peakPressure(Block *block) const;

  /// Opaque handle sharing this analysis's liveness caches across a run of
  /// `pressureAt`/`pressureBefore` queries against one IR generation -- the
  /// per-op equivalent of the cache pair every block-/function-level entry
  /// point above already threads through internally. Construct one, pass it
  /// to repeated queries over a stable range of operations (e.g. a candidate
  /// load's live range), and let it go out of scope once the IR they describe
  /// is about to change (a sink, a hoist, ...): nothing here is safe to reuse
  /// across a mutation, matching every other cache in this class.
  class QueryCache {
  public:
    QueryCache() = default;

  private:
    friend class RegisterPressureAnalysis;
    LiveValuesCache liveCache;
    AncestorLiveThroughCache ancestorCache;
  };

  /// Same as `pressureAt(Operation *)`, but reusing and extending \p cache
  /// instead of building a fresh pair of caches for this one call. Prefer
  /// this overload for any loop that queries more than one operation, e.g.
  /// walking a candidate's live range one operation at a time -- the
  /// single-op overload pays the whole ancestor-chain cost from scratch on
  /// every call, which is exactly the cost `QueryCache` exists to amortize.
  unsigned pressureAt(Operation *op, QueryCache &cache) const;

  /// Returns the peak per-thread register pressure in bytes within the given
  /// loop, considering all blocks in the loop body region, nested regions
  /// included.
  unsigned peakPressure(LoopLikeOpInterface loop) const;

  /// Returns the peak per-thread register pressure in bytes over *every* block
  /// nested in \p func: registers are allocated per kernel, so a per-kernel
  /// allocation must cover the maximum over all blocks, not any one block's.
  ///
  /// Shares its block walk with `print()` so the two cannot drift apart.
  unsigned peakPressure(FunctionOpInterface func) const;

  /// Returns the per-thread register pressure in bytes immediately *above*
  /// \p op: the values live at \p op minus \p op's own results. This is the
  /// pressure an operation inserted at that program point would face -- name
  /// the point by the operation below it.
  ///
  /// Not `pressureAt(op)`: that counts a value at its defining op, so it
  /// charges results that do not exist yet above \p op. Subtracting \p op's
  /// results is exact, not merely conservative -- nothing else live at \p op is
  /// absent immediately above it.
  ///
  /// Not `liveInPressure(op->getBlock())` either: block arguments are
  /// definitions of their block rather than live-ins, so a live block-argument
  /// value contributes nothing there.
  unsigned pressureBefore(Operation *op) const;

  /// The pressure at one operation, together with whether one nominated value
  /// is live there.
  struct PressureAtPoint {
    unsigned pressure = 0;
    /// True if the nominated value is live at the operation in the sense
    /// `pressureAt` counts it: a value consumed by, or defined by, the
    /// operation counts as live.
    bool valueLive = false;
  };

  /// Returns the pressure at \p op together with whether \p value is live
  /// there, from **one** region-aware live-set build (the same one
  /// `pressureAt(Operation *)` computes -- this overload just also reports
  /// whether \p value is a member).
  ///
  /// That set is uncached and expensive to build (see `pressureAt`), and a
  /// caller walking a run of operations while adjusting the pressure by a
  /// nominated value needs both answers at every step; asking separately
  /// doubles the dominant cost. A null \p value reports `false`.
  PressureAtPoint pressureAt(Operation *op, Value value) const;

  /// Returns the bytes \p value contributes to the figures this analysis
  /// reports: its per-thread size after filtering, or 0 when it is
  /// rematerializable (and `excludeRematerializable` is set) or has no uses.
  ///
  /// The one place the filtering rule lives, so callers adjusting a reported
  /// figure need not re-derive it from the type. Unused values must be
  /// filtered: `currentlyLiveValues` charges a userless block argument at its
  /// block's first operation, which a reordering transform would see as a
  /// phantom change.
  unsigned pressureContribution(Value value) const;

  /// Returns the per-thread register pressure in bytes from the values live-in
  /// to the given block (i.e. defined outside and used inside). Honors
  /// excludeRematerializable. Returns 0 for a block with no liveness info
  /// (e.g. unreachable).
  unsigned liveInPressure(Block *block) const;

  /// Returns true if `value` is live-in to `block`. Convenience accessor so
  /// consumers need not build their own liveness analysis.
  bool isLiveIn(Block *block, Value value) const;

  /// Returns the number of bytes `value` currently contributes to
  /// `liveInPressure(block)`, and 0 if it contributes nothing (not live-in, or
  /// filtered out by excludeRematerializable).
  ///
  /// This exists so a caller reasoning about how a transform will change a
  /// block's live-in pressure can subtract exactly the term this analysis
  /// counted, rather than re-deriving it from the value's type and duplicating
  /// the filtering rules. It is by construction consistent with
  /// `liveInPressure`: if a transform's only effect on `block`'s live-in set is
  /// to remove `v` and add `w`, then the new live-in pressure is
  /// `liveInPressure(block) - liveInContribution(block, v) +
  /// getPerThreadSizeInBytes(w.getType())`.
  unsigned liveInContribution(Block *block, Value value) const;

  /// For "default"/"auto" GRF mode, the actual GRF size the compiler will
  /// pick is not known at the point these budget helpers run. Which bound is
  /// safe to assume depends on how the *caller* uses the budget, so callers
  /// must choose explicitly rather than relying on a shared default:
  ///
  ///   - A caller that treats the budget as a ceiling on how much it may add
  ///     (e.g. HoistLayoutConversions, deciding how much it may safely hoist)
  ///     wants `Smallest`: underestimating the true GRF size keeps the
  ///     caller safely inside the real (possibly larger) limit.
  ///   - A caller that treats the budget as a threshold whose crossing
  ///     triggers a transform (e.g. ReduceVariableLiveness, deciding whether
  ///     pressure is high enough to justify sinking a load) wants `Largest`:
  ///     underestimating the true GRF size would trigger the transform more
  ///     often than the real (possibly larger) budget actually warrants.
  ///
  /// There is deliberately no default value for this parameter: a new
  /// caller must consciously pick one rather than silently inheriting
  /// whichever direction happened to suit an earlier caller.
  enum class UnknownGRFSizeAssumption {
    /// Assume the smallest GRF size the device supports (128-register mode).
    Smallest,
    /// Assume the largest GRF size the device supports.
    ///
    /// FIXME(#8074): this is currently 512-register mode unconditionally,
    /// but the backend only ever selects 512-register mode on "cri"; every
    /// other target (including BMG and PVC) caps at 256-register mode (see
    /// third_party/intel/backend/compiler.py's GRF retry logic). This should
    /// be the true per-target largest size, not a hardcoded constant.
    Largest,
  };

  /// Returns the per-hardware-thread GRF budget in bytes for the given GRF
  /// mode (one hardware thread executes a whole subgroup/warp of lanes sharing
  /// one register file).
  ///
  /// Explicit sizes ("128", "256", "512") map to the exact per-hardware-thread
  /// budget, ignoring `unknownAssumption`. For "default" and "auto", returns
  /// the smallest or largest GRF size per `unknownAssumption` (see its
  /// documentation for which one a given caller needs).
  static unsigned
  getGRFBytesPerHardwareThread(StringRef grfMode,
                               UnknownGRFSizeAssumption unknownAssumption);

  /// Returns `getGRFBytesPerHardwareThread(grfMode, unknownAssumption) /
  /// threads-per-warp`: the per-lane figure to compare against this
  /// analysis's (per-lane) output. When the module's ttg.threads-per-warp
  /// attribute is absent, getThreadsPerWarp returns 32 as a default, so the
  /// budget is divided by 32 (128 bytes/lane at default GRF mode, smallest
  /// assumption) rather than the DPAS-typical 16 (256 bytes/lane).
  static unsigned
  getPerLaneGRFBudgetInBytes(StringRef grfMode, ModuleOp mod,
                             UnknownGRFSizeAssumption unknownAssumption);

  /// Returns the per-thread size in bytes for the given type.
  ///
  /// For RankedTensorType: computes getTotalElemsPerThread * (bitwidth/8).
  /// For scalar int/float types: returns bitwidth/8.
  /// For other types: returns 0.
  static unsigned getPerThreadSizeInBytes(Type type);

  /// Print the peak pressure over all nested blocks, then per block, to the
  /// given stream. The first line is the figure `peakPressure` reports for the
  /// analysis root; it is only a whole-*function* peak when the root is one
  /// function (which is how the `-test-register-pressure` pass builds it).
  void print(raw_ostream &os) const;

private:
  /// Returns the raw (unfiltered) `currentlyLiveValues()` result at \p point,
  /// memoizing it in \p cache.
  const Liveness::ValueSetT &getRawLiveValuesAt(Operation *point,
                                                LiveValuesCache &cache) const;

  /// Returns (and memoizes in \p ancestorCache) the union of live-through
  /// values contributed by \p ancestor and every op enclosing it up to (but
  /// not including) \p rootOp. See `AncestorLiveThroughCache`'s doc for why
  /// this is memoized per ancestor rather than recomputed per query.
  const Liveness::ValueSetT &
  getLiveThroughAncestorSet(Operation *ancestor, LiveValuesCache &liveCache,
                            AncestorLiveThroughCache &ancestorCache,
                            Operation *rootOp) const;

  /// Returns the full region-aware live set at \p op: its own block-local raw
  /// live values unioned with whatever lives through every enclosing
  /// region-holding op. The one computation `pressureAt`, `pressureBefore`,
  /// and the two-argument `pressureAt` overload all share, so they cannot
  /// disagree about what is live at a nested point.
  Liveness::ValueSetT
  computeLiveValues(Operation *op, LiveValuesCache &liveCache,
                    AncestorLiveThroughCache &ancestorCache) const;

  /// Implements `pressureAt`, reusing and extending \p liveCache and
  /// \p ancestorCache for the liveness queries it performs.
  unsigned pressureAt(Operation *op, LiveValuesCache &liveCache,
                      AncestorLiveThroughCache &ancestorCache) const;

  /// Implements `peakPressure(Block *)`, reusing and extending \p liveCache
  /// and \p ancestorCache for the liveness queries it performs.
  unsigned peakPressure(Block *block, LiveValuesCache &liveCache,
                        AncestorLiveThroughCache &ancestorCache) const;

  /// Returns true if the defining op of \p value is rematerializable (cheap to
  /// regenerate on demand, such as constants or simple range ops).
  bool isRematerializable(Value value) const;

  /// Returns the peak of `peakPressure(Block *)` over every block nested in
  /// \p root, in the walk order `print()` reports them in. Shared by
  /// `peakPressure(FunctionOpInterface)` and `print()`.
  unsigned peakPressureOverNestedBlocks(Operation *root) const;

  LivenessAnalysis liveness;
  RegisterPressureOptions options;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_ANALYSIS_REGISTER_PRESSURE_H
