---
description: 'Intel GPU hardware architecture: Xe generations, GRF modes, DPAS encoding, target capabilities, register pressure, compilation pipeline'
applyTo: '**/TritonIntelGPUTransforms/**/*.cpp, **/TritonIntelGPUTransforms/**/*.h, **/Dialect/TritonIntelGPU/**/*.td, **/Dialect/TritonIntelGPU/**/*.cpp, **/Dialect/TritonIntelGPU/**/*.h, **/backend/compiler.py, **/backend/driver.c, **/backend/driver.py, **/Analysis/**/*.h, **/Analysis/**/*.cpp, **/Analysis/**/*.tpp'
---

# Intel GPU Hardware Architecture

## Xe Architecture Overview

> **IMPORTANT**: Before writing any hardware-dependent code, you MUST read the full hardware specs in `.claude/reference/hardware-reference.md` using the Read tool.

**Do not guess** architecture specs (generation details, XVE/XMX specs, terminology mapping) — read them from `.claude/reference/hardware-reference.md`.

## GRF (General Register File)

Each hardware thread has a private register file. **Do not guess** GRF register specs or mode details (128/256/512/auto flags) — read them from `.claude/reference/hardware-reference.md`.

### Auto-GRF Mode Selection (`grf_mode='default'`)
1. Compile with default (small) GRF
2. Extract spill size from ZEBIN `.ze_info` section (AOT) or query Level Zero
   `spillMemSize` (JIT) — both are **bytes per hardware thread**
3. Normalize to **dword-equivalents per lane** (`bytes / (4 × sub-group size)`),
   the unit CUDA/HIP report `n_spills` in and that external consumers threshold on
4. If that exceeds `16` → recompile with the largest GRF mode this target
   auto-escalates to (256-GRF on every non-`cri` target; 512-GRF on `cri` —
   see `get_max_grf_mode()` in `compiler.py`, issue #8074)

The threshold is `16` dword-equivalents/lane, aligned between
`MAX_REG_SPILL_SLOTS_PER_LANE` in `compiler.py` and `kMaxSpillSlotsPerLane` in
`driver.c`. It is PyTorch inductor's default `spill_threshold` for non-HIP, so a
spill at or below it cannot change inductor's autotuning verdict and a rebuild
would only cost compile time. Compare in the normalized unit, not in bytes —
inductor tests the truncated per-lane count, so a byte threshold over-triggers.

### Constraints
- **256-GRF and 512-GRF require `num_warps ≤ 32`** (256-GRF halves, and 512-GRF quarters, the hardware threads available per subslice — see the occupancy column in `.claude/reference/hardware-reference.md`'s GRF mode table — capping the launchable work-group size below what `num_warps > 32` needs)
- `grf_mode` options: `'default'`, `'128'`, `'256'`, `'512'`, `'auto'`

## Subgroups and SIMD Execution

### Subgroup-to-Warp Mapping
In the Intel backend: **subgroup = warp**.
- Default `warp_size = 32` (configurable)
- DPAS operations use `threadsPerWarp = 16` (fixed)
- `warp_size` must be in device's `sub_group_sizes` list
- Workgroup size = `num_warps × warp_size`
- `num_warps` must be ≤ device's `max_num_sub_groups`

**Do not guess** per-architecture subgroup sizes — read them from `.claude/reference/hardware-reference.md`.

### DPAS Execution Size
The DPAS N dimension equals `min(sub_group_sizes)`:
- **PVC, BMG**: execution_size = 16
- **DG2**: execution_size = 8

## Target Architectures

**Do not guess** target architecture details (arch strings, DPAS/2D block IO support) — read the full table from `.claude/reference/hardware-reference.md`.

## Device Capabilities

**Do not guess** device capability-to-module-attribute mappings — read the full table from `.claude/reference/hardware-reference.md`.

### LTS Driver Detection
LTS (Long Term Support) driver version threshold: `(1, 6, 35096, 9)`.
When `is_lts=true`, certain lowering paths use GenISA intrinsics instead of SPIR-V builtins.

## DPAS Hardware Constants and Engine Types

**Do not guess** DPAS hardware constants or engine type combinations (Xe2/Xe3P) — read them from `.claude/reference/hardware-reference.md`. DpasEncodingAttr parameters and layout details are in `intel-layout-encodings.md`.

## Register Pressure Management

### ReduceVariableLiveness Pass
Activated when `opt.reduce_variable_liveness = true`. Takes a `grf-mode` option
(`'default'`, `'auto'`, `'128'`, `'256'`, `'512'`) that must match the `grf_mode`
the kernel is compiled with.

Gate: a 2D operand load that is live-in to a loop is sunk into it (leaving a
prefetch behind) when the loop body's **peak** register pressure, from
`RegisterPressureAnalysis::peakPressure(loop)`, is at or above the **per-lane**
GRF budget, `getPerLaneGRFBudgetInBytes(grfMode, mod, UnknownGRFSizeAssumption::Largest)`.
At `threads-per-warp = 16` that is 256 B/lane for `'128'`, 512 for `'256'`, and
1024 for `'512'`. For `'default'`/`'auto'` the true GRF size isn't known at
this point in the pipeline, and this gate treats the budget as a threshold to
sink rather than a ceiling, so the safe assumption under uncertainty is the
*largest* mode this target's escalation reaches. That only means something
target-specific for `'default'`, whose own AOT/JIT retry is the one path
that actually realizes the `ttig.max_grf_mode` module attribute (512 B/lane
on every non-`cri` target, 1024 B/lane on `cri`, or 1024 B/lane if the
attribute is absent entirely, e.g. hand-written TTGIR that never went
through `TritonAnnotateModule`). `'auto'`'s escalation happens inside IGC
with no backend path that reads back or constrains it, so `ttig.max_grf_mode`
has no established relationship to what IGC picks under `'auto'` and is not
applied there -- `'auto'` always resolves to the unconditional 1024 B/lane
bound, same as an absent attribute, regardless of target. See `RegisterPressureAnalysis`'s
`UnknownGRFSizeAssumption` for the
full rationale, including why `HoistLayoutConversions` correctly assumes the
opposite (`Smallest`) for the same unknown modes. There is no fixed
tensor-size floor; sizes only matter through their contribution to the
measured pressure.

Peak, not live-in, pressure is the gate: `liveInPressure` derives from
`LivenessBlockInfo::in()`, which excludes block arguments and so never counts the
loop-carried DPAS accumulator.

The decision is per loop: once a loop is over budget, every eligible load in it
is sunk. Spill cost vs. redundant-reload cost is not modelled, and the loop trip
count is not consulted.

### Design Rationale
Intel GPUs load dot operands directly into registers from memory (no intermediate SLM staging for A/B matrices). The hardware I/O buffer and cache handle redundant accesses. This differs from NVIDIA GPUs which typically stage through shared memory.

### Optimization Passes for Register Pressure
- `ReduceVariableLiveness`: minimizes live register ranges
- `ReduceDataDuplication`: eliminates redundant register copies
- `OptimizeReductionLocality`: improves cache locality for reductions

## Compilation Pipeline

### Stage 1: TTIR (Triton IR)
```
inliner → convert_block_pointer_to_tdesc → rewrite_tensor_descriptor_to_pointer →
cse → triton_licm → remove_boundary_checks → remove_masks →
stride_versioning → fuse_reshape → canonicalizer → combine →
simplify_signed_arithmetic → reorder_broadcast → cse → symbol_dce →
loop_unroll
```

### Stage 2: TTGIR (GPU IR)
```
--- separate pass manager ---
annotate_module
--- main pass manager ---
convert_to_ttgpuir →
coalesce → remove_layout_conversions →
accelerate_matmul → materialize_block_pointer → remove_layout_conversions →
optimize_dot_operands(intel) → pipeline(num_stages, use_barrier) →
[reduce_variable_liveness] → fuse_nested_loops →
canonicializer → triton_licm → canonicalizer →
combine_tensor_select_and_if → optimize_thread_locality →
optimize_dot_operands(upstream) → cse → prefetch → optimize_dot_operands(upstream) →
remove_layout_conversions → reduce_data_duplication →
reorder_instructions → cse → symbol_dce → sccp → canonicalizer →
[optimize_reduction_locality] → arith_emulate_unsupported_floats(bf16→f32)
```

### Stage 3: LLIR (LLVM IR)
```
scf_to_cf → gluon_inliner → index_to_llvmir → allocate_shared_memory →
allocate_global_scratch_memory → [instrumentation] →
to_llvmir → gen_to_llvm → canonicalizer → rewrite_stack_ptr →
cse → arith_to_llvmir → canonicalizer → cse → symbol_dce → [di_scope]
→ MLIR-to-LLVM → set_fast_math → [link_extern_libs] →
optimize_module(O3) → post_process_llir
```

### Stage 4: SPIR-V
LLVM IR → SPIR-V translation via `translate_to_spirv()`. GRF mode flags added to build_flags.

### Stage 5: ZEBIN
SPIR-V → native binary via `ocloc compile`. Auto-GRF spill detection may trigger recompilation.

## Memory Hierarchy

**Do not guess** cache sizes per architecture — read them from `.claude/reference/hardware-reference.md`.
