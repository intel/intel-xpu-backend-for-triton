# AI Agent Context for Intel XPU Backend for Triton

> This file provides AI coding agents with comprehensive context about the Intel XPU Backend
> for the Triton compiler. It consolidates research that maps classical compiler concepts to
> this codebase.
>
> **For C++ coding style guidelines, see [.github/copilot-instructions.md](.github/copilot-instructions.md)**

---

## Quick Reference

| What | Where |
|------|-------|
| JIT entry point | `python/triton/runtime/jit.py` (`jit`, `JITFunction.run`, `JITFunction._do_compile`) |
| Compilation orchestration | `python/triton/compiler/compiler.py` (`compile`) |
| Intel backend | `third_party/intel/backend/compiler.py` (`XPUBackend`, `make_ttir`, `make_ttgir`, `make_llir`, `make_spv`, `make_zebin`) |
| Core dialects | `include/triton/Dialect/` (`Triton`, `TritonGPU`) |
| Intel dialects | `third_party/intel/include/Dialect/` (`TritonIntelGPU`, `TritonGEN`) |
| Python bindings | `python/src/main.cc`, `third_party/intel/triton_xpu.cc` |
| All dialect registration | `bin/RegisterTritonDialects.h` (`registry.insert<...>`) |
| Configuration knobs | `python/triton/knobs.py` (centralized env var management) |

---

## How To Use This File (RAG Style)

Use this retrieval order to minimize search time and reduce mistakes:

1. **Find intent** in the request (frontend, pass, lowering, runtime, tests).
2. **Jump to section 9** (Task-to-Code Navigation Map) for the first edit location.
3. **Use sections 10-11** for validation sequence and guardrails before finalizing.
4. **Only then** expand to broader repository search.

### Intent → First Places To Read

| Request intent keywords | Read first |
|---|---|
| `jit`, `constexpr`, `specialization`, `cache`, `launch` | `python/triton/runtime/jit.py`, `python/triton/compiler/compiler.py` |
| `ttir`, `combine`, `licm`, `loop_unroll`, `boundary_check` | `third_party/intel/backend/compiler.py` (`make_ttir`), `lib/Dialect/Triton/Transforms/` |
| `ttgir`, `coalesce`, `dpas`, `pipeline`, `layout` | `third_party/intel/backend/compiler.py` (`make_ttgir`), `third_party/intel/lib/TritonIntelGPUTransforms/` |
| `convert_layout`, `to llvm`, `intrinsic`, `spirv` | `lib/Conversion/TritonGPUToLLVM/`, `third_party/intel/lib/TritonIntelGPUToLLVM/`, `third_party/intel/lib/TritonGENToLLVM/` |
| `shared memory`, `barrier`, `alias`, `allocation` | `lib/Analysis/Allocation.cpp`, `lib/Analysis/Membar.cpp`, `lib/Analysis/Alias.cpp` |
| `backend discovery`, `plugin backend` | `python/triton/backends/__init__.py`, `python/triton/backends/compiler.py` |
| `env var`, `config`, `knobs`, `debug flag` | `python/triton/knobs.py` |
| `gluon`, `experimental language` | `third_party/intel/backend/compiler.py` (`gluon_to_ttgir`), `python/triton/compiler/` |
| `tensor descriptor`, `tdesc`, `block pointer` | `third_party/intel/lib/TritonIntelGPUTransforms/MaterializeBlockPointer.cpp` |
| `dpas`, `matmul`, `dot`, `systolic` | `third_party/intel/lib/TritonIntelGPUTransforms/AccelerateMatmul.cpp`, `third_party/intel/lib/TritonIntelGPUToLLVM/DotOpToLLVM/DPAS.cpp` |

### Verified Constraints (Do Not Assume Otherwise)

- Triton path is **JIT-first**, not AOT-first (`jit.py` + `compiler.py`).
- Stage registration is language-dependent in Intel backend (`add_stages` handles `TRITON` and `GLUON`).
- Register allocation is downstream (LLVM/IGC); this repo mainly controls IR + lowering quality.
- `grf_mode=256` is invalid with high warp counts (validated in Intel backend options).
- JIT cache safety includes global-value change checks; mutated globals can invalidate assumptions.
- Environment variables are managed centrally via `python/triton/knobs.py`, not loose `os.environ` reads.
- The Intel backend uses a **two-level lowering**: TritonGPU ops → TritonGEN ops → LLVM intrinsics.

---

## 1. Project Overview

This repository implements the **Intel XPU backend** for the Triton compiler. Triton is a JIT
compiler that takes Python kernel functions decorated with `@triton.jit` and compiles them to
native GPU binaries. The compilation pipeline is built on **MLIR** (Multi-Level IR) and proceeds
through progressively lower intermediate representations.

### Compilation Pipeline

```
Python @triton.jit kernel
    │
    ▼ python/triton/compiler/code_generator.py (CodeGenerator: ast.NodeVisitor)
┌─────────────────────────────────────────────────────────────────────────────┐
│  TTIR (Triton IR, tt dialect)                                               │
│  • Hardware-agnostic tensor operations: tt.load, tt.store, tt.dot, tt.reduce│
│  • Tensors have shapes but NO layout encoding                               │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │ make_ttir() — 16 passes including:
          │   inliner, CSE, LICM, combine, loop_unroll
          │   Intel: boundary_check_removal, stride_versioning, fuse_reshape,
          │          convert_block_pointer_to_tdesc, remove_masks
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TTGIR (TritonGPU IR, ttg + ttig dialects)                                  │
│  • Tensors carry layout encoding attributes (BlockedEncoding, DpasEncoding) │
│  • Shared memory ops: ttg.local_alloc, ttg.local_load                       │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │ make_ttgir() — ~30 passes across two phases:
          │   annotate_module, coalesce, accelerate_matmul (DPAS), pipeline,
          │   materialize_block_pointer, optimize_dot_operands (×3),
          │   optimize_thread_locality, reduce_data_duplication
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LLVM IR (MLIR dialect → native LLVM)                                       │
│  • Two-level lowering: TritonGPU → TritonGEN → LLVM intrinsics              │
│  • Scalar operations with GenISA intrinsics                                 │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │ make_llir() → make_spv() (SPIR-V) → make_zebin() (ocloc/IGC)
          ▼
        zebin (native Intel GPU binary)
```

### Key Files by Stage

| Stage | Python Orchestration | C++ Implementation |
|-------|---------------------|-------------------|
| Frontend | `python/triton/compiler/code_generator.py` | - |
| TTIR passes | `compiler.py` (`make_ttir`) | `lib/Dialect/Triton/Transforms/`, Intel: `third_party/intel/lib/TritonIntelGPUTransforms/` |
| TTGIR passes | `compiler.py` (`make_ttgir`) | `third_party/intel/lib/TritonIntelGPUTransforms/` |
| LLVM lowering | `compiler.py` (`make_llir`) | `lib/Conversion/TritonGPUToLLVM/`, `third_party/intel/lib/TritonIntelGPUToLLVM/` |
| SPIR-V | `compiler.py` (`make_spv`) | `third_party/intel/triton_xpu.cc` |
| Binary | `compiler.py` (`make_zebin`) | External: `ocloc` (Intel Graphics Compiler) |

### XPUBackend Class

`XPUBackend(BaseBackend)` at `third_party/intel/backend/compiler.py` is the main backend class:

| Method | Purpose |
|--------|---------|
| `add_stages()` | Registers pipeline stages per language (`TRITON` or `GLUON`) |
| `make_ttir()` | TTIR optimization pass pipeline (16 passes) |
| `make_ttgir()` | TTGIR GPU-specific optimization pipeline (~30 passes) |
| `gluon_to_ttgir()` | Gluon language → TTGIR path (skips TTIR) |
| `make_llir()` | Lowers TTGIR → LLVM IR via two-level lowering (15 passes) |
| `make_spv()` | Translates LLVM IR → SPIR-V |
| `make_zebin()` | Compiles SPIR-V → native GPU binary via `ocloc` |
| `load_dialects()` | Loads TritonIntelGPU + TritonGEN dialects |
| `annotate_module()` | Adds module-level annotations (sub-group size, HW caps) |
| `validate_options()` | Validates compilation options against HW constraints |
| `parse_options()` / `parse_target()` | Parse backend-specific options |
| `is_lts()` | Detects LTS (Long-Term Support) driver versions |
| `optimize_llvm_mod()` | Runs LLVM O3 optimization on the LLVM module |

---

## 2. Dialect Architecture

### Dialect Hierarchy

```
MLIR Built-in: arith · math · scf · cf · gpu · LLVM · ub
                         │
    ┌────────────────────┴────────────────────┐
    │           Core Triton Dialects          │
    ├─────────────────────────────────────────┤
    │  Triton (tt)                            │
    │  └─▶ TritonGPU (ttg)                    │
    │      ├─▶ TritonIntelGPU (ttig)          │
    │      │   └─▶ TritonGEN (gen intrinsics) │
    │      ├─▶ TritonNvidiaGPU (ttng)         │
    │      └─▶ TritonAMDGPU                   │
    └─────────────────────────────────────────┘
  Also: Gluon (experimental), Proton (profiling), TritonInstrument
```

### Intel-Specific Dialects

**TritonIntelGPU (ttig)** — `third_party/intel/include/Dialect/TritonIntelGPU/IR/`

Encoding attributes:
- `DpasEncodingAttr` — Layout for DPAS (systolic array) operations. Params: `repeatCount`, `systolicDepth`, `executionSize`, `opsPerChannel`, `warpsPerCTA`, `repCluster`, `threadsPerWarp`
- `WarpEncodingAttr` — Warp-level layout for sub-group operations
- `Subgroup2DBlockEncodingAttr` — 2D block I/O layout. Params: `warpsPerCTA`, `instrShape`, `numBlocks`, `order`, `kWidth`

Operations (`TritonIntelGPUOps.td`):
- `ttig.alloc` — Intel-specific allocation
- `ttig.prefetch` — Data prefetch
- `ttig.broadcast` — Intel broadcast
- `ttig.sub_group_transpose` — Sub-group transpose

**TritonGEN** — `third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOps.td`

Low-level Intel GPU intrinsics:
- `triton_gen.dpas` — Matrix DPAS multiply-accumulate
- `triton_gen.bdpas` — Block-scale DPAS for MXFP (FP4/FP8 with scaling)
- `triton_gen.2Dblockload` / `2Dblockstore` / `2Dblockprefetch` — 2D block I/O
- `triton_gen.sub_group_block_read` / `sub_group_block_write` — Sub-group block R/W
- `triton_gen.barrier` — Workgroup barrier
- `triton_gen.split_barrier_arrive` / `split_barrier_wait` — Split barriers for pipelining
- `triton_gen.predicated_load` / `predicated_store` — Predicated memory ops
- `triton_gen.f_to_tf32` — Float to TF32 rounding

### Layout Encodings (Critical Concept)

Layout encodings describe how tensor elements map to threads, warps, and registers:

```
#triton_gpu.blocked<{sizePerThread=[1,4], threadsPerWarp=[4,8], warpsPerCTA=[4,1], order=[1,0]}>
```

- `sizePerThread` — Elements per thread per dimension
- `threadsPerWarp` — Thread organization within a warp
- `warpsPerCTA` — Warp organization within a CTA
- `order` — Memory access order (fastest-varying dimension first)

**LinearLayout** (`include/triton/Tools/LinearLayout.h`) — Mathematical abstraction for index transformations using linear functions over GF(2). All encodings implement `toLinearLayout()` — the unifying interface. Long-term goal is to replace legacy encodings with LinearLayouts entirely.

---

## 3. Complete Pass Pipeline

### `make_ttir()` — TTIR Passes (16 passes)

All passes in exact pipeline order from `third_party/intel/backend/compiler.py`:

| # | Pass | Source | Purpose |
|---|------|--------|---------|
| 1 | `add_inliner` | MLIR built-in | Inline all non-kernel functions |
| 2 | `add_convert_block_pointer_to_tdesc` | Intel TTIR | Convert block pointers to tensor descriptors |
| 3 | `add_rewrite_tensor_descriptor_to_pointer` | Intel TTIR | Rewrite unsupported tensor descriptors back to pointers |
| 4 | `add_cse` | MLIR built-in | Common subexpression elimination |
| 5 | `add_triton_licm` | Triton TTIR | Loop-invariant code motion (with load masking) |
| 6 | `add_remove_boundary_checks` | Intel TTIR | Remove provably-safe bounds checks |
| 7 | `add_remove_masks` | Intel TTIR | Eliminate unnecessary mask operations |
| 8 | `add_stride_versioning` | Intel TTIR | Create specialized code paths for unit-stride access |
| 9 | `add_fuse_reshape` | Intel TTIR | Fuse reshape operations to reduce intermediates |
| 10 | `add_canonicalizer` | MLIR built-in | Constant folding, identity elimination, CFG simplification |
| 11 | `add_combine` | Triton TTIR | Algebraic simplifications: dot+add fusion, broadcast+mul+reduce→dot |
| 12 | `add_simplify_signed_arithmetic` | Intel TTIR | Intel-specific signed arithmetic simplifications |
| 13 | `add_reorder_broadcast` | Triton TTIR | `elementwise(splat(a), splat(b))` → `splat(elementwise(a,b))` |
| 14 | `add_cse` | MLIR built-in | CSE cleanup after combine/reorder |
| 15 | `add_symbol_dce` | MLIR built-in | Dead code elimination |
| 16 | `add_loop_unroll` | Triton TTIR | Attribute-driven unrolling (`tt.loop_unroll_factor`) |

### `make_ttgir()` — TTGIR Passes (~30 passes, two phases)

**Phase 1: Module annotation** — adds HW capabilities, sub-group size, LTS driver flag:

| # | Pass | Purpose |
|---|------|---------|
| 1 | `add_triton_annotate_module` | Annotate module with device properties, sub-group size, HW caps |

**Phase 2: GPU optimization** (main pipeline):

| # | Pass | Source | Purpose |
|---|------|--------|---------|
| 1 | `add_convert_tdesc_to_block_pointer` | Intel TTIR | Convert tensor descriptors to block pointers for TTGIR |
| 2 | `add_convert_to_ttgpuir` | Triton | TTIR → TTGIR: assign initial `BlockedEncodingAttr` layouts |
| 3 | `add_coalesce` | Intel | Memory access coalescing via AxisInfo analysis |
| 4 | `add_remove_layout_conversions` | Intel | Eliminate redundant layout conversions (#1) |
| 5 | `add_accelerate_matmul` | Intel | Convert `tt.dot` to `DpasEncodingAttr` (DPAS hardware) |
| 6 | `add_materialize_block_pointer` | Intel | Convert block pointers to 2D block I/O instructions |
| 7 | `add_remove_layout_conversions` | Intel | Eliminate redundant layout conversions (#2) |
| 8 | `add_optimize_dot_operands` | Intel | Propagate dot-operand layouts through producers |
| 9 | `add_pipeline` | Intel | Multi-stage software pipelining of loops |
| 10 | `add_reduce_variable_liveness` | Intel | Move loads closer to uses (conditional on option) |
| 11 | `add_fuse_nested_loops` | Triton | Fuse compatible nested loops |
| 12 | `add_canonicalizer` | MLIR | Cleanup |
| 13 | `add_triton_licm` | Triton | LICM at TTGIR level |
| 14 | `add_canonicalizer` | MLIR | Cleanup |
| 15 | `add_combine_tensor_select_and_if` | Triton | Combine tensor select with if operations |
| 16 | `add_optimize_thread_locality` | Triton | Improve data locality across threads |
| 17 | `add_optimize_dot_operands` | Triton | Dot operand layout propagation (#2, with `hoist=True`) |
| 18 | `add_cse` | MLIR | CSE cleanup |
| 19 | `add_prefetch` | Triton | Insert data prefetch operations |
| 20 | `add_optimize_dot_operands` | Triton | Dot operand layout propagation (#3, with `hoist=True`) |
| 21 | `add_remove_layout_conversions` | Intel | Eliminate redundant layout conversions (#3) |
| 22 | `add_reduce_data_duplication` | Intel | Reduce redundant data copies across threads |
| 23 | `add_reorder_instructions` | Triton | Reorder for compute/memory overlap |
| 24 | `add_cse` | MLIR | CSE cleanup |
| 25 | `add_symbol_dce` | MLIR | Dead code elimination |
| 26 | `add_sccp` | MLIR | Sparse conditional constant propagation |
| 27 | `add_canonicalizer` | MLIR | Final cleanup |
| 28 | `add_optimize_reduction_locality` | Intel | Improve reduction locality (conditional on knob) |
| 29 | `add_arith_emulate_unsupported_floats` | Intel | Emulate bf16 ops as f32 where unsupported |

### `make_llir()` — LLVM Lowering Passes (15 passes)

| # | Pass | Purpose |
|---|------|---------|
| 1 | `add_scf_to_cf` | Lower structured control flow to basic blocks + branches |
| 2 | `add_inliner` (gluon) | Inline remaining Gluon functions |
| 3 | `add_index_to_llvmir` | Lower index dialect to LLVM |
| 4 | `add_allocate_shared_memory` | Assign shared memory (SLM) addresses |
| 5 | `add_allocate_global_scratch_memory` | Assign global scratch memory |
| 6 | `add_to_llvmir` | **Level 1 lowering**: TritonGPU → TritonGEN ops |
| 7 | `add_gen_to_llvm` | **Level 2 lowering**: TritonGEN → LLVM intrinsics |
| 8 | `add_canonicalizer` | Cleanup |
| 9 | `add_rewrite_stack_ptr` | Rewrite stack pointer operations for Intel |
| 10 | `add_cse` | CSE |
| 11 | `add_arith_to_llvmir` | Lower arith dialect to LLVM |
| 12 | `add_canonicalizer` | Cleanup |
| 13 | `add_cse` | CSE |
| 14 | `add_symbol_dce` | Dead code elimination |
| 15 | `add_di_scope` | Debug info scopes (conditional) |

After MLIR passes: `llvm.to_module()` converts to native LLVM, then `optimize_module()` runs LLVM O3.

### `gluon_to_ttgir()` — Gluon Language Pipeline (6 passes)

Gluon is an experimental language that enters the pipeline at TTGIR level (skipping TTIR):

| # | Pass | Source | Purpose |
|---|------|--------|--------|
| 1 | `add_inliner` | Gluon | Inline Gluon functions |
| 2 | `add_resolve_auto_encodings` | Gluon | Resolve automatic layout encodings |
| 3 | `add_sccp` | MLIR | Sparse conditional constant propagation |
| 4 | `add_loop_aware_cse` | Triton | Loop-aware common subexpression elimination |
| 5 | `add_canonicalizer` | Gluon | Gluon-specific canonicalization |
| 6 | `add_combine_tensor_select_and_if` | Triton | Combine tensor select with if operations |

After `gluon_to_ttgir`, the pipeline continues with `make_llir` → `make_spv` → `make_zebin`.

### Two-Level "Instruction Selection"

The Intel backend uses a **two-level lowering** instead of a single monolithic ISel:

```
  TritonGPU ops  ──►  TritonGEN ops  ──►  LLVM intrinsics
     (tt.dot)       (gen.dpas)           (llvm.genx.GenISA.dpas)
     (tt.load)      (gen.2Dblockload)    (llvm.genx.GenISA.LSC2DBlockRead)
```

**Level 1** (`add_to_llvmir`): Maps Triton ops to TritonGEN ops — captures Intel HW semantics
(cache controls, DPAS parameters, 2D block descriptors) without committing to intrinsic encoding.

**Level 2** (`add_gen_to_llvm`): Maps TritonGEN ops to LLVM calls/intrinsics — the final
lowering to `llvm.genx.GenISA.*` calls and inline assembly.

This separation allows TritonGEN to serve as a **hardware abstraction layer**.

Note: There is also a `TritonGEN → SPIR-V` path (`TritonGENToSPIRV`) registered in `RegisterTritonDialects.h`, providing an alternative lowering route.

### Key Lowering Files

| File | Operations Lowered |
|---|---|
| `TritonIntelGPUToLLVM/LoadStoreOpToLLVM.cpp` | `tt.load`, `tt.store` → block loads/stores, scatter/gather, 2D block I/O |
| `TritonIntelGPUToLLVM/DotOpToLLVM/DPAS.cpp` | `tt.dot` → DPAS intrinsics (matrix multiply) |
| `TritonIntelGPUToLLVM/ReduceOpToLLVM.cpp` | `tt.reduce` → cross-lane reduction trees |
| `TritonIntelGPUToLLVM/ElementwiseOpToLLVM.cpp` | Elementwise math → scalar/vector ops |
| `TritonIntelGPUToLLVM/ConvertLayoutOpToLLVM.cpp` | Layout conversions → shuffles/shared memory |
| `TritonIntelGPUToLLVM/BF16Casts.cpp` | BF16 type conversions |
| `TritonIntelGPUToLLVM/Fp4ToFpOpToLLVM.cpp` | FP4 conversion lowering |
| `TritonIntelGPUToLLVM/SPMDOpToLLVM.cpp` | `tt.get_program_id` → workgroup ID intrinsics |
| `TritonIntelGPUToLLVM/TensorPtrOpsToLLVM.cpp` | `tt.make_tensor_ptr`, `tt.advance` → tensor pointer lowering |
| `TritonIntelGPUToLLVM/HistogramOpToLLVM.cpp` | Histogram operation lowering |
| `TritonGENToLLVM/TritonGENToLLVMPass.cpp` | All TritonGEN ops → `llvm.genx.GenISA.*` intrinsics |

---

## 4. Critical Analyses

### Core Analyses

| Analysis | File | Purpose |
|----------|------|---------|
| **AxisInfo** | `lib/Analysis/AxisInfo.cpp` | Tracks divisibility, contiguity, and constancy of tensor values per dimension. Forward dataflow analysis using `SparseForwardDataFlowAnalysis`. Drives coalescing and block pointer decisions |
| **Allocation** | `lib/Analysis/Allocation.cpp` | Shared memory allocation via liveness-based graph coloring (analogous to register allocation for SLM slots) |
| **Membar** | `lib/Analysis/Membar.cpp` | Determines where barriers/fences are needed for shared memory synchronization. Forward dataflow over read/write sets, tracking hazards per allocation slice |
| **Alias** | `lib/Analysis/Alias.cpp` | May-alias analysis for shared memory buffers. Tracks which `MemDesc` values may point to the same allocation |
| **BufferRegion** | `lib/Analysis/BufferRegion.cpp` | Buffer region analysis for memory access patterns |

### Intel Analysis Overrides

Intel provides substantial overrides and new analyses in `third_party/intel/lib/Analysis/`:

| Analysis | File | Purpose |
|----------|------|---------|
| **AxisInfo** (Intel) | `third_party/intel/lib/Analysis/AxisInfo.cpp` (1600+ lines) | Full Intel-specific override of core AxisInfo with Intel GPU-specific dataflow rules |
| **Range** | `third_party/intel/lib/Analysis/Range.cpp` (800+ lines) | **Intel-only**: Range analysis for value bounds tracking (no core counterpart) |
| **Liveness** | `third_party/intel/lib/Analysis/Liveness.cpp` | **Intel-only**: `LiveInterval` tracking, used by `reduce_variable_liveness` pass |
| **Membar** (Intel) | `third_party/intel/lib/Analysis/Membar.cpp` | Barrier elision filter for back-to-back sub-group transposes |
| **Allocation** (Intel) | `third_party/intel/lib/Analysis/Allocation.cpp` | Override: returns 0 for sub-group shuffles, custom sizing for sub-group transposes |
| **Utility** | `third_party/intel/lib/Analysis/Utility.cpp` | Intel utility functions for sub-group transpose decisions and layout helpers |

---

## 5. Memory Model

### Intel GPU Memory Hierarchy

| Level | Intel Term | Scope | Notes |
|-------|-----------|-------|-------|
| GRF | General Register File | Per-thread | 128 or 256 registers, 32B each |
| SLM | Shared Local Memory | Per-work-group | Explicit programmer control |
| L1/L3 | Caches | Per-tile | Hardware-managed |
| HBM | Device Memory | Per-device | Up to 128 GB |

### Memory Operations

**Block pointers** (`tt.make_tensor_ptr`) are the preferred abstraction:
```python
ptr = tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)
data = tl.load(ptr, boundary_check=[0, 1])
```

These enable the Intel backend to use efficient 2D block load/store instructions (LSC).
The `make_ttir` pass pipeline includes `add_convert_block_pointer_to_tdesc` and
`add_rewrite_tensor_descriptor_to_pointer` to handle tensor descriptor conversion.

### Shared Memory

- Allocated via `ttg.local_alloc` after `allocate_shared_memory` pass
- Barrier analysis (`lib/Analysis/Membar.cpp`) determines synchronization points
- Alias analysis (`lib/Analysis/Alias.cpp`) tracks which buffers may overlap
- Allocation uses liveness-based graph coloring for SLM offset assignment

---

## 6. Runtime & JIT

### Kernel Launch Flow

```python
@triton.jit
def kernel(a_ptr, b_ptr, n: tl.constexpr):
    ...

kernel[(grid,)](a, b, N)  # Triggers JIT compilation on first call
```

1. `JITFunction.__getitem__(grid)` returns a callable
2. `JITFunction.run()` checks cache, compiles on miss
3. Cache key includes: arg types, constexpr values, alignment info, target device
4. Compilation proceeds through all stages (TTIR → TTGIR → LLVM → SPIR-V → zebin)
5. Binary is cached and launched via Level Zero runtime

### Specialization

Triton generates specialized code variants based on:
- Argument types (dtype, pointer vs scalar)
- Constexpr values (baked in as constants)
- Pointer alignment (enables vectorized loads if 16-byte aligned)
- `divisibility_hint` parameter annotations

### Caching

- In-memory: `JITFunction.device_caches[device_key][signature_key]`
- On-disk: `~/.triton/cache/` (hash-based directory structure)

### Global Variable Mutation Detection

`JITFunction` tracks global variables via `used_global_vals`. On each invocation, current
globals are compared against the snapshot. Mismatches invalidate the cache and trigger
recompilation. By default, non-constexpr globals are forbidden; set
`TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1` to relax this.

---

## 7. Build & Configuration

### Build Commands

```bash
# Full build (first time or after LLVM changes)
pip install -e python --no-build-isolation -v

# Incremental C++ rebuild
pip install -e python --no-build-isolation -v -C cmake.define.TRITON_BUILD_PYTHON_MODULE=OFF

# Run tests
cd python && pytest test/ -v

# Run specific LIT tests
./build/bin/triton-opt --help
./build/bin/triton-opt test/Triton/combine.mlir --triton-combine
```

### Configuration System (Knobs)

Environment variables are managed centrally via `python/triton/knobs.py`. Access in code via `knobs.<category>.<name>`. Key categories:

| Category | Key Variables | Purpose |
|----------|--------------|---------|
| **Runtime** | `TRITON_DEBUG`, `TRITON_INTERPRET` | Debug output, interpreter mode |
| **Cache** | `TRITON_DUMP_DIR`, `TRITON_CACHE_DIR`, `TRITON_OVERRIDE_DIR` | IR dumping, cache location, kernel override |
| **Compilation** | `TRITON_KERNEL_OVERRIDE`, `TRITON_KERNEL_DUMP`, `TRITON_ALWAYS_COMPILE`, `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` | Override/dump kernels, force recompilation |
| **Intel** | `TRITON_INTEL_OPTIMIZE_REDUCTION_LOCALITY`, `TRITON_INTEL_DISABLE_IGC_OPT`, `TRITON_LIBDEVICE_PATH`, `TRITON_INTEL_DEVICE_ARCH` | Intel-specific tuning |
| **Build** | `CC` | Native compiler path (in knobs.py) |
| **Other** | `TRITON_BACKENDS_IN_TREE` (`backends/__init__.py`), `MAX_JOBS` (`setup.py`) | Not in knobs.py — direct `os.environ` reads |

### Adding a New Pass

1. Define pass in TableGen: `include/triton/Dialect/.../Transforms/Passes.td`
2. Implement in C++: `lib/Dialect/.../Transforms/NewPass.cpp` (or `third_party/intel/lib/TritonIntelGPUTransforms/`)
3. Register in `PassRegistration.h`
4. Expose to Python via `python/src/passes.cc` or `third_party/intel/triton_xpu.cc`
5. Add to pipeline in `third_party/intel/backend/compiler.py` (`make_ttir` or `make_ttgir`)
6. Add LIT tests in `test/`

### Adding a New Operation

1. Define in TableGen: `include/triton/Dialect/.../IR/Ops.td`
2. Run `cmake --build build --target TritonTableGen`
3. Implement verifier/canonicalizer in `lib/Dialect/.../IR/Ops.cpp`
4. Add lowering pattern in `lib/Conversion/` or `third_party/intel/lib/TritonIntelGPUToLLVM/`
5. Add tests

---

## 8. Key Deviations from Classical Compilers

This is **not** a textbook compiler. Key differences:

| Classical Compiler | Triton |
|-------------------|--------|
| Scanner + Parser | Python's `ast.parse()` — no custom grammar |
| SSA construction | MLIR provides SSA by construction (block args = φ-functions) |
| Single IR level | **Multi-level IR**: TTIR → TTGIR → LLVM, each at different abstraction |
| Instruction selection (ISel) | MLIR conversion patterns, not tree-pattern matching. Two-level lowering |
| Instruction scheduling | `pipeline` + `reorder_instructions` passes + downstream IGC |
| Register allocation | Delegated to IGC backend (auto-retry with 256 GRF on spill >1000) |
| Optimization focus | **Layout selection** (how tensors map to threads) — no textbook chapter |
| Flat memory model | Hierarchical: GRF → SLM → L1/L3 → HBM with explicit management |
| General-purpose code | Fixed-shape tensors, no recursion, no heap, no dynamic allocation |
| AOT compilation | **JIT** with specialization per call-site (arg types, constexpr, alignment) |
| Independent passes | Many passes run **multiple times** (CSE ×6, canonicalizer ×7, remove_layout_conversions ×3) |

### Why This Design?

1. **Embedded DSL** — Python syntax avoids language design/maintenance
2. **MLIR infrastructure** — Gets SSA, passes, patterns, verifiers for free
3. **Constrained domain** — Fixed-shape tensors, no recursion, no heap
4. **Downstream backends** — LLVM + IGC handle final codegen, scheduling, register allocation
5. **Layout is king** — The dominant optimization is not "make this faster" but "find the best tensor-to-thread mapping"

---

## 9. Task-to-Code Navigation Map

Use this map first before broad searching.

| If you need to change... | Start here | Then check |
|---|---|---|
| Python frontend semantics (`tl.*`, AST lowering) | `python/triton/compiler/code_generator.py` | `python/triton/language/`, `python/triton/runtime/jit.py` |
| Compilation stage ordering | `third_party/intel/backend/compiler.py` (`add_stages`) | `python/triton/compiler/compiler.py` (`compile`) |
| New TTIR optimization | `lib/Dialect/Triton/Transforms/` | `include/triton/Dialect/Triton/Transforms/Passes.td` |
| New Intel TTIR pass | `third_party/intel/lib/TritonIntelGPUTransforms/` | `third_party/intel/include/Dialect/TritonIntelGPU/Transforms/Passes.td` |
| New Intel TTGIR optimization | `third_party/intel/lib/TritonIntelGPUTransforms/` | `third_party/intel/include/Dialect/TritonIntelGPU/Transforms/Passes.td` |
| Triton op definition / verifier / canonicalizer | `include/triton/Dialect/Triton/IR/TritonOps.td` | `lib/Dialect/Triton/IR/` |
| TritonGPU op/layout changes | `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` / `TritonGPUAttrDefs.td` | `lib/Dialect/TritonGPU/IR/` |
| Intel encoding attrs (DPAS, 2D block) | `third_party/intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.td` | `third_party/intel/lib/TritonIntelGPUTransforms/` |
| Intel-specific ops (alloc, prefetch, transpose) | `third_party/intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUOps.td` | `third_party/intel/lib/TritonIntelGPUToLLVM/` |
| TritonGEN intrinsic definitions | `third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOps.td` | `third_party/intel/lib/TritonGENToLLVM/` |
| TTGIR → LLVM lowering (Level 1) | `third_party/intel/lib/TritonIntelGPUToLLVM/` | `lib/Conversion/TritonGPUToLLVM/` |
| TritonGEN → LLVM lowering (Level 2) | `third_party/intel/lib/TritonGENToLLVM/` | `third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOps.td` |
| Shared memory/barrier behavior | `lib/Analysis/Allocation.cpp`, `lib/Analysis/Membar.cpp`, `lib/Analysis/Alias.cpp` | `lib/Dialect/TritonGPU/Transforms/` |
| Python bindings / pass exposure | `python/src/main.cc`, `python/src/passes.cc`, `third_party/intel/triton_xpu.cc` | `python/triton/backends/` |
| Environment variable / knob | `python/triton/knobs.py` | `third_party/intel/backend/compiler.py` |
| Gluon language support | `third_party/intel/backend/compiler.py` (`gluon_to_ttgir`) | `python/triton/compiler/` |

---

## 10. Debug & Validation Playbooks

### A) Pass-level C++ change

1. Run a focused IR test first:
  - `./build/bin/triton-opt test/Triton/<test>.mlir --<pass-name>`
2. Run relevant lit suite:
  - `cd test && lit -v Triton/` (or `TritonGPU/`, `TritonIntelGPU/`, `TritonGEN/`)
3. Only then run broader Python tests when needed:
  - `cd python && pytest test/ -v`

### B) Python frontend/runtime change

1. Run targeted pytest file:
  - `cd python && pytest test/<area>/<file>.py -v`
2. If JIT behavior changed, enable IR dumps:
  - `TRITON_DUMP_DIR=/tmp/triton_ir TRITON_DEBUG=1 pytest ...`
3. Validate at least one kernel end-to-end on Intel target.

### C) Lowering/codegen change (LLVM/SPIR-V/zebin)

1. Inspect IR stage outputs (`ttir`, `ttgir`, `llir`, `spv`) from dump dir.
2. Verify expected intrinsic exists in LLVM IR (e.g., `llvm.genx.GenISA.dpas`, `llvm.genx.GenISA.LSC2DBlockRead`).
3. Re-run minimal reproducer kernel with same specialization inputs.

### D) Adding/modifying a knob

1. Define in `python/triton/knobs.py` under the appropriate `*_knobs` class.
2. Access via `knobs.<category>.<name>`.
3. Use in `compiler.py` with `if knobs.<category>.<name>:` pattern.

### Fast sanity checklist

- Pass pipeline still runs in the intended order.
- No unexpected layout conversion churn (`convert_layout` loops).
- Register-pressure-sensitive changes checked with realistic tile sizes.
- Tests cover both correctness and expected IR shape (FileCheck patterns).
- Two-level lowering preserved: no TritonGEN ops remain after `add_gen_to_llvm`.

---

## 11. Intel XPU Backend Guardrails for Agents

### Prefer these patterns

- Keep transformations in MLIR rewrite passes; avoid ad-hoc IR mutation.
- Reuse AxisInfo/LinearLayout utilities instead of duplicating layout math.
- Keep pass effects local and composable; rely on canonicalize/CSE cleanup.
- Add/adjust lit tests whenever pass behavior or IR shape changes.
- Use the knobs system for new configuration; don't add loose `os.environ` reads.
- Respect the two-level lowering: TritonGPU → TritonGEN → LLVM (don't skip levels).

### Avoid these mistakes

- Don't introduce generic LLVM-style assumptions before TTGIR layout is settled.
- Don't bypass encoding attrs when transforming dot/matmul paths.
- Don't add expensive global scans when analysis results already exist.
- Don't change stage ordering in backend pipeline without validating dependent passes.
- Don't hardcode line numbers in references — use symbol/function names instead.
- Don't assume a pass runs once — many passes (CSE, canonicalizer, remove_layout_conversions, optimize_dot_operands) run **multiple times** in the pipeline.

### Performance-sensitive hotspots

- `accelerate_matmul` / DPAS layout legality and `warpsPerTile` distribution.
- `materialize_block_pointer` and 2D block load/store preconditions.
- `coalesce` decisions driven by contiguity/divisibility from AxisInfo.
- `reduce_variable_liveness` impact on spill behavior downstream in IGC.
- `add_pipeline` stage count (`num_stages`) vs. register pressure tradeoff.
- `add_optimize_thread_locality` for cross-thread data sharing efficiency.

### Test directories

| Directory | Content |
|-----------|---------|
| `test/Triton/` | Core Triton dialect LIT tests |
| `test/TritonGPU/` | TritonGPU dialect LIT tests |
| `test/TritonIntelGPU/` | Intel GPU-specific LIT tests |
| `test/TritonGEN/` | TritonGEN dialect LIT tests |
| `test/Conversion/` | Conversion pass LIT tests |
| `test/Analysis/` | Analysis pass LIT tests |
| `test/Gluon/` | Gluon language LIT tests |
| `python/test/` | Python pytest tests (end-to-end) |
