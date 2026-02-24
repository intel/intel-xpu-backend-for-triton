---
description: 'Intel GPU TritonGEN dialect operations: DPAS, 2D block I/O, barriers, predicated I/O, format conversion — semantics and hardware constraints'
applyTo: '**/Dialect/TritonGEN/**/*.cpp, **/Dialect/TritonGEN/**/*.h, **/Dialect/TritonGEN/**/*.td, **/TritonGENToLLVM/**/*.cpp, **/TritonGENToLLVM/**/*.h, **/TritonGENToSPIRV/**/*.cpp'
---

# Intel GPU TritonGEN Dialect Operations

## Overview

The TritonGEN dialect (`mlir::triton::TritonGEN`, prefix `triton_gen`) defines Intel GPU-specific operations. Source: `third_party/intel/include/Dialect/TritonGEN/IR/` (definitions) and `third_party/intel/lib/Dialect/TritonGEN/IR/` (implementation).

## Memory Spaces

The backend follows the SPIR-V / OpenCL storage class convention.

> **IMPORTANT**: Before writing or modifying TritonGEN dialect operations, you MUST read the full operation specs in `.claude/reference/operations-reference.md` using the Read tool.

**Do not guess** memory space constant values (kFunction=0 through kGeneric=4) — read them from `.claude/reference/operations-reference.md`.

Rules:
- Split barrier data pointers use address space **3** (workgroup/SLM).
- Sub-group block reads/writes require address space **1** (global) or **3** (local).

## Synchronization Operations

### `triton_gen.barrier`
Workgroup barrier ensuring all outstanding memory transactions are complete.
- **Parameter**: `mem_fence` (MemFence enum)
- **vISA mapping**: `BARRIER` (opcode 0x59)
- **Constraint**: Behavior undefined in divergent control flow. Thread-group execution model only.

### `triton_gen.split_barrier_arrive` / `triton_gen.split_barrier_wait`
Split-phase (asynchronous) barrier for producer-consumer patterns.
- `arrive` returns `!llvm.ptr<3>` (barrier data in SLM address space 3)
- `wait` takes the barrier data pointer and blocks until arrival count reaches zero
- **Scope**: Workgroup, with subgroups as barrier objects
- **Memory semantics**: Implicit acquire-release
- **vISA mapping**: `SBARRIER` (opcode 0x7c)

### Named Barriers (`NBARRIER`, opcode 0x60)
General consumer-producer synchronization for a subgroup of threads. Not directly in TritonGEN dialect but available in vISA:
- Barrier ID: 0-31
- Types: producer-consumer (0), producer-only (1), consumer-only (2)
- All participating threads must use identical `id`, `num_producers`, `num_consumers`

## DPAS — `triton_gen.dpas`

Matrix multiply-accumulate: **D = C + A × B**

### Matrix Dimensions
```
M = repeat count (rc)
N = execution size (fixed: 16 for PVC/BMG, 8 for ATSM)
K = systolic_depth × OPS_PER_CHAN
```

**Do not guess** DPAS type/precision values (OPS_PER_CHAN, element type rules, verifier constraints, register alignment) — read the lookup tables from `.claude/reference/operations-reference.md`.

## Block Scale DPAS — `triton_gen.bdpas`

Matrix multiply-accumulate with scaling for MXFP: **D = C + (A × B) × scaleA × scaleB**

**Do not guess** BDPAS constraints, scale types, or supported precisions — read them from `.claude/reference/operations-reference.md`.

## 2D Block I/O Operations

**Do not guess** 2D block I/O parameters, address/tile constraints, tile width limits, or per-operation rules (load/store/prefetch) — read them from `.claude/reference/operations-reference.md`.

## Sub-group Block I/O, Predicated I/O, Format Conversion, Enums

**Do not guess** sub-group block I/O details, predicated I/O parameters, format conversion specs, or enum values (PrecisionType, LoadCacheControl, StoreCacheControl, MemFence, MemScope) — read them from `.claude/reference/operations-reference.md`.

Cache control SPIR-V decoration mappings are in `intel-gpu-lowering.md`.
