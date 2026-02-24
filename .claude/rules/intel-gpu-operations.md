---
description: 'Intel GPU TritonGEN dialect operations: DPAS, 2D block I/O, barriers, predicated I/O, format conversion — semantics and hardware constraints'
applyTo: '**/Dialect/TritonGEN/**/*.cpp, **/Dialect/TritonGEN/**/*.h, **/Dialect/TritonGEN/**/*.td, **/TritonGENToLLVM/**/*.cpp, **/TritonGENToLLVM/**/*.h, **/TritonGENToSPIRV/**/*.cpp'
---

# Intel GPU TritonGEN Dialect Operations

## Overview

The TritonGEN dialect (`mlir::triton::TritonGEN`, prefix `triton_gen`) defines Intel GPU-specific operations. Source: `third_party/intel/include/Dialect/TritonGEN/IR/` (definitions) and `third_party/intel/lib/Dialect/TritonGEN/IR/` (implementation).

## Memory Spaces

The backend follows the SPIR-V / OpenCL storage class convention:

| Constant | Value | Description |
|----------|-------|-------------|
| `kFunction` | 0 | OpenCL workitem / private memory |
| `kCrossWorkgroup` | 1 | OpenCL global memory |
| `kUniformConstant` | 2 | OpenCL constant memory |
| `kWorkgroup` | 3 | OpenCL local / SLM (Shared Local Memory) |
| `kGeneric` | 4 | OpenCL generic memory |

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

### OPS_PER_CHAN by Precision
| Precision | Bitwidth | OPS_PER_CHAN | K (depth=8) |
|-----------|----------|-------------|-------------|
| TF32 | 32 | 1 | 8 |
| BF16, FP16 | 16 | 2 | 16 |
| U8, S8, F8E5M2, F8E4M3FN | 8 | 4 | 32 |
| U4, S4, U2, S2, F4E2M1 | <8 | 8 | 64 |

### Verifier Constraints (from `TritonGENOps.cpp`)
- `rc` must be **1, 2, 4, or 8**
- `pa` and `pb` must match, **except** FP8 variants (E5M2/E4M3FN can mix with each other)
- C and D must have identical type
- C/D vector length = `rc` (or `ceil(rc/2)` for packed types)
- B vector length = 8 (systolic depth) or `ceil(8/2)`
- **vISA opcode**: 0x83. Format: `DPAS.W.A.SD.RC (Exec_size) <dst> <src0> <src1> <src2>`

### Element Type Rules by Precision

| Precision | C/D Element Type | A Element Type | B Element Type |
|-----------|-----------------|----------------|----------------|
| U8, S8 | i32 | i16 | i32 |
| FP16 | f16 or f32 | i16 | i32 |
| BF16 | bf16 or f32 | i16 | i32 |
| TF32 | f32 | f32 or i32 | f32 or i32 |
| F8E5M2, F8E4M3FN | bf16 or f32 | i16 | i32 |
| F4E2M1 | bf16 or f32 | i16 | i32 |

Note: For non-TF32 precisions, A and B operands are packed into i16/i32 containers.

### Register Alignment (vISA)
- Dst, Src0 (C), Src1 (B): **GRF-aligned**
- Src2 (A): **DWORD-aligned** (alignment depends on precision and systolic depth)
- No source modifiers or register regions allowed

## Block Scale DPAS — `triton_gen.bdpas`

Matrix multiply-accumulate with scaling for MXFP: **D = C + (A × B) × scaleA × scaleB**

### Additional Constraints (beyond DPAS common rules)
- `rc` must be exactly **8** (no other values allowed for BDPAS)
- Scale shapes: scaleA = M×(K/32), scaleB = (K/32)×N
- Scale operands are optional (default to 1.0 in E8M0 format = 0x7f when absent)

### Scale Type by Precision
| Precision | Scale Type |
|-----------|------------|
| BF16, FP16, F8E5M2, F8E4M3FN | i8 |
| F4E2M1 | vector<2×i8> |

### Supported Precisions for BDPAS
BF16, FP16, F8E5M2 (bf8), F8E4M3FN (hf8), F4E2M1 (e2m1). Does **not** support TF32, U8, or S8.

## 2D Block I/O Operations

### Common Parameters
| Parameter | Unit | Description |
|-----------|------|-------------|
| `ptr` | — | Base address of the matrix |
| `base_width` | **bytes** | Matrix stride-one dimension size |
| `base_height` | elements | Matrix height |
| `base_pitch` | **bytes** | Row stride (≥ base_width, multiple of 16) |
| `x` | **elements** | Tile X offset |
| `y` | elements | Tile Y offset |
| `tile_width` | elements | Tile width |
| `tile_height` | elements | Tile height |
| `v_blocks` | — | Number of vertically adjacent blocks |
| `elem_size_in_bits` | bits | 8, 16, 32, or 64 |

### Address Payload Constraints (Verifier)
- `base_width` ≥ 64 bytes, ≤ 2²⁴
- `base_pitch` ≥ 64 bytes, ≤ 2²⁴, multiple of **16 bytes**, ≥ `base_width`
- `base_height` ≤ 2²⁴
- `base_width` alignment: must be aligned to `max(4, elem_size)` bytes
- X offset alignment: multiple of 4 for 8-bit elements, multiple of 2 for 16-bit elements

### Tile Shape Constraints
- `tile_width`, `tile_height`, `v_blocks` must all be **powers of 2**
- `tile_height` ≤ 32 (load/prefetch), ≤ **8** (store)
- `v_blocks` ≤ 4 (load/prefetch), must be **1** (store)
- Max bytes per row: `elem_size_in_bits × tile_width × v_blocks ≤ max_bits_per_row`
  - Load/store: max **64 bytes** per row (512 bits)
  - Prefetch: **64-256 bytes** per row (depends on `support_prefetch_256b` attribute)

### Tile Width Limits by Element Size

| elem_size_in_bits | Min tile_width | Max tile_width (load/store) | Max tile_width (prefetch 256B) |
|-------------------|---------------|---------------------------|-------------------------------|
| 8 | 4 | 64 | 256 |
| 16 | 2 | 32 | 128 |
| 32 | 1 | 16 | 64 |
| 64 | 1 | 8 | 32 |

### v_blocks Limits by Element Size (load only, not prefetch)
- 32-bit elements: v_blocks ∈ {1, 2} (not 4)
- 64-bit elements: v_blocks must be 1

### `triton_gen.2Dblockload` — Load-Specific Rules
- `transpose` and `vnni_transform` are **mutually exclusive**
- **Transpose rules** (only for 32-bit and 64-bit elements):
  - v_blocks must be 1
  - 32-bit: tile_width 1-8
  - 64-bit: tile_height must be exactly 8, tile_width ∈ {1, 2, 4}
- **VNNI transform rules** (only for 8-bit and 16-bit elements):
  - 8-bit: tile_height ≥ 4
  - 16-bit: tile_height ≥ 2
- Result element type is 32-bit when elem_size_in_bits=32 or vnni_transform=true
- Out-of-bounds elements are filled with **0**

### `triton_gen.2Dblockstore` — Store-Specific Rules
- `tile_height` ≤ **8** (more restrictive than load's 32)
- `v_blocks` must be **1** (no multi-block stores)

### `triton_gen.2Dblockprefetch`
- No result (void). Side effect: writes to L2Cache resource.
- Relaxed v_blocks constraints (no element-size-specific limits for prefetch)
- Supports 256-byte prefetch when `ttig.support_prefetch_256b` module attribute is set

## Sub-group Block I/O

### `triton_gen.sub_group_block_read` / `sub_group_block_write`
Strided block access within a subgroup. Each work-item accesses:
```
ptr[sub_group_local_id + i × sub_group_size]
```

- **Element types**: i8, i16, i32, i64
- **Vector lengths**: 2, 4, 8 (16 only for i8)
- **Address spaces**: global (1) or local/SLM (3)
- **Alignment**: pointer must be aligned to element type size

## Predicated I/O

### `triton_gen.predicated_load`
Conditional load: if predicate is true, loads from pointer; otherwise returns default value.
- Parameters: `ptr`, `alignment` (i64), `predicate` (i1), `default_value`, `cache_control` (LoadCacheControl, default: DEFAULT)
- Requires `has_predicated_io` device capability

### `triton_gen.predicated_store`
Conditional store: if predicate is true, stores to pointer; otherwise no-op.
- Parameters: `ptr`, `value`, `alignment` (i64), `predicate` (i1), `cache_control` (StoreCacheControl, default: DEFAULT)

## Format Conversion

### `triton_gen.f_to_tf32`
Converts f32 to TF32 (Tensor Float 32, E8M10) with rounding to nearest even. Operates on scalars or vectors of f32. Input and output types are both f32 (TF32 is stored in f32 container).

## Enum Reference

### PrecisionType
| Name | Value | Mnemonic | Notes |
|------|-------|----------|-------|
| UNUSED | 0 | unused | |
| U8 | 1 | u8 | |
| U4 | 2 | u4 | |
| U2 | 3 | u2 | |
| S8 | 4 | i8 | |
| S4 | 5 | i4 | |
| S2 | 6 | i2 | |
| F8E5M2 | 7 | bf8 | New in Xe3P |
| F8E4M3FN | 8 | hf8 | New in Xe3P |
| TF32 | 10 | tf32 | |
| BF16 | 11 | bf16 | |
| FP16 | 12 | f16 | |
| F4E2M1 | 13 | e2m1 | New in Xe3P |

Note: Values must match IGC implementation (coupling acknowledged in source as FIXME).

### LoadCacheControl
| Name | Value | Description |
|------|-------|-------------|
| DEFAULT | 0 | Default behavior |
| L1UC_L3UC | 1 | L1 uncached, L3 uncached |
| L1UC_L3C | 2 | L1 uncached, L3 cached |
| L1C_L3UC | 3 | L1 cached, L3 uncached |
| L1C_L3C | 4 | L1 cached, L3 cached |
| L1S_L3UC | 5 | L1 streaming, L3 uncached |
| L1S_L3C | 6 | L1 streaming, L3 cached |
| L1IAR_L3C | 7 | L1 invalidate-after-read, L3 cached |

### StoreCacheControl
| Name | Value | Description |
|------|-------|-------------|
| DEFAULT | 0 | Default behavior |
| L1UC_L3UC | 1 | L1 uncached, L3 uncached |
| L1UC_L3WB | 2 | L1 uncached, L3 write-back |
| L1WT_L3UC | 3 | L1 write-through, L3 uncached |
| L1WT_L3WB | 4 | L1 write-through, L3 write-back |
| L1S_L3UC | 5 | L1 streaming, L3 uncached |
| L1S_L3WB | 6 | L1 streaming, L3 write-back |
| L1WB_L3WB | 7 | L1 write-back, L3 write-back |
| L1WT_L3WT | 8 | L1 write-through, L3 write-through |
| L1S_L3S | 9 | L1 streaming, L3 streaming |

### MemFence
| Name | Value | Mnemonic |
|------|-------|----------|
| NONE | 0 | None |
| LOCAL | 1 | Local |
| GLOBAL | 2 | Global |
| LOCAL_AND_GLOBAL | 3 | LocalAndGlobal |

### MemScope
| Name | Value | Mnemonic |
|------|-------|----------|
| WORK_ITEM | 0 | WorkItem |
| WORK_GROUP | 1 | WorkGroup |
| DEVICE | 2 | Device |
| ALL_SVM_DEVICES | 3 | AllSvmDevices |
| SUB_GROUP | 4 | SubGroup |

Cache control SPIR-V decoration mappings are in `intel-gpu-lowering.md`.
