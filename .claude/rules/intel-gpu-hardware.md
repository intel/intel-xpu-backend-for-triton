---
description: 'Intel GPU hardware architecture: Xe generations, GRF modes, DPAS encoding, target capabilities, register pressure, compilation pipeline'
applyTo: '**/TritonIntelGPUTransforms/**/*.cpp, **/TritonIntelGPUTransforms/**/*.h, **/Dialect/TritonIntelGPU/**/*.td, **/Dialect/TritonIntelGPU/**/*.cpp, **/Dialect/TritonIntelGPU/**/*.h, **/backend/compiler.py, **/backend/driver.c, **/backend/driver.py, **/Analysis/**/*.h, **/Analysis/**/*.cpp, **/Analysis/**/*.tpp'
---

# Intel GPU Hardware Architecture

## Xe Architecture Overview

### Terminology Mapping
| Legacy Term | Modern Term | Abbreviation |
|-------------|-------------|--------------|
| Execution Unit (EU) | Xe Vector Engine | XVE |
| Systolic / DPAS unit | Xe Matrix eXtension | XMX |
| Dual Subslice (DSS) | Xe-core | XC |

### Architecture Generations

| Generation | Products | XVE per Xe-core | Threads/XVE | ALU Width | XMX Width | DPAS Exec Size |
|------------|----------|-----------------|-------------|-----------|-----------|----------------|
| Xe-LP | Tiger Lake, DG1 | 16 | 7 | 256-bit (SIMD8 FP32) | — | — |
| Xe-HPG | Arc A-series, DG2 | 16 | 8 | 256-bit | 1024-bit | 8 |
| Xe-HPC | Data Center GPU Max (PVC) | 8 | 8 | 512-bit (SIMD16 FP32) | 4096-bit | 16 |
| Xe2 | Arc B-series (BMG) | 8 | 8 | 256-bit | 4096-bit | 16 |
| Xe3P | Crescent Island (CRI) | 8 | 8 | 256-bit | TBD | 16 |

Key performance characteristics per Xe-core:
- **Xe-HPC**: 256 FP32 ops/cycle, 4096 FP16/BF16 ops/cycle (via XMX)
- **Xe-HPG**: 128 FP32 ops/cycle, 2048 FP16/BF16 ops/cycle (via XMX)

## GRF (General Register File)

Each hardware thread has a private register file.

### Register Specifications
- **Register width**: 32 bytes (256 bits)
- **Effective GRF payload** (SIMD16): 64 bytes (16 lanes × 4 bytes)

### GRF Modes

| Mode | Registers | Total Size | Threads per XVE | Build Flag |
|------|-----------|------------|-----------------|------------|
| Small GRF (128) | 128 | 4 KB | 8 (max occupancy) | `-cl-intel-128-GRF-per-thread` |
| Large GRF (256) | 256 | 8 KB | 4 (halved occupancy) | `-cl-intel-256-GRF-per-thread` |
| Auto Large GRF | Auto | Auto | Auto | `-cl-intel-enable-auto-large-GRF-mode` |

### Auto-GRF Mode Selection (`grf_mode='default'`)
1. Compile with default (small) GRF
2. Extract spill size from ZEBIN `.ze_info` section
3. If `spill_size > 1000` bytes → recompile with 256-GRF mode
4. Threshold of 1000 is empirical, aligned between `compiler.py` and `driver.c`

### Constraints
- **256-GRF requires `num_warps ≤ 32`** (because halved thread occupancy limits available hardware threads)
- `grf_mode` options: `'default'`, `'128'`, `'256'`, `'auto'`

## Subgroups and SIMD Execution

### Subgroup-to-Warp Mapping
In the Intel backend: **subgroup = warp**.
- Default `warp_size = 32` (configurable)
- DPAS operations use `threadsPerWarp = 16` (fixed)
- `warp_size` must be in device's `sub_group_sizes` list
- Workgroup size = `num_warps × warp_size`
- `num_warps` must be ≤ device's `max_num_sub_groups`

### Supported Subgroup Sizes (typical)
| Architecture | Subgroup Sizes |
|-------------|---------------|
| Xe-HPG (DG2/Arc A) | 8, 16, 32 |
| Xe-HPC (PVC) | 16, 32 |
| Xe2 (BMG) | 16, 32 |

### DPAS Execution Size
The DPAS N dimension equals `min(sub_group_sizes)`:
- **PVC, BMG**: execution_size = 16
- **DG2**: execution_size = 8

## Target Architectures

Architecture detection via `parse_device_arch()` in `arch_parser.c`, mapping SYCL architecture enums to string names.

| Target | Arch String | DPAS | 2D Block IO | Notes |
|--------|-------------|------|-------------|-------|
| PVC | `pvc` | Yes (exec=16) | Yes | Xe-HPC, Data Center GPU Max |
| DG2 | `dg2` | Yes (exec=8) | Yes | Xe-HPG, Arc A-series |
| BMG | `bmg` | Yes (exec=16) | Yes | Xe2, Arc B-series (Battlemage) |
| LNL | `lnl` | Yes (exec=16) | Yes | Xe2, Lunar Lake |
| MTL | `mtl` | TBD | TBD | Meteor Lake (integrated) |
| ARL-H | `arl_h` | TBD | TBD | Arrow Lake H |
| ARL-S | `arl_s` | TBD | TBD | Arrow Lake S |
| PTL-H | `ptl_h` | Yes (exec=16) | Yes | Xe3, Panther Lake H |
| PTL-U | `ptl_u` | Yes (exec=16) | Yes | Xe3, Panther Lake U |

## Device Capabilities

Capabilities queried via Level Zero and set as module attributes:

| Property | Module Attribute | Description |
|----------|-----------------|-------------|
| `has_subgroup_matrix_multiply_accumulate` | `ttig.support_subgroup_matrix_multiply_accumulate` | DPAS support |
| `has_subgroup_matrix_multiply_accumulate_bfloat8` | `ttig.support_subgroup_matrix_multiply_accumulate_bf8` | BF8/HF8 in DPAS (Xe3P+) |
| `has_subgroup_scaled_matrix_multiply_accumulate` | `ttig.support_subgroup_scaled_matrix_multiply_accumulate` | Block Scale DPAS (BDPAS) |
| `has_subgroup_2d_block_io` | `ttig.support_2d_block_io` | 2D block load/store/prefetch |
| `has_16bit_atomics` | `ttig.support_16bit_atomics` | Native 16-bit atomic operations |
| `has_bfloat16_arithmetic` | `ttig.support_bfloat16_arithmetic` | BF16 native arithmetic |
| `has_bfloat16_conversion` | `ttig.support_bfloat16_conversion` | BF16 conversion instructions |
| `has_f8_conversions` | `ttig.support_f8_conversion` | FP8 (E5M2/E4M3FN) conversions |
| `has_f4_conversions` | `ttig.support_f4_conversion` | FP4 (E2M1) conversions |
| `has_predicated_io` | `ttig.support_predicated_io` | Predicated load/store |
| `has_256b_prefetch` | `ttig.support_prefetch_256b` | 256-byte 2D block prefetch |

### LTS Driver Detection
LTS (Long Term Support) driver version threshold: `(1, 6, 35096, 9)`.
When `is_lts=true`, certain lowering paths use GenISA intrinsics instead of SPIR-V builtins.

## DpasEncodingAttr (`#ttig.dpas`)

Encoding attribute for DPAS layout in the TritonIntelGPU dialect.

### Parameters
| Parameter | Description | Values |
|-----------|-------------|--------|
| `repeatCount` | M dimension of DPAS tile | 1, 2, 4, **8** (typical) |
| `systolicDepth` | Depth of systolic array | Always **8** |
| `executionSize` | N dimension (SIMD width) | **16** (PVC/BMG) or **8** (ATSM) |
| `opsPerChannel` | K packing factor | 1 (TF32), 2 (FP16/BF16), 4 (INT8/FP8/FP4) |
| `warpsPerCTA` | Warp distribution in CTA | Array, e.g., [4, 1] |
| `repCluster` | Repetition cluster size | Array, optimization for memory access |
| `threadsPerWarp` | Subgroup size for DPAS | Currently only **16** |
| `fp4KPack` | FP4 packing along K | Optional, for F4E2M1 |

### DPASCapability Constants
```cpp
systolicDepth = 8;
repeatCount = 8;
opsChanBitWidths = 32;  // opsPerChannel = 32 / element_bitwidth
executionSize = 16 or 8;  // architecture-dependent
```

### Matrix Shape Derivation
```
M = repeatCount
N = executionSize
K = systolicDepth × opsPerChannel
```

Data is distributed row-major across threads in the subgroup:
- If column size = subgroup size: one scalar per thread = one row
- If column size < subgroup size: one scalar spans multiple rows
- If column size > subgroup size: one scalar covers a partial row

## DPAS Engine Types

### Xe2 (`DPASEngineTypeXe2`) — 19 type combinations
**Standard dot types** (D_C_A_B format):
- FP32_FP32_FP16_FP16 (default), FP32_FP32_BF16_BF16, FP32_FP32_TF32_TF32
- FP16_FP16_FP16_FP16, BF16_BF16_BF16_BF16
- U32_U32_U8_U8, S32_S32_S8_S8

**Scaled dot types** (for `DotScaledOp`):
- FP32_FP32_{BF16,FP16,FP8,FP4}_{FP8,FP16,BF16,FP4} (all cross-combinations)

### Xe3P (`DPASEngineTypeXe3P`) — adds:
- **BF16_BF16_FP8_FP8** (native BF16 accumulation with FP8 inputs)
- Same scaled types as Xe2

### Factory Selection
`DPASAnalysisFactory::createDPASAnalysis()` selects V1 (Xe2) or V2 (Xe3P) based on `support_subgroup_matrix_multiply_accumulate_bf8` module attribute.

## WarpEncodingAttr (`#ttig.warp`)

Thread tile distribution encoding:
- `sizePerThread`: elements computed per thread
- `threadsPerWarp`: subgroup size
- `order`: access order (fastest-changing dimension first)

## Subgroup2DBlockEncodingAttr (`#ttig.subgroup_2d_block`)

Encoding for 2D block I/O layouts:
- `instrShape`: (height, width) of the 2D block operation
- `numBlocks`: count of blocks per load
- `threadsPerWarp`: subgroup size
- `warpsPerCTA`: warp distribution
- `order`: access order
- `kWidth`: layout conversion parameter for K dimension

## Register Pressure Management

### ReduceVariableLiveness Pass
Activated when `opt.reduce_variable_liveness = true`. Thresholds from source:
- Total block size threshold: **32768 bytes**
- Large tensor threshold: **128 × 128 × 2 = 32768 bytes** (per-dimension shape ≥ 128)

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
convert_tdesc_to_block_pointer → convert_to_ttgpuir →
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

### Cache Sizes (approximate, per Xe-core)
| Architecture | L1 Cache / SLM | L2 Cache |
|-------------|----------------|----------|
| Xe-LP | 128 KB SLM per DSS | 16 MB per slice |
| Xe-HPG | 128 KB SLM, 256 KB L1 per Xe-core | 16 MB |
| Xe-HPC | 512 KB L1/SLM per Xe-core | 408 MB (HBM + cache) |

### LSC Fence Operations (vISA)
Memory ports (`LscSFID`):
- **UGM** (0x0): Untyped global memory
- **UGML** (0x1): Low-bandwidth untyped global memory (cross-tile)
- **TGM** (0x2): Typed global memory
- **SLM** (0x3): Shared local memory

Fence operations (`LscFenceOp`):
- **NONE** (0x0): No operation
- **EVICT** (0x1): Removes dirty and clean lines from L1
- **INVALIDATE** (0x2): Clears clean lines only, retains dirty
- **DISCARD** (0x3): Removes all lines without eviction
- **CLEAN** (0x4): Writes dirty lines to memory, keeps cached
- **FLUSHL3** (0x5): Flushes L3 cache only

Fence scopes (`LscScope`):
- **GROUP** (0x0): Threadgroup
- **LOCAL** (0x1): Local (DSSs)
- **TILE** (0x2): Multiple DSSs
- **GPU** (0x3): Entire GPU with LLC
- **GPUS** (0x4): All system GPUs
- **SYSTEM** (0x5): System-level
- **SYSACQ** (0x6): Device memory commitment

## Key Numerical Constants

| Item | Value |
|------|-------|
| Default warp_size | 32 |
| DPAS threadsPerWarp | 16 |
| GRF register width | 32 bytes |
| Small GRF register count | 128 |
| Large GRF register count | 256 |
| GRF payload (SIMD16) | 64 bytes |
| Spill threshold for auto-GRF | 1000 bytes |
| Max num_warps with 256-GRF | 32 |
| DPAS systolic depth | 8 |
| DPAS max repeat count | 8 |
| DPAS exec size (PVC/BMG) | 16 |
| DPAS exec size (DG2) | 8 |
| 2D Block load max tile_height | 32 |
| 2D Block store max tile_height | 8 |
| 2D Block max bytes per row | 64 (load/store) |
| 2D Block prefetch max bytes per row | 256 (if supported) |
| HW base address alignment | 64 bytes |
| SW compensated base alignment | 4 bytes |
