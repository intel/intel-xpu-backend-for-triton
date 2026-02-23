---
description: 'Intel GPU lowering: TritonGEN ops to SPIR-V builtins and GenISA intrinsics, cache control decorations, SPIR-V extensions'
applyTo: '**/TritonGENToLLVM/**/*.cpp, **/TritonGENToLLVM/**/*.h, **/TritonGENToSPIRV/**/*.cpp, **/Target/SPIRV/**/*.cpp, **/TritonIntelGPUToLLVM/**/*.cpp, **/TritonIntelGPUToLLVM/**/*.h'
---

# Intel GPU Lowering: TritonGEN to SPIR-V and GenISA

## Lowering Strategy

Two lowering paths exist for TritonGEN operations:

1. **SPIR-V builtins** (preferred) — used with newer (non-LTS) drivers
2. **GenISA intrinsics** (fallback) — used with LTS drivers or unsupported SPIR-V signatures

### `isSPVBuiltinAvailable()` Decision Logic
Returns `false` (use GenISA) when:
- Module has the `ttig.is_lts` attribute (LTS driver)
- The specific operation signature is not valid in the SPIR-V interface (see below)
- Subgroup size ≠ 16 (SPIR-V 2D block operations only support subgroup size 16)

## DPAS Lowering

### SPIR-V Path
Function: `__spirv_SubgroupMatrixMultiplyAccumulateINTEL`

Arguments: `(k_dim: i32, a: aTy, b: bTy, c: cTy, flags: i32)`

- `k_dim` = systolicDepth × opsPerChannel (e.g., 16 for BF16, 32 for INT8)
- Operand types are packed: BF16 operands passed as i16 (OpenCL convention)
- `flags` encodes precision information as a hex bitmask

### Type Packing Rules
| Precision | A type in MLIR | A type in SPIR-V call |
|-----------|---------------|----------------------|
| TF32 | f32 or i32 | f32 or i32 (no change) |
| BF16 | i16 | i16 (BF16 → i16 in OCL) |
| FP16 | i16 | i16 |
| INT8/FP8 | i16 | i16 (packed pairs) |

C/D operands may need bitcast (e.g., bf16 vector → i16 vector for SPIR-V call, then bitcast back).

### GenISA Path (LTS fallback)
Standard DPAS uses SPIR-V even on LTS. Only Block Scale DPAS always uses GenISA.

## Block Scale DPAS Lowering

**Always** uses GenISA intrinsics (no SPIR-V builtin available):

Function: `llvm.genx.GenISA.sub.group.bdpas.<type_mangling>`

Arguments: `(c, a, b, scaleA, scaleB, pa, pb, sd, rc, signless, precision_overrides...)`

- Default scale value when operand is absent: **0x7f** (represents 1.0 in E8M0 format)
- The scale value is broadcast to all lanes

## 2D Block I/O Lowering

### Load

**SPIR-V path**: `__spirv_Subgroup2DBlockLoad` (or `__spirv_Subgroup2DBlockLoadTranspose`, `__spirv_Subgroup2DBlockLoadTransform` for transpose/VNNI)

**GenISA path**: `llvm.genx.GenISA.LSC2DBlockRead.<type_mangling>`

GenISA argument order:
```
(ptr: i64, baseWidth: i32, baseHeight: i32, basePitch: i32,
 x: i32, y: i32, subBlockWidth: i32, subBlockHeight: i32,
 subBlockCount: i32, transform: i1, vnniTransform: i1,
 cache: i32)
```

### Store

**SPIR-V path**: `__spirv_Subgroup2DBlockStoreINTEL`

**GenISA path**: `llvm.genx.GenISA.LSC2DBlockWrite.<type_mangling>`

For SPIR-V stores, the value is first stored to a local alloca, then a pointer to the alloca is passed to the builtin.

### Prefetch

**SPIR-V path**: `__spirv_Subgroup2DBlockPrefetchINTEL`

**GenISA path**: `llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid`

Prefetch always uses SPIR-V (never falls back to GenISA) — `isSPVBuiltinAvailableImpl` returns `true` unconditionally for prefetch.

### Unsupported SPIR-V 2D Block Load Signatures
The following 2D block load configurations fall back to GenISA even on non-LTS drivers:
- 8-bit, 8×8, v_blocks=1, no VNNI (`intel_sub_group_2d_block_read_8b_8r8x1c`)
- 8-bit, 8×8, v_blocks=2, no VNNI (`intel_sub_group_2d_block_read_8b_8r8x2c`)
- 8-bit, 8×8, v_blocks=4, no VNNI (`intel_sub_group_2d_block_read_8b_8r8x4c`)
- 8-bit, 32×8, v_blocks=4, no VNNI (`intel_sub_group_2d_block_read_8b_32r8x4c`)
- 32-bit, 32×4, v_blocks=1, transpose (`intel_sub_group_2d_block_read_transpose_32b_32r4x1c`)

### 64-Byte Alignment Compensation
Hardware requires 64-byte aligned base addresses. The lowering compensates non-aligned addresses:
```
alignedPtr = ptr & ~0x3f           // Clear lower 6 bits
offsetInBytes = ptr & 0x3f         // Extract misalignment
adjustedBaseWidth = baseWidth + offsetInBytes
adjustedX = x + (offsetInBytes / elemSizeInBytes)
```
This is applied in `computeAlignedBasePtrWidthAndOffset()` for all 2D block operations.

## Split Barrier Lowering

### `triton_gen.split_barrier_arrive` → 3-step sequence:
1. `intel_manageable_barrier_init(num_producers: i32, num_consumers: i32) → ptr<3>`
   - Both counts set to `num_warps` (looked up from module)
2. `intel_manageable_barrier_arrive(bData: ptr<3>) → void`

The `init` call returns the barrier data pointer (in SLM, address space 3).

### `triton_gen.split_barrier_wait` → 2-step sequence:
1. `intel_manageable_barrier_wait(bData: ptr<3>) → void`
2. `intel_manageable_barrier_release(bData: ptr<3>) → void`

### Workgroup Barrier (`triton_gen.barrier`)
Lowered to `spirv::ControlBarrierOp` with:
- Execution scope: Workgroup
- Memory scope: Workgroup
- Memory semantics: derived from MemFence attribute

## Predicated I/O Lowering

### Predicated Load
Function: `llvm.genx.GenISA.PredicatedLoad.<type_mangling>`

Arguments: `(ptr, alignment, predicate, default_value)`

### Predicated Store
Function: `llvm.genx.GenISA.PredicatedStore.<type_mangling>`

Arguments: `(ptr, value, alignment, predicate)`

Both always use GenISA intrinsics (no SPIR-V equivalent).

## TF32 Conversion Lowering

Function: `__spirv_RoundFToTF32INTEL`

Lowers `triton_gen.f_to_tf32` to a SPIR-V builtin call. Operates on f32 scalars or vectors.

## Cache Control Decorations

### SPIR-V Metadata Format
Cache controls are encoded as `!spirv.DecorationCacheControlINTEL` LLVM IR metadata:

```llvm
!{i32 <decoration_id>, i32 <cache_level>, i32 <cache_control>, i32 <operand_number>}
```

**Decoration IDs**:
- **6442** = Load cache control
- **6443** = Store cache control

**Cache levels**: 0 = L1, 1 = L3

### LoadCacheControl → Decoration Mapping
| LoadCacheControl | L1 Decoration | L3 Decoration |
|-----------------|---------------|---------------|
| L1UC_L3UC | Uncached | Uncached |
| L1UC_L3C | Uncached | Cached |
| L1C_L3UC | Cached | Uncached |
| L1C_L3C | Cached | Cached |
| L1S_L3UC | Streaming | Uncached |
| L1S_L3C | Streaming | Cached |
| L1IAR_L3C | InvalidateAfterRead | Cached |

### Example LLVM IR
```llvm
%1 = load i32, ptr %0, !spirv.DecorationCacheControlINTEL !1
!1 = !{!2, !3}
!2 = !{i32 6442, i32 0, i32 1, i32 0}  ; Load, L1, Cached, operand 0
!3 = !{i32 6442, i32 1, i32 0, i32 0}  ; Load, L3, Uncached, operand 0
```

### Validation Rules
- Two decorations of the same kind (load or store) **cannot** target the same cache level
- Decorations list must be non-empty
- Per SPV_INTEL_cache_controls specification

## SPIR-V Extensions

The backend registers the following SPIR-V extensions (from `SPIRVTranslation.cpp`):

### Intel Extensions
| Extension | Purpose |
|-----------|---------|
| `SPV_INTEL_16bit_atomics` | 16-bit atomic operations |
| `SPV_INTEL_2d_block_io` | 2D block load/store/prefetch |
| `SPV_INTEL_arbitrary_precision_integers` | Non-standard integer widths |
| `SPV_INTEL_arithmetic_fence` | Arithmetic fencing |
| `SPV_INTEL_bfloat16_arithmetic` | BF16 native arithmetic |
| `SPV_INTEL_bfloat16_conversion` | BF16 conversions |
| `SPV_INTEL_cache_controls` | L1/L3 cache control decorations |
| `SPV_INTEL_float4` | FP4 type support |
| `SPV_INTEL_fp_conversions` | FP8 conversion operations |
| `SPV_INTEL_fp_fast_math_mode` | Fast math mode flags |
| `SPV_INTEL_inline_assembly` | Inline assembly support |
| `SPV_INTEL_kernel_attributes` | Kernel attribute decorations |
| `SPV_INTEL_memory_access_aliasing` | Memory aliasing hints |
| `SPV_INTEL_split_barrier` | Split-phase barriers |
| `SPV_INTEL_subgroup_matrix_multiply_accumulate` | DPAS operations |
| `SPV_INTEL_subgroups` | Subgroup operations |
| `SPV_INTEL_tensor_float32_conversion` | TF32 conversion |
| `SPV_INTEL_unstructured_loop_controls` | Loop control decorations |
| `SPV_INTEL_vector_compute` | Vector compute model |

### Khronos/EXT Extensions
| Extension | Purpose |
|-----------|---------|
| `SPV_EXT_shader_atomic_float_add` | Float atomic add |
| `SPV_EXT_shader_atomic_float16_add` | FP16 atomic add |
| `SPV_EXT_float8` | FP8 type support |
| `SPV_KHR_bfloat16` | BF16 type in SPIR-V |
| `SPV_KHR_bit_instructions` | Bitwise instructions |
| `SPV_KHR_non_semantic_info` | Non-semantic metadata |

### GenISA Unknown Intrinsics
When using GenISA intrinsics, the SPIR-V translator must be configured to allow unknown LLVM intrinsics (they are passed through and resolved by IGC during JIT compilation).
