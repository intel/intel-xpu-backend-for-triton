# Hardware Reference

> On-demand reference data. Not automatically loaded. Rules files in `.claude/rules/` point here.

## From intel-gpu-hardware.md

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

### GRF Register Specifications
- **Register width**: 32 bytes (256 bits)
- **Effective GRF payload** (SIMD16): 64 bytes (16 lanes × 4 bytes)

### GRF Modes

| Mode | Registers | Total Size | Threads per XVE | Build Flag |
|------|-----------|------------|-----------------|------------|
| Small GRF (128) | 128 | 4 KB | 8 (max occupancy) | `-cl-intel-128-GRF-per-thread` |
| Large GRF (256) | 256 | 8 KB | 4 (halved occupancy) | `-cl-intel-256-GRF-per-thread` |
| Auto Large GRF | Auto | Auto | Auto | `-cl-intel-enable-auto-large-GRF-mode` |

### Supported Subgroup Sizes (typical)
| Architecture | Subgroup Sizes |
|-------------|---------------|
| Xe-HPG (DG2/Arc A) | 8, 16, 32 |
| Xe-HPC (PVC) | 16, 32 |
| Xe2 (BMG) | 16, 32 |

### Target Architectures

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

### Device Capabilities

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

### DPAS Hardware Constants

```cpp
systolicDepth = 8;
repeatCount = 8;
opsChanBitWidths = 32;  // opsPerChannel = 32 / element_bitwidth
executionSize = 16 or 8;  // architecture-dependent
```

DpasEncodingAttr parameters and layout details are in `intel-layout-encodings.md`.

### DPAS Engine Types

#### Xe2 (`DPASEngineTypeXe2`) — 19 type combinations
**Standard dot types** (D_C_A_B format):
- FP32_FP32_FP16_FP16 (default), FP32_FP32_BF16_BF16, FP32_FP32_TF32_TF32
- FP16_FP16_FP16_FP16, BF16_BF16_BF16_BF16
- U32_U32_U8_U8, S32_S32_S8_S8

**Scaled dot types** (for `DotScaledOp`):
- FP32_FP32_{BF16,FP16,FP8,FP4}_{FP8,FP16,BF16,FP4} (all cross-combinations)

#### Xe3P (`DPASEngineTypeXe3P`) — adds:
- **BF16_BF16_FP8_FP8** (native BF16 accumulation with FP8 inputs)
- Same scaled types as Xe2

#### Factory Selection
`DPASAnalysisFactory::createDPASAnalysis()` selects V1 (Xe2) or V2 (Xe3P) based on `support_subgroup_matrix_multiply_accumulate_bf8` module attribute.

### Memory Hierarchy

#### Cache Sizes (approximate, per Xe-core)
| Architecture | L1 Cache / SLM | L2 Cache |
|-------------|----------------|----------|
| Xe-LP | 128 KB SLM per DSS | 16 MB per slice |
| Xe-HPG | 128 KB SLM, 256 KB L1 per Xe-core | 16 MB |
| Xe-HPC | 512 KB L1/SLM per Xe-core | 408 MB (HBM + cache) |

#### LSC Fence Operations (vISA)
Memory ports: UGM (0x0, global), UGML (0x1, cross-tile), TGM (0x2, typed), SLM (0x3, local). Fence ops range from NONE (0x0) through FLUSHL3 (0x5), with scopes from GROUP (0x0) to SYSACQ (0x6). See vISA spec for full enum values.
