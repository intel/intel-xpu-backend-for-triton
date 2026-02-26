# Build, Debug, and Encoding Reference

> On-demand reference data. Not automatically loaded. Rules files in `.claude/rules/` point here.

## From CLAUDE.md

### Build Commands

```bash
# Build Triton (incremental after initial setup)
scripts/compile-triton.sh

# Build LLVM from source + Triton
scripts/compile-triton.sh --llvm

# Build with ccache
scripts/compile-triton.sh --ccache

# Clean build
scripts/compile-triton.sh --clean

# Incremental rebuild (after initial pip install)
ninja -C $(python -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())' 2>/dev/null || echo build)

# Just rebuild triton-opt
make triton-opt

# Upstream-style dev install (pip-based)
make dev-install
```

### Build Tips
- Set `MAX_JOBS=8` on machines with <64GB RAM.
- Set `MAX_JOBS=16` if machines have more than 64GB RAM.
- Set `TRITON_BUILD_WITH_CCACHE=true` for faster rebuilds.
- Set `TRITON_BUILD_WITH_CLANG_LLD=true` to use clang/lld (faster linking).
- Use `pip install -e . --no-build-isolation` for faster incremental builds.
- LLVM version is pinned in `cmake/llvm-hash.txt`.

### Key Source Directories

| Directory | Purpose |
|---|---|
| `third_party/intel/backend/` | Python backend (compiler.py, driver.py) |
| `third_party/intel/lib/TritonIntelGPUTransforms/` | Core GPU optimization passes |
| `third_party/intel/lib/TritonIntelGPUToLLVM/` | GPU IR → LLVM IR conversion |
| `third_party/intel/lib/TritonGENToLLVM/` | GEN dialect → LLVM conversion |
| `third_party/intel/lib/Target/SPIRV/` | SPIR-V translation |
| `third_party/intel/lib/Dialect/` | Custom dialect definitions (TritonGEN, TritonIntelGPU) |
| `third_party/intel/include/` | C++ headers for all Intel-specific components |
| `third_party/intel/triton_xpu.cc` | Plugin registration — binds all passes to Python |
| `python/triton/` | Python language, runtime, and compiler framework |
| `unittest/` | C++ unit tests (Analysis, Dialect, Tools) |
| `scripts/` | Build, test, and CI automation |

### IR Debugging

```bash
# Full IR dump at every compilation stage
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py

# Dump MLIR passes (optionally filter by kernel name)
MLIR_ENABLE_DUMP=1 python my_kernel.py
MLIR_ENABLE_DUMP=_my_kernel python my_kernel.py

# Dump LLVM IR
LLVM_IR_ENABLE_DUMP=1 python my_kernel.py

# Run in interpreter mode (no GPU needed)
TRITON_INTERPRET=1 python my_kernel.py

# Best autotuned config with readable dump dirs
TRITON_PRINT_AUTOTUNING=1 python my_kernel.py
```

Full list of configuration knobs: `python/triton/knobs.py`. Intel-specific knobs are under `knobs.intel.*`.

### Dependencies

- **oneAPI**: Primary dependency — install via [Intel PyTorch Dependency Bundle](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html)
- **Python requirements**: `python/requirements.txt` (build), `scripts/requirements-test.txt` (test)

## From intel-layout-encodings.md

### DpasEncodingAttr Parameters

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `repeatCount` | unsigned | Range [1, 8] | M dimension of DPAS tile |
| `systolicDepth` | unsigned | Must be **8** | Depth of systolic array (hardware-fixed) |
| `executionSize` | unsigned | **16** (PVC/BMG) or **8** (DG2) | N dimension (SIMD width) |
| `opsPerChannel` | unsigned | 1, 2, or 4 | K packing factor: `32 / element_bitwidth` |
| `warpsPerCTA` | ArrayRef<unsigned> | Rank 2 or 3 | Warp distribution in CTA, e.g., [4, 1] |
| `repCluster` | ArrayRef<unsigned> | Rank 2 or 3 | Repetition cluster size for memory optimization |
| `threadsPerWarp` | unsigned | Must be **16**, must be >= executionSize | Subgroup size for DPAS |
| `fp4KPack` | optional<unsigned> | If present, must be **2**; only valid when opsPerChannel=4 | FP4 packing along K |

### opsPerChannel by Element Type

| Element Type | Bitwidth | opsPerChannel | K (depth=8) |
|-------------|----------|---------------|-------------|
| TF32 | 32 | 1 | 8 |
| BF16, FP16 | 16 | 2 | 16 |
| INT8, FP8 (E5M2/E4M3FN) | 8 | 4 | 32 |
| FP4 (E2M1) | 4 | 4 (with fp4KPack=2) | 64 |

### WarpEncodingAttr Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sizePerThread` | ArrayRef<unsigned> | Elements computed per thread in each dimension |
| `threadsPerWarp` | ArrayRef<unsigned> | Number of threads per warp in each dimension |
| `order` | ArrayRef<unsigned> | Access order (fastest-changing dimension first) |

### Subgroup2DBlockEncodingAttr Parameters

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `warpsPerCTA` | ArrayRef<unsigned> | Rank 2 | Warp distribution in CTA |
| `CGALayout` | CGAEncodingAttr | — | CTA layout encoding |
| `instrShape` | ArrayRef<unsigned> | Rank 2 (height, width) | Shape of 2D block operation |
| `numBlocks` | unsigned | — | Count of vertically adjacent blocks per load |
| `order` | ArrayRef<unsigned> | Rank 2 | Access order |
| `kWidth` | unsigned | 1, 2, or 4 | Layout conversion parameter for K dimension |
| `threadsPerWarp` | unsigned | Must be **16** | Subgroup size for 2D block I/O |

### DotOperandEncodingAttr Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `opIdx` | unsigned | — | 0 = Operand A, 1 = Operand B |
| `parent` | Attribute | — | Parent MMA encoding (DpasEncodingAttr for Intel) |
| `kWidth` | unsigned | 0 | K-dimension element packing width |

## From intel-gpu-lowering.md

### Type Packing Rules
| Precision | A type in MLIR | A type in SPIR-V call |
|-----------|---------------|----------------------|
| TF32 | f32 or i32 | f32 or i32 (no change) |
| BF16 | i16 | i16 (BF16 → i16 in OCL) |
| FP16 | i16 | i16 |
| INT8/FP8 | i16 | i16 (packed pairs) |

C/D operands may need bitcast (e.g., bf16 vector → i16 vector for SPIR-V call, then bitcast back).

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
