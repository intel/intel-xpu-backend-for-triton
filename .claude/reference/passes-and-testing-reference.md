# Passes and Testing Reference

> On-demand reference data. Not automatically loaded. Rules files in `.claude/rules/` point here.

## From intel-pass-patterns.md

### TableGen File Locations

| Scope | TableGen File | Pass Prefix |
|-------|--------------|-------------|
| TTIR transforms | `intel/include/Dialect/Triton/Transforms/Passes.td` | `triton-intel-<name>` |
| TTGIR transforms | `intel/include/Dialect/TritonIntelGPU/Transforms/Passes.td` | `tritonintelgpu-<name>` |
| GPU→TritonGEN | `intel/include/GPUToTritonGEN/Passes.td` | `convert-gpu-to-tritongen` |
| TritonGEN→LLVM | `intel/include/TritonGENToLLVM/Passes.td` | `convert-tritongen-to-llvm` |
| TritonGEN→SPIR-V | `intel/include/TritonGENToSPIRV/Passes.td` | `convert-tritongen-to-spirv` |
| IntelGPU→LLVM | `intel/include/TritonIntelGPUToLLVM/Passes.td` | `convert-triton-intel-gpu-to-llvm` |
| Module annotation | `intel/include/TritonAnnotateModule/Passes.td` | `triton-annotate-module` |

### Pattern Base Class Reference

| Base Class | Signature | TypeConverter | Use Case |
|-----------|-----------|---------------|----------|
| `OpRewritePattern<Op>` | `matchAndRewrite(op, PatternRewriter &)` | No | Intra-dialect transforms |
| `ConvertOpToLLVMPattern<Op>` | `matchAndRewrite(op, OpAdaptor, ConversionPatternRewriter &)` | Yes | Lowering to LLVM |
| `ConvertTritonGPUOpToLLVMPattern<Op>` | Same as above + target info | Yes | Triton GPU → LLVM |
| `OpConversionPattern<Op>` | `matchAndRewrite(op, OpAdaptor, ConversionPatternRewriter &)` | Yes | Generic dialect conversion |

### Standard Aliases

```cpp
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;
```

### Common Utilities

#### Layout and Encoding Queries (`Utility.h`)
```cpp
bool hasDpasEncoding(RankedTensorType t);        // Has DpasEncodingAttr
bool hasDotDpasEncoding(RankedTensorType t);     // Has DotOperand with DPAS parent
std::optional<DotOperandEncodingAttr> getDotEncoding(RankedTensorType t);
Attribute inferSrcEncoding(Operation *op, Attribute encoding);
```

#### DPAS Calculation Helpers
```cpp
SmallVector<unsigned> calculateDPASInstShapeA(unsigned rc, unsigned sd, unsigned opc);
SmallVector<unsigned> calculateWarpsPerTile(unsigned capRC, unsigned capExec,
                                            ArrayRef<int64_t> shape, unsigned numWarps);
SmallVector<unsigned> calculateRepCluster(const DPASCapability &cap, ...);
```

#### Layout Propagation
```cpp
LogicalResult getConvertBackwardSlice(
    OpOperand &root, SetVector<Value> &slice, Attribute rootEncoding,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation = nullptr);
```

#### DPAS Analysis
```cpp
auto dpasAnalysis = ttgi::DPASAnalysisFactory::createDPASAnalysis(mod);
auto result = ttgi::DPASAnalysisFactory::canUseDPAS(funcOp, dpasAnalysis);
// Returns DPASAnalysisResult::True/False/Maybe
```

#### Lowering Helpers
```cpp
TritonLLVMOpBuilder b(loc, rewriter);  // Builder with convenience methods
createDeviceFunctionCall(rewriter, fnName, retTy, argTys, args, attrs);
intel::mangle(fnName, argTypes);       // SPIR-V name mangling
```

### Capability Gating Attributes

Module attributes for device capabilities are listed in `intel-gpu-hardware.md` (Device Capabilities). Use `TritonIntelGPUDialect::get<Capability>AttrName()` getters to query them. Key pass-to-capability dependencies:

- **MaterializeBlockPointer, Pipeline**: `support_2d_block_io`
- **AccelerateMatmul** (via DPASAnalysis): `support_subgroup_matrix_multiply_accumulate`
- **DPASAnalysisFactory** (Xe2 vs Xe3P): `support_subgroup_matrix_multiply_accumulate_bf8`
- **LoadStoreOpToLLVM**: `support_predicated_io`
- **isSPVBuiltinAvailable()**: `is_lts`
- **ArithEmulation**: `support_bfloat16_arithmetic`
- **Type conversion lowering**: `support_bfloat16_conversion`, `support_f8_conversion`, `support_f4_conversion`
- **AtomicOpLowering**: `support_16bit_atomics`

### Complete Pass Inventory

#### TTIR Transforms (`lib/Dialect/Triton/Transforms/`)

| Pass | CLI Flag | Pattern | Gate |
|------|----------|---------|------|
| RemoveMasks | `triton-intel-remove-masks` | Manual walk + TypeSwitch | — |
| StrideVersioning | `triton-intel-stride-versioning` | Manual walk + if-conversion | — |
| BlockPointerToTensorDesc | `triton-intel-block-pointer-to-tdesc` | Manual pattern matching | — |
| RewriteTensorDescToPointer | `triton-intel-rewrite-tensor-descriptor-to-pointer` | Manual walk | — |
| FuseReshape | `triton-intel-fuse-reshape` | Manual pattern matching | — |
| SimplifySignedArithmetic | `triton-intel-simplify-signed-arithmetic` | Manual pattern matching | — |

#### TTGIR Transforms (`lib/TritonIntelGPUTransforms/`)

| Pass | CLI Flag | Pattern | Gate |
|------|----------|---------|------|
| AccelerateMatmul | `tritonintelgpu-accelerate-matmul` | OpRewritePattern (greedy) | DPASAnalysis |
| MaterializeBlockPointer | `tritonintelgpu-materialize-block-pointer` | Manual walk + annotation | `support_2d_block_io` |
| OptimizeDotOperands | `tritonintelgpu-optimize-dot-operands` | OpRewritePattern (greedy) | — |
| RemoveLayoutConversions | `tritonintelgpu-remove-layout-conversions` | Multi-stage (walk + greedy) | — |
| ReduceDataDuplication | `tritonintelgpu-reduce-data-duplication` | OpRewritePattern (greedy) | — |
| Pipeline | `tritonintelgpu-pipeline` | Loop transform + options | `support_2d_block_io` |
| ReduceVariableLiveness | `tritonintelgpu-reduce-variable-liveness` | Manual walk + reorder | — |
| OptimizeReductionLocality | `tritonintelgpu-optimize-reduction-locality` | Manual walk + reshape | — |
| RewriteStackPtr | `tritonintelgpu-rewrite-stack-ptr` | Manual walk + symbol replace | — |
| AnnotateModule | `triton-annotate-module` | Attribute annotation (struct options) | — |

#### Conversion/Lowering Passes

| Pass | CLI Flag | Pattern | Source |
|------|----------|---------|--------|
| ConvertTritonIntelGPUToLLVM | `convert-triton-intel-gpu-to-llvm` | Multi-phase ConvertOp + TypeConverter | `lib/TritonIntelGPUToLLVM/` |
| AllocateSharedMemory | `intel-allocate-shared-memory` | Manual walk + metadata | `lib/TritonIntelGPUToLLVM/` |
| ConvertTritonGENToLLVM | `convert-tritongen-to-llvm` | ConvertOpToLLVMPattern | `lib/TritonGENToLLVM/` |
| ConvertTritonGENToSPIRV | `convert-tritongen-to-spirv` | Dialect translation | `lib/TritonGENToSPIRV/` |
| ConvertGPUToTritonGEN | `convert-gpu-to-tritongen` | OpRewritePattern + func call lowering | `lib/GPUToTritonGEN/` |

#### Post-Processing (`lib/Target/LLVMIR/`)

| Transform | File | Description |
|-----------|------|-------------|
| LLVMIRGuardMaskedDivRem | `LLVMIRGuardMaskedDivRem.cpp` | Guard masked div/rem against zero divisors |
| PostProcess | `PostProcess.cpp` | Final LLVM IR cleanup |

## From intel-testing.md

### Test Directory Structure

Intel-specific tests are in these directories:
```
test/
├── Triton/Intel/                     # TTIR Intel passes
│   ├── BlockPointerToTensorDesc/
│   ├── RemoveMasks/
│   ├── StrideVersioning/
│   └── ...
├── TritonIntelGPU/                   # TTGIR Intel passes
│   ├── RemoveLayoutConversions/
│   ├── accelerate-matmul-pvc.mlir
│   ├── coalesce.mlir
│   ├── loop-pipeline.mlir
│   └── ...
├── Conversion/intel/                 # Lowering passes
│   ├── tritongpu_to_gen.mlir
│   ├── tritonintelgpu_to_llvm.mlir
│   └── ...
├── Analysis/intel/                   # Analysis tests
└── TritonGEN/                        # GEN dialect tests
```

### Intel Pass CLI Flags

TTIR passes:
- `-triton-intel-remove-masks`
- `-triton-intel-stride-versioning`
- `-triton-intel-block-pointer-to-tdesc`
- `-triton-intel-fuse-reshape`

TTGIR passes:
- `-tritonintelgpu-accelerate-matmul`
- `-tritonintelgpu-optimize-dot-operands`
- `-tritonintelgpu-remove-layout-conversions`
- `-tritonintelgpu-materialize-block-pointer`
- `-tritonintelgpu-pipeline`
- `-tritonintelgpu-reduce-data-duplication`
- `-tritonintelgpu-reduce-variable-liveness`
- `-tritonintelgpu-optimize-reduction-locality`

Conversion passes:
- `--convert-triton-intel-gpu-to-llvm`
- `--convert-tritongen-to-llvm`
- `--intel-allocate-shared-memory`

### FileCheck Directives

| Directive | Purpose | Example |
|-----------|---------|---------|
| `CHECK:` | Match on next unmatched line | `// CHECK: tt.load` |
| `CHECK-LABEL:` | Named section (resets variable scope) | `// CHECK-LABEL: @my_func` |
| `CHECK-NEXT:` | Must be on immediately following line | `// CHECK-NEXT: llvm.return` |
| `CHECK-SAME:` | Continues on same line | `// CHECK-SAME: %[[VAL:.*]]` |
| `CHECK-DAG:` | Order-independent match | `// CHECK-DAG: [[PTR:%.+]]` |
| `CHECK-NOT:` | Pattern must NOT appear | `// CHECK-NOT: tt.trans` |
| `CHECK-COUNT-N:` | Pattern appears exactly N times | `// CHECK-COUNT-2: scf.for` |
| `COM:` | Comment (ignored by FileCheck) | `// COM: test explanation` |

### Environment Variables Used in Lit Tests

| Variable | Purpose |
|----------|---------|
| `TRITON_INTEL_PREDICATED_LOAD` | Enable/disable predicated loads |
| `TRITON_INTEL_PREDICATED_STORE` | Enable/disable predicated stores |
| `TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS` | Block I/O for all layouts |
| `TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32` | DPAS with warp size 32 |
| `TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT` | One matrix per block-transposed load |
| `TRITON_INTEL_2DBLOCK_MULTIPLE_C_MATRICES_PER_LOAD` | Multiple C matrices per 2D load |
| `TRITON_INTEL_REMOVELAYOUTCONVERSION_SUPPORT_FOR_LOOP` | Layout conversion through for-loops |

### Test Module Attributes

Tests declare hardware capabilities and configuration via module attributes:
```mlir
module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 16 : i32,
  "ttig.support_2d_block_io",
  "ttig.support_subgroup_matrix_multiply_accumulate",
  "ttig.support_predicated_io"
} {
  // test functions here
}
```

### Architecture Detection Functions

From `triton/_internal_testing.py`:

```python
is_xpu()         # Any Intel XPU
is_xpu_pvc()     # Xe-HPC (Data Center GPU Max)
is_xpu_bmg()     # Xe2 (Battlemage / Arc B-series)
is_xpu_dg2()     # Xe-HPG (Arc A-series, DG2)
is_xpu_cri()     # Xe3P (Crescent Island)
is_xpu_lnl()     # Lunar Lake
is_xpu_mtl()     # Meteor Lake
is_xpu_arl_h()   # Arrow Lake H
is_xpu_arl_s()   # Arrow Lake S
is_xpu_ptl_h()   # Panther Lake H
is_xpu_ptl_u()   # Panther Lake U

# Generation shortcuts:
is_xpu_xe2()     # Same as is_xpu_bmg()
is_xpu_xe3()     # ptl_h or ptl_u
is_xpu_xe3p()    # Same as is_xpu_cri()
```

### Device Capability Checks

From `triton/_internal_testing.py`:
- `check_type_supported(dtype, device)` — xfails on unsupported dtypes (e.g., float64)
- `check_threads_supported(num_warps, threads_per_warp, device)` — xfails on unsupported warp/workgroup sizes

### Numerical Tolerance Conventions

| Precision | Typical rtol | Typical atol | Notes |
|-----------|-------------|-------------|-------|
| float32 | 0.01 | — | Standard |
| float16 | 0.01 | 7e-3 | May need atol |
| bfloat16 | 0.5 | — | Large tolerance |
| int types | exact | exact | Use `np.testing.assert_equal` |
| float8 | 0.1 | 0.1 | Very loose |

### Pytest Fixtures

From `conftest.py`:
- `device(request)` — Returns device from `--device` CLI option
- `fresh_triton_cache()` — Forces recompilation (sets `knobs.compilation.always_compile = True`)
- `fresh_knobs()` — Resets all knobs except build/nvidia/amd
- `with_allocator()` — Sets up custom memory allocator

### Test Runner Script Categories

| Flag | Category | Parallel | Notes |
|------|----------|----------|-------|
| `--unit` | C++ lit + gtest | Yes | `ctest` + `lit -v` |
| `--intel` | Intel backend Python | 8 workers | `python/test/unit/intel/` |
| `--language` | Language features | 8 workers | Excludes mxfp, scaled_dot |
| `--core` | Combined core | Mixed | intel + language + mxfp + debug + runtime |
| `--minicore` | Subset of core | Mixed | Smaller subset of core tests |
| `--runtime` | Runtime | Serial | Avoids race conditions |
| `--interpreter` | Interpreter mode | 16 workers | `TRITON_INTERPRET=1`, CPU only |
| `--tutorial` | Tutorials | Serial | Runs numbered tutorials |
| `--benchmarks` | Microbenchmarks | — | Performance benchmarks |
| `--inductor` | PyTorch inductor | — | Integration tests |

**Default** (`scripts/test-triton.sh` with no flags) runs: unit + core + tutorial + microbenchmarks + triton_kernels.

### Key Test Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRITON_TEST_SKIPLIST_DIR` | `scripts/skiplist/default` | Skip list directory |
| `TEST_UNSKIP` | `false` | Ignore skip/xfail decorators |
| `TRITON_INTERPRET` | — | Run in interpreter mode (no GPU) |
| `PYTEST_MAX_PROCESSES` | 8 | Parallel worker count |

Other variables: `TRITON_TEST_SUITE` (suite name), `TRITON_TEST_REPORTS`/`TRITON_TEST_REPORTS_DIR` (JUnit XML), `TRITON_TEST_IGNORE_ERRORS`, `TRITON_DISABLE_LINE_INFO`.

### Makefile Test Targets

| Target | What it runs |
|--------|-------------|
| `make test-lit` | MLIR lit tests (`ninja check-triton-lit-tests`) |
| `make test-cpp` | C++ gtest (`ninja check-triton-unit-tests`) |
| `make test-unit` | Full Python test suite |
| `make test-nogpu` | Tests without GPU (lit + cpp + frontend) |
| `make test` | Everything (lit + cpp + python) |
