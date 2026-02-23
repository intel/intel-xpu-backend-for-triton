---
description: 'Testing conventions: MLIR lit tests, Python pytest, C++ gtest, skip lists, test runner scripts, CI configuration'
applyTo: '**/test/**/*.mlir, **/test/**/*.py, **/unittest/**/*.cpp, **/scripts/test-triton.sh, **/scripts/pytest-utils.sh, **/scripts/skiplist/**/*.txt, **/conftest.py, **/_internal_testing.py, **/CMakeLists.txt'
---

# Testing Conventions

## Test Infrastructure Overview

Three test frameworks, each with distinct conventions:

| Framework | Location | Tool | Purpose |
|-----------|----------|------|---------|
| MLIR Lit | `test/` | `triton-opt` + FileCheck | IR transformation correctness |
| Python pytest | `python/test/` | pytest + pytest-xdist | End-to-end kernel correctness |
| C++ gtest | `unittest/`, `third_party/intel/unittest/` | GoogleTest | Unit testing C++ analysis/dialect code |

## MLIR Lit Tests

### Configuration

**`test/lit.cfg.py`** — Main configuration:
- Test format: `ShTest` (shell-based)
- Test suffixes: `.mlir` and `.ll`
- FileCheck option: `FILECHECK_OPTS = "--enable-var-scope"` (scopes CHECK variables per CHECK-LABEL)

**Registered tools**: `triton-opt`, `triton-llvm-opt`, `mlir-translate`, `llc`, `%PYTHON`

### Test Directory Structure

Intel-specific tests are in these directories:
```
test/
├── Triton/Intel/                     # TTIR Intel passes
│   ├── BlockPointerToTensorDesc/
│   ├── RemoveBoundaryChecks/
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

### File Naming Conventions

- Name reflects the pass or feature: `accelerate-matmul.mlir`, `coalesce.mlir`
- Architecture variants use suffix: `accelerate-matmul-pvc.mlir`, `accelerate-matmul-ats.mlir`
- Validation tests: `tritonintelgpu-invalid.mlir`
- Regression tests: `remove_layout_conversions_5947.mlir` (issue number)

### RUN Line Conventions

**Basic pattern:**
```
// RUN: triton-opt %s -split-input-file -pass-name | FileCheck %s
```

**Multiple passes chained:**
```
// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s
```

**Environment variables for feature flags:**
```
// RUN: env TRITON_INTEL_PREDICATED_LOAD=0 triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,NO-PREDICATED
// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,PREDICATED
```

**Diagnostic verification (for error/invalid tests):**
```
// RUN: triton-opt %s -split-input-file -verify-diagnostics %s
```

### Key RUN Line Components

| Component | Purpose |
|-----------|---------|
| `%s` | Input test file (auto-substituted) |
| `-split-input-file` | Split at `// -----` markers — each section compiles separately |
| `--check-prefixes=A,B` | Match both `// A:` and `// B:` lines |
| `--implicit-check-not=PATTERN` | Assert pattern does NOT appear in output |
| `--dump-input-context=20` | Show 20 lines of context on mismatch |
| `-allow-unregistered-dialect` | Allow unregistered dialects in IR |

### Intel Pass CLI Flags

TTIR passes:
- `-triton-intel-remove-boundary-checks`
- `-triton-intel-remove-masks`
- `-triton-intel-stride-versioning`
- `-triton-intel-block-pointer-to-tdesc`
- `-triton-intel-tdesc-to-block-pointer`
- `-triton-intel-fuse-reshape`

TTGIR passes:
- `-tritonintelgpu-coalesce`
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

### Variable Capture Patterns

```mlir
// Capture a value:
// CHECK: %[[LOAD:.*]] = tt.load %arg0

// Reference captured value:
// CHECK: tt.store %arg1, %[[LOAD]]

// Match any value (no capture):
// CHECK: tt.dot {{.*}}, %[[B]], {{.*}}

// Multiple captures on one line:
// CHECK: llvm.func @kernel(%[[A:.*]]: i32, %[[B:.*]]: !llvm.ptr<1>)
```

### Prefix-Based Conditional Checking

For testing multiple code paths in one file:
```mlir
// RUN: env FLAG=0 triton-opt %s ... | FileCheck %s --check-prefixes=CHECK,PATH-A
// RUN: env FLAG=1 triton-opt %s ... | FileCheck %s --check-prefixes=CHECK,PATH-B

// CHECK: common_operation
// PATH-A: specific_to_path_a
// PATH-B: specific_to_path_b
```

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
  "ttig.support_sg_2d_block",
  "ttig.support_dpas",
  "ttig.support_predicated_io"
} {
  // test functions here
}
```

### Running Lit Tests

```bash
# All lit tests via Makefile
make test-lit

# Via build directory
cd build/cmake.linux-x86_64-cpython-3.10 && lit -v test/

# Specific directory
lit test/TritonIntelGPU/

# Single test file
lit test/TritonIntelGPU/accelerate-matmul-pvc.mlir

# Direct triton-opt invocation
build/.../bin/triton-opt test/TritonIntelGPU/coalesce.mlir -tritonintelgpu-coalesce
```

## Python Tests (pytest)

### Directory Structure

```
python/test/
├── unit/
│   ├── intel/                  # Intel backend-specific
│   │   ├── test_block_io.py
│   │   ├── test_block_load.py
│   │   ├── test_conversions.py
│   │   ├── test_core.py
│   │   ├── test_driver.py
│   │   ├── test_native_code_generation.py
│   │   └── test_regressions.py
│   ├── language/               # Language features (shared)
│   │   ├── test_core.py
│   │   ├── test_matmul.py
│   │   ├── test_standard.py
│   │   └── ...
│   ├── runtime/                # Runtime tests
│   └── tools/                  # Tooling tests
├── regression/                 # Regression tests
├── gluon/                      # Gluon dialect tests
└── conftest.py                 # Global pytest configuration
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

```python
# Type support check — call at start of test
def check_type_supported(dtype, device):
    if device in ['xpu']:
        if dtype in [torch.float64, "float64"] and not xpu_has_fp64():
            pytest.xfail("float64 not supported on current xpu hardware")

# Thread/warp configuration check
def check_threads_supported(num_warps, threads_per_warp, device):
    if device != "xpu":
        return
    props = triton.runtime.driver.active.utils.get_device_properties(...)
    if threads_per_warp not in props['sub_group_sizes']:
        pytest.xfail('unsupported warp size')
    if threads_per_warp * num_warps > props['max_work_group_size']:
        pytest.xfail('unsupported workgroup size')
```

### Pytest Markers and Decorators

**Parameterization:**
```python
@pytest.mark.parametrize("dtype_x, dtype_y, op", [
    ("int32", "int32", "+"), ("float16", "float16", "*"), ...
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_binary_op(dtype_x, dtype_y, op, num_ctas, device):
    ...
```

**Conditional skip/xfail:**
```python
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(
    not triton.runtime.driver.active.get_current_target().arch['has_subgroup_2d_block_io'],
    reason="Block loads not supported", run=False)
def test_block_load(device):
    ...
```

**Interpreter compatibility:**
```python
@pytest.mark.interpreter  # Marks test as interpreter-compatible
def test_basic_ops(device):
    ...
```

### Test Structure Pattern

Standard pattern for a Python kernel test:
```python
def test_feature(M, N, dtype_str, device):
    # 1. Check device/type support
    check_type_supported(dtype_str, device)

    # 2. Define Triton kernel
    @triton.jit
    def kernel(X, Y, Z, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = x + y
        tl.store(Z + offs, z)

    # 3. Generate reference data
    rs = RandomState(seed=17)
    x = numpy_random((M, N), dtype_str=dtype_str, rs=rs)
    y = numpy_random((M, N), dtype_str=dtype_str, rs=rs)
    z_ref = x + y

    # 4. Convert to device tensors
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    z_tri = to_triton(np.empty_like(z_ref), device=device)

    # 5. Launch kernel
    grid = (M * N // 128,)
    kernel[grid](x_tri, y_tri, z_tri, BLOCK=128)

    # 6. Compare with tolerance
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
```

### Numerical Tolerance Conventions

| Precision | Typical rtol | Typical atol | Notes |
|-----------|-------------|-------------|-------|
| float32 | 0.01 | — | Standard |
| float16 | 0.01 | 7e-3 | May need atol |
| bfloat16 | 0.5 | — | Large tolerance |
| int types | exact | exact | Use `np.testing.assert_equal` |
| float8 | 0.1 | 0.1 | Very loose |

### Test Data Generation

```python
from triton._internal_testing import numpy_random, to_triton, to_numpy

# Seed-controlled random data (seed=17 by default)
x = numpy_random((M, N), dtype_str="float16", rs=RandomState(17))

# Integer data avoids zero (division-safe)
x = numpy_random((M,), dtype_str="int32")  # x[x == 0] = 1

# Convert numpy → device tensor
x_tri = to_triton(x, device=device, dst_type="bfloat16")

# Convert device tensor → numpy
x_np = to_numpy(x_tri)
```

### Pytest Fixtures

From `conftest.py`:
- `device(request)` — Returns device from `--device` CLI option
- `fresh_triton_cache()` — Forces recompilation (sets `knobs.compilation.always_compile = True`)
- `fresh_knobs()` — Resets all knobs except build/nvidia/amd
- `with_allocator()` — Sets up custom memory allocator

### Running Python Tests

```bash
# Via test runner script
scripts/test-triton.sh --intel              # Intel-specific tests
scripts/test-triton.sh --core               # Core tests (language + intel + runtime)
scripts/test-triton.sh --language            # Language feature tests
scripts/test-triton.sh --interpreter         # Interpreter mode (no GPU)

# Direct pytest invocation
python -m pytest python/test/unit/intel/ -x -s --device xpu
python -m pytest python/test/unit/language/test_core.py::test_name -x -s --device xpu

# Parallel execution (default: 8 workers)
python -m pytest -n 8 --dist=worksteal python/test/unit/intel/
```

## C++ Unit Tests (gtest)

### Directory Structure

```
unittest/
├── Analysis/
│   └── UtilityTest.cpp
└── Dialect/TritonGPU/
    ├── DialectTest.cpp
    ├── SwizzleTest.cpp
    ├── LinearLayoutConversionsTest.cpp
    └── DumpLayoutTest.cpp

third_party/intel/unittest/
├── Dialect/TritonIntelGPU/
│   ├── DPAStoLinearLayoutTest.cpp
│   └── LinearLayoutConversionsTest.cpp
└── Conversion/TritonIntelGPUToLLVM/
    └── XeAsmFormatTest.cpp
```

### CMake Registration

```cmake
add_triton_ut(
  NAME TestDPASLinearLayout
  SRCS DPAStoLinearLayoutTest.cpp
  LIBS
    TritonAnalysis TritonGPUIR TritonIntelGPUIR
    TritonIntelGPUTransforms LLVMSupport MLIRSupport
)
```

Tests are auto-discovered via `gtest_discover_tests()` with 60-second timeout.

### MLIR Context Setup

**Per-test context (typical for Intel tests):**
```cpp
class DPAStoLinearLayoutTest : public ::testing::Test {
public:
  void SetUp() {
    ctx.getOrLoadDialect<TritonIntelGPUDialect>();
  }
protected:
  MLIRContext ctx;
};
```

**With IR construction:**
```cpp
class XeAsmFormatTest : public ::testing::Test {
protected:
  XeAsmFormatTest() {
    ctx.loadDialect<arith::ArithDialect>();
    createValues();
  }
  void createValues() {
    OpBuilder builder(&ctx);
    builder.setInsertionPointToStart(&block);
    v[0] = arith::ConstantIntOp::create(builder, builder.getUnknownLoc(), 1, 1);
  }
  MLIRContext ctx;
  Block block;
  Value v[4];
};
```

**Static shared context:**
```cpp
class InferLayoutTest : public ::testing::Test {
public:
  InferLayoutTest()
      : inferLayout(
            ctx.getOrLoadDialect<TritonGPUDialect>()
                ->getRegisteredInterface<DialectInferLayoutInterface>()) {}
protected:
  static MLIRContext ctx;
  DialectInferLayoutInterface *inferLayout;
};
```

### Running C++ Tests

```bash
# Via Makefile
make test-cpp

# Via CTest
cd build/cmake.linux-x86_64-cpython-3.10 && ctest .

# Run specific test binary
build/.../unittest/TestDPASLinearLayout
```

## Skip Lists

### Directory Organization

Skip lists are organized by GPU architecture in `scripts/skiplist/`:
```
skiplist/
├── default/          # Common skips for all GPUs
├── a770/             # Intel Arc A770 (Xe-HPG)
├── arl-h/            # Arrow Lake H
├── arl-s/            # Arrow Lake S
├── mtl/              # Meteor Lake
├── lts/              # LTS driver
└── xe2/              # Xe2 (BMG)
```

Each directory contains per-suite skip files:
```
intel.txt           # Intel-specific tests
language.txt        # Language feature tests
runtime.txt         # Runtime tests
debug.txt           # Debug tests
gluon.txt           # Gluon tests
tutorials.txt       # Tutorial tests
mxfp.txt            # MXFP tests
scaled_dot.txt      # Scaled dot tests
triton_kernels.txt  # Triton kernel tests
```

### Skip List Format

**Exact test path:**
```
python/test/unit/intel/test_block_load.py::test_block_load_dpas_layout
```

**With parameters:**
```
python/test/unit/intel/test_core.py::test_gather_warp_shuffle[src_shape1-indices_shape1-0-linear<{...}>]
```

**Regex patterns (with `@regexp` suffix):**
```
python/test/unit/language/test_core.py::test_dot3d[r".*-int8-.*$"]@regexp
python/test/gluon/test_lowerings.py::test_scan_layouts[r"True-.*"]@regexp
```

**Tutorial names (simple):**
```
06-fused-attention
08-grouped-gemm
```

**Comments (link to tracking issue):**
```
# https://github.com/intel/intel-xpu-backend-for-triton/issues/3921
python/test/unit/intel/test_block_load.py::test_block_load_dpas_layout
```

### Skip List Application

From `scripts/pytest-utils.sh`:
- Skip lists passed via `--skip-from-file=path/to/suite.txt`
- `--select-fail-on-missing` ensures skipped tests exist (catches stale skip entries)
- Multiple skip files combined with semicolons
- Extra suffix-based skip lists via `TRITON_EXTRA_SKIPLIST_SUFFIXES`

### TEST_UNSKIP Mode

Setting `TEST_UNSKIP=true` ignores all skip/xfail decorators:
- Adds `--timeout=500` per test
- Adds `--max-worker-restart=500` for crash tolerance
- Useful for validating which skips are still needed

## Test Runner Script

### `scripts/test-triton.sh` Categories

| Flag | Category | Parallel | Notes |
|------|----------|----------|-------|
| `--unit` | C++ lit + gtest | Yes | `ctest` + `lit -v` |
| `--intel` | Intel backend Python | 8 workers | `python/test/unit/intel/` |
| `--language` | Language features | 8 workers | Excludes mxfp, scaled_dot |
| `--core` | Combined core | Mixed | intel + language + mxfp + debug + runtime |
| `--runtime` | Runtime | Serial | Avoids race conditions |
| `--interpreter` | Interpreter mode | 16 workers | `TRITON_INTERPRET=1`, CPU only |
| `--tutorial` | Tutorials | Serial | Runs numbered tutorials |
| `--benchmarks` | Microbenchmarks | — | Performance benchmarks |
| `--inductor` | PyTorch inductor | — | Integration tests |

**Default** (`scripts/test-triton.sh` with no flags) runs: unit + core + tutorial + microbenchmarks + triton_kernels.

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRITON_TEST_SKIPLIST_DIR` | `scripts/skiplist/default` | Skip list directory |
| `TRITON_TEST_SUITE` | — | Suite name for reports and skip list lookup |
| `TRITON_TEST_REPORTS` | `false` | Enable JUnit XML reports |
| `TRITON_TEST_REPORTS_DIR` | `$HOME/reports/$TIMESTAMP` | Report output directory |
| `TRITON_TEST_IGNORE_ERRORS` | `false` | Continue on test failures |
| `TRITON_TEST_WARNING_REPORTS` | `false` | Capture pytest warnings |
| `TEST_UNSKIP` | `false` | Ignore skip/xfail decorators |
| `TRITON_DISABLE_LINE_INFO` | — | Disable debug line info (faster compile) |
| `TRITON_INTERPRET` | — | Run in interpreter mode (no GPU) |
| `PYTEST_MAX_PROCESSES` | 8 | Parallel worker count |

### Makefile Test Targets

| Target | What it runs |
|--------|-------------|
| `make test-lit` | MLIR lit tests (`ninja check-triton-lit-tests`) |
| `make test-cpp` | C++ gtest (`ninja check-triton-unit-tests`) |
| `make test-unit` | Full Python test suite |
| `make test-nogpu` | Tests without GPU (lit + cpp + frontend) |
| `make test` | Everything (lit + cpp + python) |

## Writing New Tests

### New Lit Test Checklist

1. Place in the appropriate directory under `test/` matching the pass location
2. Name file after the feature/pass being tested
3. Include `// RUN:` line with `-split-input-file` for multiple test cases
4. Use `// CHECK-LABEL:` to scope each function's checks
5. Declare required module attributes (num-warps, threads-per-warp, capabilities)
6. Separate test cases with `// -----`
7. Use `// COM:` for explanatory comments
8. For negative tests, use `-verify-diagnostics` and `// expected-error` annotations

### New Python Test Checklist

1. Place in `python/test/unit/intel/` for Intel-specific tests
2. Use `@pytest.mark.skipif(not is_xpu(), reason="...")` for XPU-only tests
3. Call `check_type_supported()` early for dtype-dependent tests
4. Use `numpy_random()` with explicit seed for reproducibility
5. Use `to_triton()` / `to_numpy()` for tensor conversion
6. Set appropriate `rtol`/`atol` for the precision being tested
7. Add to relevant skip list if known to fail on specific architectures

### New C++ Test Checklist

1. Place in `third_party/intel/unittest/` for Intel-specific tests
2. Register via `add_triton_ut()` in CMakeLists.txt with required LIBS
3. Inherit from `::testing::Test` and load required dialects in `SetUp()`
4. Use `MLIRContext` for attribute construction and verification
5. Tests auto-discover via `gtest_discover_tests()`
