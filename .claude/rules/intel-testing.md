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

> **IMPORTANT**: Before writing or modifying tests, you MUST read the full testing reference in `.claude/reference/passes-and-testing-reference.md` using the Read tool.

**Do not guess** test directory structure — read it from `.claude/reference/passes-and-testing-reference.md`.

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

**Do not guess** Intel pass CLI flags or FileCheck directives — read them from `.claude/reference/passes-and-testing-reference.md`.

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

**Do not guess** lit test environment variables or test module attribute syntax — read them from `.claude/reference/passes-and-testing-reference.md`.

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
build/.../bin/triton-opt test/TritonIntelGPU/loop-pipeline.mlir -tritonintelgpu-pipeline
```

## Python Tests (pytest)

### Directory Structure

- `python/test/unit/intel/` — Intel backend-specific tests
- `python/test/unit/language/` — Language feature tests (shared)
- `python/test/unit/runtime/` — Runtime tests
- `python/test/regression/` — Regression tests
- `python/test/gluon/` — Gluon dialect tests
- `python/test/conftest.py` — Global pytest configuration

**Do not guess** architecture detection function names (`is_xpu_*()`) or device capability checks — read them from `.claude/reference/passes-and-testing-reference.md`.

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

Standard steps for a Python kernel test:
1. Call `check_type_supported(dtype_str, device)` for dtype-dependent tests
2. Define `@triton.jit` kernel
3. Generate reference data with `numpy_random((M, N), dtype_str=..., rs=RandomState(17))`
4. Convert with `to_triton(x, device=device)` / `to_numpy(x_tri)`
5. Launch kernel and compare: `np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)`

**Do not guess** numerical tolerance conventions, pytest fixtures, or test runner details — read them from `.claude/reference/passes-and-testing-reference.md`.

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

- `unittest/` — Upstream unit tests (Analysis, Dialect/TritonGPU)
- `third_party/intel/unittest/` — Intel-specific tests (Dialect/TritonIntelGPU, Conversion/TritonIntelGPUToLLVM)

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

Inherit from `::testing::Test`, load required dialects in `SetUp()` or constructor:
```cpp
class DPAStoLinearLayoutTest : public ::testing::Test {
public:
  void SetUp() { ctx.getOrLoadDialect<TritonIntelGPUDialect>(); }
protected:
  MLIRContext ctx;
};
```
For tests needing IR construction, add `OpBuilder`/`Block` members. For shared context across tests, use `static MLIRContext`.

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

Each directory contains per-suite skip files: `intel.txt`, `language.txt`, `gluon.txt`, `tutorials.txt`, `mxfp.txt`, `scaled_dot.txt`, `triton_kernels.txt`, etc.

### Skip List Format

```
# Comment with tracking issue link
python/test/unit/intel/test_block_load.py::test_block_load_dpas_layout
python/test/unit/intel/test_core.py::test_gather_warp_shuffle[param1-param2]
python/test/unit/language/test_core.py::test_dot3d[r".*-int8-.*$"]@regexp
06-fused-attention
```

Formats: exact test path, path with parameters, regex patterns (with `@regexp` suffix), tutorial names, and `#` comments.

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

**Do not guess** test runner categories (`test-triton.sh` flags), environment variables, or Makefile targets — read them from `.claude/reference/passes-and-testing-reference.md`.

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
