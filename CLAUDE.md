# CLAUDE.md

This file provides project-level guidance to Claude Code (claude.ai/code) for the Intel XPU Backend for Triton.

## Overview

This is an out-of-tree Intel GPU backend for the [Triton](https://github.com/triton-lang/triton) compiler. The codebase is a fork of upstream Triton with Intel-specific extensions in `third_party/intel/`.

## Build Commands

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

Build tips:
- Set `MAX_JOBS=8` on machines with <64GB RAM.
- Set `MAX_JOBS=16` if machines has more than 64GB RAM.
- Set `TRITON_BUILD_WITH_CCACHE=true` for faster rebuilds.
- Set `TRITON_BUILD_WITH_CLANG_LLD=true` to use clang/lld (faster linking).
- Use `pip install -e . --no-build-isolation` for faster incremental builds.
- LLVM version is pinned in `cmake/llvm-hash.txt`.

## Testing

```bash
# Run tests via the Intel test script
scripts/test-triton.sh                      # default tests (unit + core + tutorial + microbench)
scripts/test-triton.sh --unit               # C++ unit tests (lit + gtest)
scripts/test-triton.sh --core               # core tests
scripts/test-triton.sh --minicore           # subset of core tests
scripts/test-triton.sh --intel              # Intel-specific tests only
scripts/test-triton.sh --language           # language tests
scripts/test-triton.sh --tutorial           # tutorial tests
scripts/test-triton.sh --benchmarks         # performance benchmarks
scripts/test-triton.sh --interpreter        # interpreter mode tests
scripts/test-triton.sh --inductor           # PyTorch inductor integration

# Run tests via Makefile (upstream-style)
make test-lit                               # MLIR lit tests
make test-cpp                               # C++ gtest unit tests
make test-unit                              # Python unit tests (requires GPU)
make test-nogpu                             # Tests that don't need a GPU

# Run a single Python test file
python -m pytest python/test/unit/language/test_core.py -x -s
python -m pytest python/test/unit/language/test_core.py::test_name -x -s

# Run a single lit test
<build_dir>/bin/triton-opt test/path/to/test.mlir <pass-flags>
```

Test skip lists for Intel are in `scripts/skiplist/`.

## Architecture

### Intel Backend Integration

Intel-specific code lives in `third_party/intel/` and is symlinked into the Python module tree:
- `python/triton/backends/intel/` → `third_party/intel/backend/`
- `python/triton/language/extra/intel/` → `third_party/intel/language/intel/`

The backend registers as the `xpu` target in Triton's multi-backend system.

### Compilation Pipeline

```
Triton Python → TTIR → TTGIR → LLVM IR → SPIR-V → ZEBIN
```

Each stage is orchestrated by `XPUBackend` in `third_party/intel/backend/compiler.py`:

1. **`make_ttir()`** — Triton IR optimization. Intel-specific passes in `third_party/intel/lib/Dialect/Triton/Transforms/` (RemoveBoundaryChecks, RemoveMasks, StrideVersioning, BlockPointerToTensorDesc, etc.)
2. **`make_ttgir()`** — GPU IR lowering. Core optimization passes in `third_party/intel/lib/TritonIntelGPUTransforms/` (Coalesce, AccelerateMatmul, Pipeline, MaterializeBlockPointer, ReduceVariableLiveness, etc.)
3. **`make_llir()`** — LLVM IR generation via `third_party/intel/lib/TritonIntelGPUToLLVM/` and `third_party/intel/lib/TritonGENToLLVM/`
4. **`make_spv()`** — SPIR-V translation via `third_party/intel/lib/Target/SPIRV/`
5. **`make_zebin()`** — Final binary via `ocloc` (Intel offline compiler), with auto GRF mode selection

### Custom MLIR Dialects

- **TritonGEN** (`third_party/intel/lib/Dialect/TritonGEN/`) — Intel GPU-specific operations
- **TritonIntelGPU** (`third_party/intel/lib/Dialect/TritonIntelGPU/`) — GPU IR with Intel layout attributes

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

### Runtime Stack

The driver (`third_party/intel/backend/driver.py`) integrates with:
- **SYCL runtime** (`libsycl.so`) — discovered via `icpx`, `ONEAPI_ROOT`, or `intel-sycl-rt` wheel
- **Level Zero** (`ze_loader`) — low-level GPU access
- **IGC** (Intel Graphics Compiler) — JIT compilation from SPIR-V

## C++ Coding Conventions

This project follows LLVM/MLIR coding standards (see `.github/copilot-instructions.md` for full details):
- **Naming**: camelCase for functions/variables, CamelCase for types/classes
- **Casting**: use `dyn_cast`, `cast`, `isa` — never C-style casts
- **Containers**: prefer `SmallVector`, `DenseMap`, `StringRef`, `ArrayRef`
- **Error handling**: use `llvm::Error`/`llvm::Expected<T>`, no exceptions
- **Include order**: main header, local, LLVM/MLIR, system

## IR Debugging

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
TRITON_KERNEL_DUMP_BEST_CONFIG=1 TRITON_PRINT_AUTOTUNING=1 python my_kernel.py
```

Full list of configuration knobs: `python/triton/knobs.py`. Intel-specific knobs are under `knobs.intel.*`.

## General Preferences

- Never ask for permission before running any read-only or non-destructive shell command (e.g. `grep`, `cat`, `find`, `ls`, `file`, `wc`, `diff`, `nm`, `objdump`, `ldd`, `which`, `env`, `echo`, `pwd`, etc.). Run them directly.

## Dependencies

- **oneAPI**: Primary dependency — install via [Intel PyTorch Dependency Bundle](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html)
- **Python requirements**: `python/requirements.txt` (build), `scripts/requirements-test.txt` (test)
