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
- Set `MAX_JOBS=16` if machines have more than 64GB RAM.
- Set `TRITON_BUILD_WITH_CCACHE=true` for faster rebuilds.
- Set `TRITON_BUILD_WITH_CLANG_LLD=true` to use clang/lld (faster linking).
- Use `pip install -e . --no-build-isolation` for faster incremental builds.
- LLVM version is pinned in `cmake/llvm-hash.txt`.

## Pre-commit

Install via `pip install pre-commit`. Run `python3 -m pre_commit run --show-diff-on-failure --color=always --all-files --verbose` before pushing a PR to ensure all checks pass (formatting, linting, etc.).

## Architecture

### Intel Backend Integration

Intel-specific code lives in `third_party/intel/` and is symlinked into the Python module tree:
- `python/triton/backends/intel/` → `third_party/intel/backend/`
- `python/triton/language/extra/intel/` → `third_party/intel/language/intel/`

The backend registers as the `xpu` target in Triton's multi-backend system.

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
TRITON_PRINT_AUTOTUNING=1 python my_kernel.py
```

Full list of configuration knobs: `python/triton/knobs.py`. Intel-specific knobs are under `knobs.intel.*`.

## General Preferences

- Never ask for permission before running any read-only or non-destructive shell command (e.g. `grep`, `cat`, `find`, `ls`, `file`, `wc`, `diff`, `nm`, `objdump`, `ldd`, `which`, `env`, `echo`, `pwd`, etc.). Run them directly.

## Dependencies

- **oneAPI**: Primary dependency — install via [Intel PyTorch Dependency Bundle](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html)
- **Python requirements**: `python/requirements.txt` (build), `scripts/requirements-test.txt` (test)
