# CLAUDE.md

This file provides project-level guidance to Claude Code (claude.ai/code) for the Intel XPU Backend for Triton.

## Overview

This is an out-of-tree Intel GPU backend for the [Triton](https://github.com/triton-lang/triton) compiler. The codebase is a fork of upstream Triton with Intel-specific extensions in `third_party/intel/`.

## Build

Quick start: `scripts/compile-triton.sh` (incremental), `make dev-install` (pip-based).

> **IMPORTANT**: Before troubleshooting builds or configuring build options, you MUST read the full build commands and tips in `.claude/reference/build-and-debug-reference.md` using the Read tool.

## Pre-commit

Install via `pip install pre-commit`. Run `python3 -m pre_commit run --show-diff-on-failure --color=always --all-files --verbose` before pushing a PR to ensure all checks pass (formatting, linting, etc.).

## Architecture

### Intel Backend Integration

Intel-specific code lives in `third_party/intel/` and is symlinked into the Python module tree:
- `python/triton/backends/intel/` → `third_party/intel/backend/`
- `python/triton/language/extra/intel/` → `third_party/intel/language/intel/`

The backend registers as the `xpu` target in Triton's multi-backend system.

> **IMPORTANT**: Before navigating or modifying the project structure, you MUST read the full directory table in `.claude/reference/build-and-debug-reference.md` using the Read tool.

### Runtime Stack

The driver (`third_party/intel/backend/driver.py`) integrates with:
- **SYCL runtime** (`libsycl.so`) — discovered via `icpx`, `ONEAPI_ROOT`, or `intel-sycl-rt` wheel
- **Level Zero** (`ze_loader`) — low-level GPU access
- **IGC** (Intel Graphics Compiler) — JIT compilation from SPIR-V

## IR Debugging

> **IMPORTANT**: Before debugging IR or setting dump environment variables, you MUST read the full IR debugging reference in `.claude/reference/build-and-debug-reference.md` using the Read tool.

Full knobs list: `python/triton/knobs.py`. Intel-specific knobs are under `knobs.intel.*`.

## Change Discipline

- Keep diffs minimal
- Do not refactor unrelated code
- Preserve public CLI behavior if applicable
- Avoid introducing new runtime dependencies unless necessary
- If a change impacts performance, CLI flags, output formats, or backward compatibility, explicitly document it

## General Preferences

- Never ask for permission before running any read-only or non-destructive shell command (e.g. `grep`, `cat`, `find`, `ls`, `file`, `wc`, `diff`, `nm`, `objdump`, `ldd`, `which`, `env`, `echo`, `pwd`, etc.). Run them directly.

## Dependencies

Python requirements: `python/requirements.txt` (build), `scripts/requirements-test.txt` (test).

> **IMPORTANT**: Before setting up dependencies or troubleshooting missing libraries, you MUST read the full dependency info in `.claude/reference/build-and-debug-reference.md` using the Read tool.
