# CLAUDE.md

Intel XPU Backend for Triton — out-of-tree Intel GPU backend for the Triton compiler.

## Quick Start

**Build**: `scripts/compile-triton.sh` (incremental) or `make dev-install` (pip-based)
**Pre-commit**: Install via `pip install pre-commit`. Run `python3 -m pre_commit run --show-diff-on-failure --color=always --all-files --verbose`

> **IMPORTANT**: Before build/debug/dependency/project-structure work, read `.claude/reference/build-and-debug-reference.md`.

## Do not guess — read the reference first

Four files under `.claude/reference/` hold the authoritative tables. Before answering a question or
writing code in these areas, read the matching file; never guess values, names, or signatures:

- `hardware-reference.md` — Xe architecture and generation specs, GRF sizes and modes, per-arch
  subgroup sizes, target arch strings, device-capability→module-attribute mappings, DPAS hardware
  constants and engine types, cache sizes.
- `operations-reference.md` — TritonGEN op semantics: memory-space constants, DPAS/BDPAS precision
  and element-type rules, verifier constraints, register alignment, scale types, 2D block I/O
  parameters and tile limits, sub-group and predicated I/O, format conversion, enum values.
- `build-and-debug-reference.md` — encoding attribute parameters (DpasEncodingAttr,
  Subgroup2DBlockEncodingAttr, DotOperandEncodingAttr), opsPerChannel, type-packing rules,
  cache-control decoration mappings.
- `passes-and-testing-reference.md` — pass inventory, CLI flags and prefixes, TableGen locations,
  namespace aliases, pattern base-class signatures, utility APIs, pass-to-capability gating
  attributes, test directory layout, lit env vars, FileCheck directives, test module attributes,
  `is_xpu_*` architecture detection, numerical tolerances, pytest fixtures, and test-runner and
  Makefile targets.

The detail behind each topic lives in the path-scoped rules under `.claude/rules/`, which load when
you open a matching file.

## Architecture

Intel-specific code lives in `third_party/intel/` and is symlinked into the Python module tree:
- `python/triton/backends/intel/` → `third_party/intel/backend/`
- `python/triton/language/extra/intel/` → `third_party/intel/language/intel/`

Runtime stack: SYCL + Level Zero + IGC. The backend registers as the `xpu` target.

## Change Discipline

- Keep diffs minimal
- Do not refactor unrelated code
- Preserve public CLI behavior if applicable
- Avoid introducing new runtime dependencies unless necessary
- If a change impacts performance, CLI flags, output formats, or backward compatibility, explicitly document it
