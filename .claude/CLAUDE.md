# CLAUDE.md

Intel XPU Backend for Triton — out-of-tree Intel GPU backend for the Triton compiler.

## Quick Start

**Build**: `scripts/compile-triton.sh` (incremental) or `make dev-install` (pip-based)
**Pre-commit**: Install via `pip install pre-commit`. Run `python3 -m pre_commit run --show-diff-on-failure --color=always --all-files --verbose`

→ See `.claude/reference/build-and-debug-reference.md` for: build tips, IR debugging, dependencies

## Architecture

Intel-specific code lives in `third_party/intel/` and is symlinked into the Python module tree:
- `python/triton/backends/intel/` → `third_party/intel/backend/`
- `python/triton/language/extra/intel/` → `third_party/intel/language/intel/`

Runtime stack: SYCL + Level Zero + IGC. The backend registers as the `xpu` target.

→ See `.claude/reference/build-and-debug-reference.md` for: directory structure, runtime stack details

## Change Discipline

- Keep diffs minimal
- Do not refactor unrelated code
- Preserve public CLI behavior if applicable
- Avoid introducing new runtime dependencies unless necessary
- If a change impacts performance, CLI flags, output formats, or backward compatibility, explicitly document it
