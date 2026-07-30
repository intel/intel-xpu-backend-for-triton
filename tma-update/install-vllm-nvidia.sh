#!/usr/bin/env bash
#
# install-vllm-nvidia.sh — install vLLM on an NVIDIA box for the unified-attention
# benchmark, without the XPU machinery.
#
# scripts/vllm/install-vllm.sh hardcodes VLLM_TARGET_DEVICE=xpu and pulls
# vllm-xpu-kernels; it will NOT work on CUDA. This script codifies the by-hand
# recipe from tma-update/RUN.md ("NVIDIA vLLM install"): clone vLLM at the pinned
# commit, strip its torch/triton pins so they can't clobber your source build,
# and install only the Python module tree — no C++/CUDA kernels, because the
# benchmark imports one @triton.jit kernel (unified_attention), not vLLM's ops.
#
# ORDERING RULE (same as RUN.md §1): in the target env, install torch FIRST and
# your editable Triton LAST, BEFORE running this. This script installs neither —
# it strips them from vLLM's requirements so pip cannot drag the triton==3.2.0
# wheel in over your build, and it verifies at the end that Triton is still your
# source build, failing loudly if a pip step shadowed it.
#
# Usage:
#   tma-update/install-vllm-nvidia.sh [options]
#
#   --vllm-dir DIR   where to clone vLLM        (default: <repo>/vllm)
#   --pin COMMIT     vLLM commit to check out   (default: scripts/vllm/vllm-pin.txt)
#   --python BIN     target interpreter; must already have torch + your editable
#                    triton (default: auto-detect, or set TMA_UPDATE_PYTHON)
#   --precompiled    install with VLLM_USE_PRECOMPILED=1 instead of
#                    VLLM_TARGET_DEVICE=empty (use if your pin rejects 'empty')
#   --apply-fix      apply the device-agnostic triton_unified_attention.py JIT
#                    annotation fix from scripts/vllm/vllm-fix.patch — needed if
#                    --smoke-test dies with a TypeError about NoneType and an int
#   --force          re-clone even if --vllm-dir already exists
#   -h, --help       show this message and exit
#
# We deliberately do NOT run scripts/vllm/vllm_xpu_patch.py — it rewrites
# CUDA->XPU in vLLM's test/layer dirs and would break the CUDA path.
#
# After it finishes, verify (the second check is the one people skip and regret):
#   tma-update/run_unified_attention.sh --probe
#   python3 -c "import triton,os; print(os.path.realpath(triton.__file__))"

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
PIN_FILE="${REPO_ROOT}/scripts/vllm/vllm-pin.txt"
FIX_PATCH="${REPO_ROOT}/scripts/vllm/vllm-fix.patch"

VLLM_DIR="${REPO_ROOT}/vllm"
PIN=""
PYTHON_BIN="${TMA_UPDATE_PYTHON:-}"
PRECOMPILED=false
APPLY_FIX=false
FORCE=false

die()  { echo "[install-vllm-nvidia] error: $*" >&2; exit 2; }
info() { echo "[install-vllm-nvidia] $*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-dir)    VLLM_DIR="${2:?}";   shift 2 ;;
    --pin)         PIN="${2:?}";        shift 2 ;;
    --python)      PYTHON_BIN="${2:?}"; shift 2 ;;
    --precompiled) PRECOMPILED=true;    shift ;;
    --apply-fix)   APPLY_FIX=true;      shift ;;
    --force)       FORCE=true;          shift ;;
    -h|--help)     sed -n '2,39p' "$0"; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

# --- resolve the pin ---------------------------------------------------------
if [[ -z "${PIN}" ]]; then
  [[ -f "${PIN_FILE}" ]] || die "pin file not found: ${PIN_FILE} (pass --pin COMMIT)"
  PIN="$(<"${PIN_FILE}")"
fi
[[ -n "${PIN}" ]] || die "empty vLLM pin"

# --- locate a python with torch AND triton already importable ----------------
# This script installs neither; it installs INTO the env that already has your
# source-built triton (torch first, triton last, per RUN.md §1).
if [[ -z "${PYTHON_BIN}" ]]; then
  for cand in "${CONDA_PREFIX:-}/bin/python" python3 python; do
    [[ -z "${cand}" ]] && continue
    if "${cand}" -c "import torch, triton" >/dev/null 2>&1; then PYTHON_BIN="${cand}"; break; fi
  done
fi
[[ -n "${PYTHON_BIN}" ]] || die "no python with both 'torch' and 'triton' importable.
  Do RUN.md §1 first (install torch, THEN editable triton), or pass --python BIN."

info "python=${PYTHON_BIN}"
info "vllm-dir=${VLLM_DIR}"
info "pin=${PIN}"

# --- preflight: torch present, triton is the source build (not torch's wheel) -
TMA_REPO_ROOT="${REPO_ROOT}" "${PYTHON_BIN}" - <<'PY' || die "preflight failed — see above"
import os, sys, triton
tp = os.path.realpath(triton.__file__)
print(f"[preflight] triton {triton.__version__} @ {tp}", file=sys.stderr)
if not hasattr(triton, "set_allocator"):
    print("[preflight] FAIL: triton lacks set_allocator — this is torch's bundled "
          "wheel, not your source build. Reinstall with `pip install -e .` from "
          "your triton checkout (torch FIRST, editable triton LAST).", file=sys.stderr)
    sys.exit(1)
repo = os.environ.get("TMA_REPO_ROOT", "")
if repo and not tp.startswith(repo):
    print(f"[preflight] WARN: triton at {tp} is not under this repo ({repo}). "
          "If that is intentional, ignore; otherwise re-run `pip install -e .` here.",
          file=sys.stderr)
PY

PIP=("${PYTHON_BIN}" -m pip)

# --- clone / reset to the pinned commit --------------------------------------
if [[ "${FORCE}" == true ]]; then
  info "--force: removing ${VLLM_DIR}"
  rm -rf "${VLLM_DIR}"
fi

if [[ ! -d "${VLLM_DIR}/.git" ]]; then
  rm -rf "${VLLM_DIR}"
  info "cloning vLLM into ${VLLM_DIR}"
  git clone https://github.com/vllm-project/vllm.git "${VLLM_DIR}"
else
  info "reusing existing checkout at ${VLLM_DIR} (discarding local edits)"
  git -C "${VLLM_DIR}" reset --hard -q HEAD || true
  git -C "${VLLM_DIR}" clean -qfd || true
fi

# Check out the pin; fetch it explicitly if the initial clone didn't include it.
if ! git -C "${VLLM_DIR}" checkout -q "${PIN}" 2>/dev/null; then
  info "pin ${PIN} not present locally — fetching it"
  git -C "${VLLM_DIR}" fetch --depth 1 origin "${PIN}"
  git -C "${VLLM_DIR}" checkout -q FETCH_HEAD
fi
info "vLLM at $(git -C "${VLLM_DIR}" rev-parse --short HEAD)"

# --- strip torch/triton/xformers pins so pip can't clobber the source build --
# Every pip step below can otherwise silently drag in the triton==3.2.0 wheel.
for req in requirements/common.txt requirements/cuda.txt; do
  [[ -f "${VLLM_DIR}/${req}" ]] || continue
  sed -i -E '/^(torch|triton|xformers)/d; /^--extra-index-url.*download\.pytorch\.org/d' \
    "${VLLM_DIR}/${req}"
  info "stripped torch/triton/xformers from ${req}"
done

# cuda.txt pulls common.txt via -r, so installing it covers both.
if [[ -f "${VLLM_DIR}/requirements/cuda.txt" ]]; then
  "${PIP[@]}" install -r "${VLLM_DIR}/requirements/cuda.txt"
elif [[ -f "${VLLM_DIR}/requirements/common.txt" ]]; then
  "${PIP[@]}" install -r "${VLLM_DIR}/requirements/common.txt"
else
  info "WARN: no requirements/{cuda,common}.txt found; skipping requirements install"
fi

# --- install vLLM's Python tree, no C++/CUDA kernels -------------------------
# 'empty' skips building vLLM's ops (we import a Triton kernel, not vLLM's ops).
# --no-deps stops vLLM's own torch/triton pins from being honoured.
if [[ "${PRECOMPILED}" == true ]]; then
  BUILD_ENV=(VLLM_USE_PRECOMPILED=1)
  info "installing vLLM (VLLM_USE_PRECOMPILED=1, --no-deps, editable)"
else
  BUILD_ENV=(VLLM_TARGET_DEVICE=empty)
  info "installing vLLM (VLLM_TARGET_DEVICE=empty, --no-deps, editable)"
fi
env "${BUILD_ENV[@]}" "${PIP[@]}" install --no-deps --no-build-isolation -e "${VLLM_DIR}"

# --- optional: the device-agnostic JIT-annotation fix ------------------------
if [[ "${APPLY_FIX}" == true ]]; then
  [[ -f "${FIX_PATCH}" ]] || die "fix patch not found: ${FIX_PATCH}"
  # Only the triton_unified_attention.py hunk is relevant on CUDA (the other
  # hunks are XPU/test fixes); --include applies just that file's changes.
  if git -C "${VLLM_DIR}" apply --include='*triton_unified_attention.py' --check "${FIX_PATCH}" 2>/dev/null; then
    git -C "${VLLM_DIR}" apply --include='*triton_unified_attention.py' "${FIX_PATCH}"
    info "applied triton_unified_attention.py JIT-annotation fix"
  else
    info "WARN: could not apply the triton_unified_attention.py hunk (already "
    info "      patched, or the pin diverged) — continuing without it."
  fi
fi

# --- verify: kernel imports, triton is still the source build ----------------
info "verifying install"
TMA_REPO_ROOT="${REPO_ROOT}" "${PYTHON_BIN}" - <<'PY' || die "verify failed — see above"
import os, sys, triton
tp = os.path.realpath(triton.__file__)
print(f"[verify] triton {triton.__version__} @ {tp}", file=sys.stderr)
if not hasattr(triton, "set_allocator"):
    print("[verify] FAIL: triton lost set_allocator — a pip step shadowed your "
          "source build with torch's wheel. Re-run `pip install -e .` (triton LAST).",
          file=sys.stderr)
    sys.exit(1)
repo = os.environ.get("TMA_REPO_ROOT", "")
if repo and not tp.startswith(repo):
    print(f"[verify] WARN: triton at {tp} is not under this repo ({repo}).",
          file=sys.stderr)

# Import the kernel the benchmark actually uses; --no-deps means a missing
# transitive import surfaces here, so make the fix actionable.
last = None
for mod in ("vllm.attention.ops.triton_unified_attention",
            "vllm.v1.attention.ops.triton_unified_attention"):
    try:
        m = __import__(mod, fromlist=["unified_attention"])
        print(f"[verify] OK: unified_attention imports from {mod}", file=sys.stderr)
        break
    except ImportError as e:
        last = e
else:
    print(f"[verify] FAIL: cannot import unified_attention ({last}). With --no-deps "
          "a missing dep shows up here — install it with `pip install --no-deps <pkg>` "
          "and retry.", file=sys.stderr)
    sys.exit(1)
PY

info "done. Next: tma-update/run_unified_attention.sh --probe"
