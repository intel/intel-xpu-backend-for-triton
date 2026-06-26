#!/usr/bin/env bash
# Run the paged_kv_load reproducer through the Nvidia compile pipeline against
# a Blackwell target (default sm100). Works headless — no GPU is required, and
# the host doesn't need to be Nvidia. Internally we use `triton.compile` with
# an explicit `GPUTarget(backend="cuda", arch=...)`, which runs the full TTIR
# -> TTGIR -> LLIR -> PTX -> cubin pipeline without touching the active driver.
#
# Usage:
#   tma-update/run_blackwell.sh                       # compile for sm100, no IR dump
#   tma-update/run_blackwell.sh --dump-mlir           # stream every-pass MLIR dump to stderr
#   tma-update/run_blackwell.sh --save-ir <DIR>       # save .ttir/.ttgir/.llir/.ptx/.cubin under <DIR>
#   tma-update/run_blackwell.sh --arch sm90           # override target (default sm100)
#   tma-update/run_blackwell.sh --launch              # JIT and launch on the local device (needs a real GPU)
#
# Combine flags freely:
#   tma-update/run_blackwell.sh --dump-mlir --save-ir /tmp/tma-ir --arch sm90

set -euo pipefail

ARCH="sm100"
DUMP_MLIR=0
SAVE_DIR=""
LAUNCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dump-mlir)
      DUMP_MLIR=1
      shift
      ;;
    --save-ir)
      SAVE_DIR="${2:?--save-ir requires a directory argument}"
      shift 2
      ;;
    --arch)
      ARCH="${2:?--arch requires a value (e.g. sm90, sm100)}"
      shift 2
      ;;
    --launch)
      LAUNCH=1
      shift
      ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY="${SCRIPT_DIR}/paged_kv_load.py"

# Tell paged_kv_load.py which GPUTarget to compile against.
export TMA_UPDATE_ARCH="${ARCH}"

# Default: compile-only via triton.compile(ASTSource, target=GPUTarget(...)).
# No GPU is required, and we don't try to launch anything.
if (( LAUNCH )); then
  unset TMA_UPDATE_COMPILE_ONLY || true
else
  export TMA_UPDATE_COMPILE_ONLY=1
fi

EXTRA_ENV=()
if (( DUMP_MLIR )); then
  EXTRA_ENV+=("MLIR_ENABLE_DUMP=1")
fi

if [[ -n "${SAVE_DIR}" ]]; then
  mkdir -p "${SAVE_DIR}"
  # paged_kv_load.py writes <kernel>.source / <kernel>.ttir / <kernel>.ttgir
  # into this directory after each stage. We stop after TTGIR by default
  # because the LLIR / PTX stages require Nvidia-only passes that may not be
  # compiled into the Intel fork's libtriton.
  EXTRA_ENV+=("TMA_UPDATE_OUT_DIR=${SAVE_DIR}")
fi

# Pick the python that has the in-tree triton on its path. Override with
# TMA_UPDATE_PYTHON=... if you want a specific interpreter.
PYTHON_BIN="${TMA_UPDATE_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in \
      "${CONDA_PREFIX:-}/bin/python" \
      python3 \
      python; do
    [[ -z "${candidate}" ]] && continue
    if "${candidate}" -c "import triton" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[run_blackwell] could not locate a python with 'triton' importable; set TMA_UPDATE_PYTHON" >&2
  exit 3
fi

echo "[run_blackwell] arch=${ARCH} dump_mlir=${DUMP_MLIR} save_ir=${SAVE_DIR:-<none>} launch=${LAUNCH} python=${PYTHON_BIN}" >&2
env "${EXTRA_ENV[@]}" "${PYTHON_BIN}" "${PY}"

if [[ -n "${SAVE_DIR}" ]]; then
  echo "" >&2
  echo "[run_blackwell] saved intermediates under ${SAVE_DIR}:" >&2
  find "${SAVE_DIR}" -maxdepth 1 -type f \
       \( -name '*.source' -o -name '*.ttir' -o -name '*.ttgir' \) \
       2>/dev/null | sort >&2 || true
fi
