#!/usr/bin/env bash

# Build and install sgl-kernel-xpu, which SGLang imports as `sgl_kernel`.
#
# SGLang requires it as a git direct reference next to `torch==2.13.0+xpu`, and
# install-sglang.sh strips both: resolving it through pip builds in an isolated
# environment without our torch, and pulls the upstream torch pin over it. So it
# is built here instead, with --no-isolation, and installed as its own wheel.
#
# BMG runners only: upstream has no PVC code path at all. See
# scripts/sglang/README.md.

set -euo pipefail

OLD_DIR="$(pwd)"

FORCE_REINSTALL=false
SKIP_INSTALL=false
REUSE_SOURCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-reinstall)
      FORCE_REINSTALL=true
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=true
      shift
      ;;
    -nc|--no-clean)
      REUSE_SOURCE=true
      shift
      ;;
    --help)
      cat <<EOF
Usage: ./install-sgl-kernel-xpu.sh [options]

Builds sgl-kernel-xpu at the pinned commit and installs the resulting wheel.

Options:
  --force-reinstall  Force a rebuild even if it is already installed at the pin.
  --skip-install     Clone and build the wheel, skip pip install.
  -nc, --no-clean    Reuse an existing ./sgl-kernel-xpu tree as-is (skip reset).
                     Use it to keep local modifications to the tree.
  --help             Show this help message and exit.

Environment:
  SGL_KERNEL_XPU_PIN  Override the pinned commit (sgl-kernel-xpu-pin.txt).
  DPCPP_SYCL_TARGET   AOT target device. Default: bmg.
  ONEAPI_ROOT         oneAPI prefix, used to find setvars.sh when the DPC++
                      compiler is not already on PATH.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1."
      exit 1
      ;;
  esac
done

SGLANG_SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SGLANG_SCRIPTS_DIR/../.." && pwd)"

# Provides the `pip` wrapper (pip or `uv pip`).
source "$ROOT/scripts/pip-utils.sh"

cd "$ROOT"

KERNEL_DIR="$ROOT/sgl-kernel-xpu"
# The wheel version does not encode the commit, so it is stamped alongside.
COMMIT_STAMP="$KERNEL_DIR/dist/.built-commit"
# PyPI's `sgl-kernel` is the unrelated CUDA package.
DIST_NAME="sglang-kernel-xpu"

if [ -z "${SGL_KERNEL_XPU_PIN:-}" ]; then
  SGL_KERNEL_XPU_PIN="$(<"$SGLANG_SCRIPTS_DIR/sgl-kernel-xpu-pin.txt")"
fi

# Passed through to CMake, which rejects unsupported targets itself.
SYCL_TARGET="${DPCPP_SYCL_TARGET:-bmg}"

echo "**** sgl-kernel-xpu pin: $SGL_KERNEL_XPU_PIN ****"
echo "**** AOT target: $SYCL_TARGET ****"

############################################################################
# Toolchain

setup_oneapi() {
  if command -v icpx >/dev/null 2>&1; then
    echo "**** Using icpx already on PATH: $(command -v icpx) ****"
    return
  fi

  local setvars="${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh"
  if [ ! -f "$setvars" ]; then
    echo "ERROR: DPC++ (icpx) is not on PATH and $setvars does not exist." >&2
    exit 1
  fi
  echo "**** Sourcing $setvars ****"
  # setvars.sh is not written for `set -u`.
  set +u
  # shellcheck disable=SC1090
  source "$setvars" >/dev/null
  set -u

  command -v icpx >/dev/null 2>&1 || {
    echo "ERROR: icpx still not on PATH after sourcing $setvars." >&2
    exit 1
  }
}

############################################################################
# Source preparation

clone_kernels() {
  rm -rf "$KERNEL_DIR"
  git clone https://github.com/sgl-project/sgl-kernel-xpu.git "$KERNEL_DIR"
  git -C "$KERNEL_DIR" checkout "$SGL_KERNEL_XPU_PIN"
}

# True when $KERNEL_DIR is itself a git work tree. Without this check a leftover
# $KERNEL_DIR/.git makes git walk up and resolve to *this* repository, so a
# reset/clean could operate on the Triton checkout instead.
is_kernel_repo() {
  [ -e "$KERNEL_DIR/.git" ] || return 1
  local top dir
  top="$(git -C "$KERNEL_DIR" rev-parse --show-toplevel 2>/dev/null)" || return 1
  dir="$(cd "$KERNEL_DIR" && pwd -P)" || return 1
  [ "$top" = "$dir" ]
}

prepare_source() {
  if [ "$REUSE_SOURCE" = true ] && { [ -e "$KERNEL_DIR" ] || [ -L "$KERNEL_DIR" ]; }; then
    if ! is_kernel_repo; then
      echo "ERROR: --no-clean requested, but $KERNEL_DIR is not a valid repository." >&2
      return 1
    fi
    echo "**** --no-clean: reusing existing $KERNEL_DIR as-is. ****"
    echo "sgl-kernel-xpu commit: '$(git -C "$KERNEL_DIR" rev-parse HEAD)'"
    return
  fi

  if is_kernel_repo; then
    if ! git -C "$KERNEL_DIR" rev-parse --verify --quiet "${SGL_KERNEL_XPU_PIN}^{commit}" >/dev/null; then
      git -C "$KERNEL_DIR" fetch --tags origin || true
    fi

    # build/ holds the CUTLASS-SYCL FetchContent cache; a full clean would
    # re-download and recompile everything on every run.
    if git -C "$KERNEL_DIR" reset --hard "$SGL_KERNEL_XPU_PIN" \
      && git -C "$KERNEL_DIR" clean -xffd -e build -e dist; then
      echo "**** Reset existing $KERNEL_DIR to the pinned commit. ****"
    else
      echo "**** Could not reset $KERNEL_DIR to the pinned commit, re-cloning. ****"
      clone_kernels
    fi
  else
    clone_kernels
  fi

  echo "sgl-kernel-xpu commit: '$(git -C "$KERNEL_DIR" rev-parse HEAD)'"
}

############################################################################
# Check whether the current installation already matches the pin

resolve_pin() {
  git -C "$KERNEL_DIR" rev-parse --verify --quiet "${SGL_KERNEL_XPU_PIN}^{commit}" 2>/dev/null
}

installed_at_pin() {
  pip show "$DIST_NAME" >/dev/null 2>&1 || return 1
  is_kernel_repo || return 1
  [ -f "$COMMIT_STAMP" ] || return 1

  local want head stamped
  want="$(resolve_pin)" || return 1
  [ -n "$want" ] || return 1
  head="$(git -C "$KERNEL_DIR" rev-parse HEAD)" || return 1
  [ "$head" = "$want" ] || return 1

  stamped="$(<"$COMMIT_STAMP")"
  [ "$stamped" = "$want $SYCL_TARGET" ] || return 1
}

# Build backend only. sgl-kernel-xpu declares `dependencies = []`, so nothing
# here can replace torch or Triton.
install_build_dependencies() {
  pip install "scikit-build-core>=0.10" wheel build
}

if [ "$FORCE_REINSTALL" = true ]; then
  if pip show "$DIST_NAME" >/dev/null 2>&1; then
    echo "**** --force-reinstall: uninstalling existing $DIST_NAME. ****"
    pip uninstall -y "$DIST_NAME"
  fi
  rm -f "$COMMIT_STAMP"
elif [ "$REUSE_SOURCE" = true ]; then
  echo "**** --no-clean: not checking the installed commit against the pin. ****"
elif installed_at_pin; then
  echo "**** $DIST_NAME is already installed at the pinned commit, skipping. ****"
  echo "**** Use --force-reinstall to force a rebuild. ****"
  cd "$OLD_DIR"
  exit 0
elif pip show "$DIST_NAME" >/dev/null 2>&1; then
  echo "**** Installed $DIST_NAME does not match the pin or the AOT target. ****"
  echo "**** Re-preparing $KERNEL_DIR and rebuilding.                       ****"
fi

############################################################################
# Build

prepare_source
setup_oneapi
install_build_dependencies

# A stale wheel would make the `dist/*.whl` glob below ambiguous.
rm -f "$KERNEL_DIR"/dist/*.whl

echo "**** Building the sgl-kernel-xpu wheel (slow: AOT SYCL kernels). ****"
DPCPP_SYCL_TARGET="$SYCL_TARGET" \
  CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}" \
  python -m build --wheel --no-isolation "$KERNEL_DIR"

printf '%s %s\n' "$(git -C "$KERNEL_DIR" rev-parse HEAD)" "$SYCL_TARGET" > "$COMMIT_STAMP"

############################################################################
# Install

if [ "$SKIP_INSTALL" = true ]; then
  echo "**** --skip-install: skipping pip install. ****"
  cd "$OLD_DIR"
  exit 0
fi

pip install "$KERNEL_DIR"/dist/*.whl

echo "**** sgl-kernel-xpu installed successfully ****"
python -c 'import sgl_kernel; print("sgl_kernel:", sgl_kernel.__file__)'

cd "$OLD_DIR"
