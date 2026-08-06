#!/usr/bin/env bash

# Install SGLang for XPU benchmarking/testing.
#
# Clones SGLang, checks out the pinned commit (sglang-pin.txt), applies the local
# XPU patches (sglang-test-fix.patch) and installs it in
# editable mode. Torch/Triton dependencies are stripped from SGLang's
# requirements so the repository's own (latest) torch and triton are kept.
#
# Whenever the source tree is prepared it is first brought back to a pristine
# checkout of the pinned commit, so a run that failed half-way - or a bumped pin -
# cannot leave behind a tree that is then silently reused in a wrong state.

set -euo pipefail

OLD_DIR="$(pwd)"

FORCE_REINSTALL=false
SKIP_INSTALL=false
REUSE_SOURCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-reinstall)
      # Remove an existing checkout/installation and reinstall from scratch.
      FORCE_REINSTALL=true
      shift
      ;;
    --skip-install)
      # Clone and patch only, do not pip install.
      SKIP_INSTALL=true
      shift
      ;;
    -nc|--no-clean)
      # Reuse an existing ./sglang as-is: skip the reset and the patching.
      REUSE_SOURCE=true
      shift
      ;;
    --help)
      cat <<EOF
Usage: ./install-sglang.sh [options]

Options:
  --force-reinstall  Force reinstallation even if SGLang is already installed.
  --skip-install     Clone and patch only, skip pip install.
  -nc, --no-clean    Reuse an existing ./sglang tree as-is (skip reset and patching).
                     Use it to keep local modifications to the tree.
  --help             Show this help message and exit.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1."
      exit 1
      ;;
  esac
done

# intel-xpu-backend-for-triton project root and this script's directory.
SGLANG_SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SGLANG_SCRIPTS_DIR/../.." && pwd)"

# Clone/install into the project root (matches scripts/vllm/install-vllm.sh and
# test-triton.sh's run_sglang_tests, which expect ./sglang there).
cd "$ROOT"

SGLANG_DIR="$ROOT/sglang"
SGLANG_PATCH="$SGLANG_SCRIPTS_DIR/sglang-test-fix.patch"

# Use SGLANG_PIN environment variable if set, otherwise read from file.
if [ -z "${SGLANG_PIN:-}" ]; then
  SGLANG_PIN="$(<"$SGLANG_SCRIPTS_DIR/sglang-pin.txt")"
fi

echo "**** SGLang pin: $SGLANG_PIN ****"

############################################################################
# Source preparation

clone_sglang() {
  rm -rf "$SGLANG_DIR"
  git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
  git -C "$SGLANG_DIR" checkout "$SGLANG_PIN"
}

# Apply the XPU-specific changes on top of a pristine checkout.
patch_sglang() {
  git -C "$SGLANG_DIR" apply "$SGLANG_PATCH"

  # That's how sglang assumes we'll pick out platform for now.
  # NOTE: python/pyproject.toml is tracked upstream, so the reset in prepare_source
  # reverts this overwrite - it has to be redone on every preparation.
  cp "$SGLANG_DIR/python/pyproject_xpu.toml" "$SGLANG_DIR/python/pyproject.toml"
  # Remove all torch libraries from requirements to avoid reinstalling triton & torch.
  # Remove sgl-kernel due to a bug in the current environment (newer torch); we don't use it here.
  # Remove timm because it depends on torchvision, which depends on a pinned torch.
  sed -i '/pytorch\|torch\|sgl-kernel\|timm/d' "$SGLANG_DIR/python/pyproject.toml"
  cat "$SGLANG_DIR/python/pyproject.toml"
}

# True when $SGLANG_DIR is itself a git work tree. Without this check a corrupted
# or leftover $SGLANG_DIR/.git makes git walk up and resolve to *this* repository,
# so a reset/clean could operate on the Triton checkout instead of on SGLang.
is_sglang_repo() {
  # A regular checkout has a .git directory, while a git worktree has a .git file.
  [ -e "$SGLANG_DIR/.git" ] || return 1
  local top dir
  top="$(git -C "$SGLANG_DIR" rev-parse --show-toplevel 2>/dev/null)" || return 1
  dir="$(cd "$SGLANG_DIR" && pwd -P)" || return 1
  [ "$top" = "$dir" ]
}

# Bring ./sglang to a pristine checkout of the pinned commit and patch it.
# Re-clones when an existing tree cannot be reset, e.g. when a previous run failed
# after cloning but before the checkout/patching finished.
prepare_source() {
  # --no-clean must never remove an existing path. Fail closed when it is not a
  # valid standalone repository instead of falling through to clone_sglang's rm.
  if [ "$REUSE_SOURCE" = true ] && { [ -e "$SGLANG_DIR" ] || [ -L "$SGLANG_DIR" ]; }; then
    if ! is_sglang_repo; then
      echo "ERROR: --no-clean requested, but $SGLANG_DIR is not a valid SGLang repository." >&2
      echo "ERROR: Refusing to remove or modify the existing path." >&2
      return 1
    fi
    echo "**** --no-clean: reusing existing $SGLANG_DIR as-is. ****"
    echo "SGLang commit: '$(git -C "$SGLANG_DIR" rev-parse HEAD)'"
    return
  fi

  if is_sglang_repo; then
    # Only fetch when the pinned commit is not available locally yet, so that
    # re-preparing an up-to-date checkout does not need the network.
    if ! git -C "$SGLANG_DIR" rev-parse --verify --quiet "${SGLANG_PIN}^{commit}" >/dev/null; then
      git -C "$SGLANG_DIR" fetch --tags origin || true
    fi

    if git -C "$SGLANG_DIR" reset --hard "$SGLANG_PIN" \
      && git -C "$SGLANG_DIR" clean -xffd; then
      echo "**** Reset existing $SGLANG_DIR to the pinned commit. ****"
    else
      echo "**** Could not reset $SGLANG_DIR to the pinned commit, re-cloning. ****"
      clone_sglang
    fi
  else
    clone_sglang
  fi

  echo "SGLang commit: '$(git -C "$SGLANG_DIR" rev-parse HEAD)'"
  patch_sglang
}

############################################################################
# Check whether the current installation already matches the pin

# Resolve the pin (full or short SHA, tag or branch) to a full SHA via the checkout.
resolve_pin() {
  git -C "$SGLANG_DIR" rev-parse --verify --quiet "${SGLANG_PIN}^{commit}" 2>/dev/null
}

# True only when sglang is installed AND its (editable) source tree is present,
# checked out at the pinned commit and patched.
installed_at_pin() {
  pip show sglang >/dev/null 2>&1 || return 1
  # The install is editable, so a missing tree means a broken installation.
  is_sglang_repo || return 1
  [ -f "$SGLANG_DIR/python/pyproject.toml" ] || return 1

  local want head
  want="$(resolve_pin)" || return 1
  [ -n "$want" ] || return 1
  head="$(git -C "$SGLANG_DIR" rev-parse HEAD)" || return 1
  [ "$head" = "$want" ] || return 1

  # The patch has to be applied already, i.e. reverse-applying it must be possible.
  git -C "$SGLANG_DIR" apply --reverse --check "$SGLANG_PATCH" 2>/dev/null || return 1
}

# Dependencies SGLang needs at runtime but that pyproject_xpu.toml deliberately
# leaves out. Installed before SGLang itself (as install-vllm.sh does), so a failure
# here cannot leave an installed SGLang behind that makes the next run skip ahead.
install_runtime_dependencies() {
  # sglang imports xgrammar unconditionally, but pyproject_xpu.toml leaves it out because
  # it pulls in CUDA torch. Install without deps so our XPU torch and triton survive.
  # Versions match sglang's own pyproject.toml, so an upstream release cannot break us.
  pip install --no-deps xgrammar==0.2.1 apache-tvm-ffi==0.1.11
}

# As in scripts/vllm/install-vllm.sh, --force-reinstall only drops the installed
# package; the state of the source tree is left to prepare_source. That way it
# composes with --no-clean instead of contradicting it.
if [ "$FORCE_REINSTALL" = true ]; then
  if pip show sglang >/dev/null 2>&1; then
    echo "**** --force-reinstall: uninstalling existing SGLang. ****"
    pip uninstall -y sglang
  fi
elif [ "$REUSE_SOURCE" = true ]; then
  echo "**** --no-clean: not checking the installed commit against the pin. ****"
elif installed_at_pin; then
  echo "**** SGLang is already installed at the pinned commit, skipping. ****"
  echo "**** Use --force-reinstall to force reinstallation. ****"
  cd "$OLD_DIR"
  exit 0
elif pip show sglang >/dev/null 2>&1; then
  # Never silently keep a stale installation: after a pin bump, or when the source
  # tree is missing, unpatched or at a different commit, the tree is re-prepared
  # and SGLang is reinstalled below.
  echo "**** Installed SGLang does not match the pin: wrong commit, or a missing or ****"
  echo "**** unpatched source tree. Re-preparing $SGLANG_DIR and reinstalling. ****"
fi

############################################################################
# Clone/reset, checkout pinned commit and patch for XPU

prepare_source

############################################################################
# Install

if [ "$SKIP_INSTALL" = true ]; then
  echo "**** --skip-install: skipping pip install. ****"
  cd "$OLD_DIR"
  exit 0
fi

install_runtime_dependencies
pip install -e "$SGLANG_DIR/python"

echo "**** SGLang installed successfully ****"

if ! python -c 'import torchvision' 2>/dev/null; then
  echo "**** WARNING: torchvision is not installed, any 'import sglang' will fail. ****" >&2
  echo "**** WARNING: it ships with the PyTorch wheel set (scripts/install-pytorch.sh); ****" >&2
  echo "**** WARNING: CI builds it from pytorch/.github/ci_commit_pins/vision.txt.     ****" >&2
fi

cd "$OLD_DIR"
