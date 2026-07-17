#!/usr/bin/env bash

# Install SGLang for XPU benchmarking/testing.
#
# Clones SGLang, checks out the pinned commit (sglang-pin.txt), applies the local
# XPU patches (sglang-test-fix.patch, sglang-bench-fix.patch) and installs it in
# editable mode. Torch/Triton dependencies are stripped from SGLang's
# requirements so the repository's own (latest) torch and triton are kept.

set -euo pipefail

OLD_DIR="$(pwd)"

FORCE_REINSTALL=false
SKIP_INSTALL=false

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
    --help)
      cat <<EOF
Usage: ./install-sglang.sh [options]

Options:
  --force-reinstall  Force reinstallation even if SGLang is already installed.
  --skip-install     Clone and patch only, skip pip install.
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

# Use SGLANG_PIN environment variable if set, otherwise read from file.
if [ -z "${SGLANG_PIN:-}" ]; then
  SGLANG_PIN="$(<"$SGLANG_SCRIPTS_DIR/sglang-pin.txt")"
fi

echo "**** SGLang pin: $SGLANG_PIN ****"

############################################################################
# Check if already installed

if pip show sglang >/dev/null 2>&1; then
  if [ "$FORCE_REINSTALL" = false ]; then
    echo "**** SGLang is already installed, skipping. ****"
    echo "**** Use --force-reinstall to force reinstallation. ****"
    echo "**** To get a clean install: rm -rf ./sglang && pip uninstall -y sglang ****"
    exit 0
  fi
  echo "**** --force-reinstall: uninstalling existing SGLang. ****"
  pip uninstall -y sglang
  rm -rf ./sglang
fi

############################################################################
# Clone, checkout pinned commit and patch for XPU

if [ -d "./sglang" ]; then
  if [ "$FORCE_REINSTALL" = true ]; then
    echo "**** Removing existing ./sglang ****"
    rm -rf ./sglang
  else
    echo "**** Reusing existing ./sglang directory. ****"
  fi
fi

if [ ! -d "./sglang" ]; then
  git clone https://github.com/sgl-project/sglang.git
  cd sglang
  git checkout "$SGLANG_PIN"
  git apply "$SGLANG_SCRIPTS_DIR/sglang-test-fix.patch"
  git apply "$SGLANG_SCRIPTS_DIR/sglang-bench-fix.patch"

  # That's how sglang assumes we'll pick out platform for now.
  cp python/pyproject_xpu.toml python/pyproject.toml
  # Remove all torch libraries from requirements to avoid reinstalling triton & torch.
  # Remove sgl-kernel due to a bug in the current environment (newer torch); we don't use it here.
  # Remove timm because it depends on torchvision, which depends on a pinned torch.
  sed -i '/pytorch\|torch\|sgl-kernel\|timm/d' python/pyproject.toml
  cat python/pyproject.toml
  cd ..
fi

############################################################################
# Install

if [ "$SKIP_INSTALL" = true ]; then
  echo "**** --skip-install: skipping pip install. ****"
  cd "$OLD_DIR"
  exit 0
fi

pip install -e "./sglang/python"

echo "**** SGLang installed successfully ****"

cd "$OLD_DIR"
