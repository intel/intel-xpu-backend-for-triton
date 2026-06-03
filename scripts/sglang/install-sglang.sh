#!/usr/bin/env bash

set -euo pipefail

OLD_DIR="$(pwd)"
FORCE_REINSTALL=false
SKIP_INSTALL=false
VENV=false

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
    --venv)
      VENV=true
      shift
      ;;
    --help)
      cat <<EOF
Usage: ./install-sglang.sh [options]

Options:
  --force-reinstall  Force reinstall even if sglang is already installed.
  --skip-install     Clone and patch only, skip python -m pip install.
  --venv             Activate .venv/ before installation.
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

if [ "$VENV" = true ]; then
  if [[ $OSTYPE = msys || $OSTYPE = cygwin ]]; then
    source .venv/Scripts/activate
  else
    source .venv/bin/activate
  fi
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SGLANG_PROJ="$ROOT/sglang"
SGLANG_PIN="${SGLANG_PIN:-$(<"$ROOT/benchmarks/third_party/sglang/sglang-pin.txt")}"

echo "**** SGLang pin: $SGLANG_PIN ****"

if python -m pip show sglang >/dev/null 2>&1; then
  if [ "$FORCE_REINSTALL" = false ]; then
    echo "**** sglang is already installed, skipping. ****"
    echo "**** Use --force-reinstall to force reinstallation. ****"
    exit 0
  fi
  python -m pip uninstall -y sglang
fi

if [ "$FORCE_REINSTALL" = true ] && [ -d "$SGLANG_PROJ" ]; then
  rm -rf "$SGLANG_PROJ"
fi

if [ ! -d "$SGLANG_PROJ" ]; then
  git clone https://github.com/sgl-project/sglang.git "$SGLANG_PROJ"
  cd "$SGLANG_PROJ"
  git checkout "$SGLANG_PIN"
else
  cd "$SGLANG_PROJ"
fi

for patch in sglang-test-fix.patch sglang-bench-fix.patch; do
  PATCH_PATH="$ROOT/benchmarks/third_party/sglang/$patch"
  if git apply --check "$PATCH_PATH" 2>/dev/null; then
    git apply "$PATCH_PATH"
    echo "**** Applied $patch ****"
  else
    echo "**** $patch already applied or conflicts, skipping. ****"
  fi
done

# SGLang assumes this target platform file name.
cp python/pyproject_xpu.toml python/pyproject.toml
# Keep preinstalled nightly torch/triton; remove deps that force replacement.
sed -i '/pytorch\|torch\|sgl-kernel\|timm/d' python/pyproject.toml

if [ "$SKIP_INSTALL" = true ]; then
  echo "**** --skip-install: skipping python -m pip install. ****"
  cd "$OLD_DIR"
  exit 0
fi

python -m pip install -e "$SGLANG_PROJ/python"

cd "$OLD_DIR"
