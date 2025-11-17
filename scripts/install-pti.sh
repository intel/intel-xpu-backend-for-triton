#!/usr/bin/env bash

set -euo pipefail

# Select what to build.
BUILD_LEVEL_ZERO=false
for arg in "$@"; do
  case $arg in
    --build-level-zero)
      BUILD_LEVEL_ZERO=true
      shift
      ;;
    --help)
      echo "Example usage: ./install-pti.sh [--build-level-zero]"
      exit 1
      ;;
    *)
      echo "Unknown argument: $arg."
      exit 1
      ;;
  esac
done


# Configure, build and install PyTorch from source.

# intel-xpu-backend-for-triton project root
ROOT=$(cd "$(dirname "$0")/.." && pwd)

SCRIPTS_DIR=$ROOT/scripts
PTI_PROJ=$ROOT/.scripts_cache/pti
LEVEL_ZERO_PROJ=$ROOT/.scripts_cache/level_zero_for_pti
BASE=$(dirname "$PTI_PROJ")

echo "**** BASE is set to $BASE ****"
echo "**** PTI_PROJ is set to $PTI_PROJ ****"
mkdir -p $BASE

function build_level_zero {
  rm -rf "$LEVEL_ZERO_PROJ"
  mkdir -p "$LEVEL_ZERO_PROJ"
  cd "$LEVEL_ZERO_PROJ"
  LEVEL_ZERO_VERSION=1.24.2
  LEVEL_ZERO_SHA256=b77e6e28623134ee4e99e2321c127b554bdd5bfa3e80064922eba293041c6c52

  if [[ $OSTYPE = msys ]]; then
    curl "https://github.com/oneapi-src/level-zero/archive/refs/tags/v${LEVEL_ZERO_VERSION}.tar.gz"
  else
    wget "https://github.com/oneapi-src/level-zero/archive/refs/tags/v${LEVEL_ZERO_VERSION}.tar.gz"
  fi
  echo "${LEVEL_ZERO_SHA256}  v${LEVEL_ZERO_VERSION}.tar.gz" > "v${LEVEL_ZERO_VERSION}.tar.gz.sha256"
  sha256sum -c "v${LEVEL_ZERO_VERSION}.tar.gz.sha256"
  tar -xf "v${LEVEL_ZERO_VERSION}.tar.gz"
  cd "level-zero-${LEVEL_ZERO_VERSION}"
  echo "${LEVEL_ZERO_VERSION}" | awk -F. '{print $3}' > VERSION_PATCH
  mkdir build
  cd build
  L0_INSTALL_PATH="$LEVEL_ZERO_PROJ/level-zero-${LEVEL_ZERO_VERSION}/install"
  cmake .. -DCMAKE_INSTALL_PREFIX="$L0_INSTALL_PATH" -DCMAKE_BUILD_TYPE=Release
  cmake --build . --config Release --parallel "$(nproc)"
  cmake --build . --config Release --target install
  export LEVELZERO_INCLUDE_DIR="$L0_INSTALL_PATH/include"
  export LEVELZERO_LIBRARY="$L0_INSTALL_PATH/lib/libze_loader.so"
}

function build_pti {
  rm -rf "$PTI_PROJ"
  mkdir -p "$PTI_PROJ"

  echo "****** Building $PTI_PROJ ******"
  cd "$PTI_PROJ"
  cp "$SCRIPTS_DIR"/build_pti_data/* .
  pip install uv

  export PTI_PINNED_COMMIT="$(<$ROOT/.github/pins/pti.txt)"

  uv version 0.14.0.dev1
  uv build
}

function install_pti {
  echo "****** Installing PTI ******"
  cd "$PTI_PROJ"
  pip install dist/*.whl
}

if [ "$BUILD_LEVEL_ZERO" = true ]; then
  build_level_zero
fi

build_pti
install_pti
