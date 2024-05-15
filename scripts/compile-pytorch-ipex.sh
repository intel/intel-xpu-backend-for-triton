#!/usr/bin/env bash

set -euo pipefail

# Select what to build.
BUILD_PYTORCH=false
BUILD_IPEX=false
BUILD_PINNED=false
BUILD_FROM_SOURCE=false
CLEAN=false
VENV=false
for arg in "$@"; do
  case $arg in
    --pytorch)
      BUILD_PYTORCH=true
      shift
      ;;
    --ipex)
      BUILD_IPEX=true
      shift
      ;;
    --pinned)
      # Build the pinned commit (from source if --source is used, otherwise download the wheel from github)
      BUILD_PINNED=true
      shift
      ;;
    --source)
      BUILD_FROM_SOURCE=true
      shift
      ;;
    --clean)
      CLEAN=true
      shift
      ;;
    --venv)
      VENV=true
      shift
      ;;
    --help)
      echo "Example usage: ./compile-pytorch-ipex.sh [--pytorch | --ipex | --pinned | --source | --clean | --venv]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

set +o xtrace
if [ ! -v BASE ]; then
  echo "**** BASE is not given *****"
  BASE=$(cd $(dirname "$0")/../.. && pwd)
  echo "**** Default BASE is set to $BASE ****"
fi

if [ "$VENV" = true ]; then
  source .venv/bin/activate
fi

export PYTORCH_PROJ=$BASE/pytorch
export IPEX_PROJ=$BASE/intel-extension-for-pytorch

if [ "$CLEAN" = true ]; then
  echo "**** Cleaning $PYTORCH_PROJ and $IPEX_PROJ before build ****"
  if rm -rf $PYTORCH_PROJ $IPEX_PROJ &>/dev/null; then
    pip uninstall -y torch intel_extension_for_pytorch
  fi
fi

if [ "$BUILD_PINNED" = true ]; then
  # Determine if the installed PyTorch version is the same as the pinned version.
  INSTALL_PYTORCH=true
  if pip show torch &>/dev/null; then
    PYTORCH_PINNED_COMMIT="$(<$BASE/intel-xpu-backend-for-triton/.github/pins/pytorch.txt)"
    PYTORCH_CURRENT_COMMIT=`python -c "import torch;print(torch.__version__)"`
    PYTORCH_CURRENT_COMMIT=${PYTORCH_CURRENT_COMMIT#*"git"}
    if [[ "$PYTORCH_PINNED_COMMIT" = "$PYTORCH_CURRENT_COMMIT"* ]]; then
      INSTALL_PYTORCH=false
    else
      echo "**** Current PyTorch commit $PYTORCH_CURRENT_COMMIT ****"
      echo "**** Pinned PyTorch commit $PYTORCH_PINNED_COMMIT ****"
    fi
  fi
  # Determine if the installed IPEX version is the same as the pinned version.
  INSTALL_IPEX=true
  if pip show intel-extension-for-pytorch &>/dev/null; then
    IPEX_PINNED_COMMIT="$(<$BASE/intel-xpu-backend-for-triton/.github/pins/ipex.txt)"
    IPEX_CURRENT_COMMIT=`python -c "import torch;import intel_extension_for_pytorch as ipex;print(ipex.__version__)"`
    IPEX_CURRENT_COMMIT=${IPEX_CURRENT_COMMIT#*"git"}
    if [[ "$IPEX_PINNED_COMMIT" = "$IPEX_CURRENT_COMMIT"* ]]; then
      INSTALL_IPEX=false
    else
      echo "**** Current IPEX commit $IPEX_CURRENT_COMMIT ****"
      echo "**** Pinned IPEX commit $IPEX_PINNED_COMMIT ****"
    fi
  fi

  if [[ "$INSTALL_PYTORCH" = false && "$INSTALL_IPEX" = false ]]; then
    exit 0
  fi
fi

if [ "$BUILD_FROM_SOURCE" = false ]; then
  if `! which gh &> /dev/null` || `! which jq &> /dev/null`; then
    echo "****** WARNING: gh or jq is missing ******"
    BUILD_FROM_SOURCE=true
  fi
fi

if [ "$BUILD_PINNED" = false ]; then
  BUILD_FROM_SOURCE=true
fi

if [ "$BUILD_FROM_SOURCE" = false ]; then
  PYTHON_VERSION=$( python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" )
  RUN_ID=$(gh run list -w "Triton wheels" -R intel/intel-xpu-backend-for-triton --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')
  TEMP_DIR=$(mktemp -d)
  gh run download $RUN_ID \
    --repo intel/intel-xpu-backend-for-triton \
    --pattern "wheels-py${PYTHON_VERSION}*" \
    --dir $TEMP_DIR
  cd $TEMP_DIR/wheels-py${PYTHON_VERSION}*
  pip install torch-* intel_extension_for_pytorch-*
  rm -r $TEMP_DIR
  exit 0
fi


if [ "$BUILD_PYTORCH" = false ] && [ "$BUILD_IPEX" = false ]; then
  # Avoid overriding the user's existing PyTorch by default.
  if ! pip show torch &>/dev/null; then
    BUILD_PYTORCH=true
  fi
  # Avoid overriding the user's existing IPEX by default.
  if ! pip show intel_extension_for_pytorch &>/dev/null; then
    BUILD_IPEX=true
  fi
  if [ "$BUILD_PINNED" = true ]; then
    BUILD_PYTORCH=true
    BUILD_IPEX=true
  fi
fi

############################################################################
# Configure and build the pytorch project.

build_pytorch() {
  if [ ! -d "$PYTORCH_PROJ" ]; then
    echo "**** Cloning $PYTORCH_PROJ ****"
    cd $BASE
    git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/Stonepia/pytorch.git
  fi
  echo "****** Building $PYTORCH_PROJ ******"
  cd $PYTORCH_PROJ
  if [ ! -d "$PYTORCH_PROJ/dist" ]; then
    if [ "$BUILD_PINNED" = true ]; then
      git fetch --all
      PYTORCH_COMMIT_ID="$(<$BASE/intel-xpu-backend-for-triton/.github/pins/pytorch.txt)"
      git checkout $PYTORCH_COMMIT_ID
      git submodule update --recursive
    fi
    pip install cmake ninja
    pip install mkl-static mkl-include
    pip install -r requirements.txt
    python setup.py bdist_wheel
  fi
  pip install dist/*.whl
  cd $BASE
  python -c "import torch;print(torch.__version__)"
}

############################################################################
# Configure and build the ipex project.

build_ipex() {
  if [ ! -d "$IPEX_PROJ" ]; then
    echo "**** Cloning $IPEX_PROJ ****"
    cd $BASE
    git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/intel/intel-extension-for-pytorch.git
  fi
  echo "****** Building $IPEX_PROJ ******"
  cd $IPEX_PROJ
  if [ ! -d "$IPEX_PROJ/dist" ]; then
    if [ "$BUILD_PINNED" = true ]; then
      IPEX_COMMIT_ID="$(<$BASE/intel-xpu-backend-for-triton/.github/pins/ipex.txt)"
      git fetch origin $IPEX_COMMIT_ID
      git checkout $IPEX_COMMIT_ID
      git submodule sync
      git submodule update --init --recursive
    fi
    pip install -r requirements.txt
    python setup.py bdist_wheel
  fi
  pip install dist/*.whl
  cd $BASE
  python -c "import torch;import intel_extension_for_pytorch as ipex;print(ipex.__version__)"
}

build() {
  if [ "$BUILD_PYTORCH" = true ]; then
    build_pytorch
  fi
  if [ "$BUILD_IPEX" = true ]; then
    build_ipex
  fi
}

build
