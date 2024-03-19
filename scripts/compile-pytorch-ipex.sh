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

if `! which gh &> /dev/null` || `! which jq &> /dev/null`; then
  echo "****** WARNING: gh or jq is missing ******"
  BUILD_FROM_SOURCE=true
fi
if [ "$BUILD_PINNED" = false ]; then
  BUILD_FROM_SOURCE=true
fi

if [ "$BUILD_FROM_SOURCE" = false ]; then
  # Determine if the installed PyTorch version is the same as the pinned version.
  INSTALL_PYTORCH=true
  if pip show torch &>/dev/null; then
    PYTORCH_PINNED_COMMIT="$(<$BASE/intel-xpu-backend-for-triton/.github/pins/pytorch.txt)"
    PYTORCH_CURRENT_COMMIT=`python -c "import torch;print(torch.__version__)"`
    PYTORCH_CURRENT_COMMIT=${PYTORCH_CURRENT_COMMIT#*"git"}
    if [[ "$PYTORCH_PINNED_COMMIT" = "$PYTORCH_CURRENT_COMMIT"* ]]; then
      INSTALL_PYTORCH=false
    fi
  fi
  # Determine if the installed IPEX version is the same as the pinned version.
  INSTALL_IPEX=true
  if pip show intel_extension_for_pytorch &>/dev/null; then
    IPEX_PINNED_COMMIT="$(<$BASE/intel-xpu-backend-for-triton/.github/pins/ipex.txt)"
    IPEX_CURRENT_COMMIT=`python -c "import torch;import intel_extension_for_pytorch as ipex;print(ipex.__version__)"`
    IPEX_CURRENT_COMMIT=${IPEX_CURRENT_COMMIT#*"git"}
    if [[ "$IPEX_PINNED_COMMIT" = "$IPEX_CURRENT_COMMIT"* ]]; then
      INSTALL_IPEX=false
    fi
  fi

  if [[ "$INSTALL_PYTORCH" = true || "$INSTALL_IPEX" = true ]]; then
    TEMP_DIR=`mktemp -d`
    gh run download $(gh run list -w "Triton wheels" -R intel/intel-xpu-backend-for-triton --json databaseId | jq '.[0].databaseId') -R intel/intel-xpu-backend-for-triton
    PYTHON_VERSION=$( python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" )
    cd wheels-py${PYTHON_VERSION}*
    pip install torch-* intel_extension_for_pytorch-*
    rm -r $TEMP_DIR
  fi

  exit 0
fi

export PYTORCH_PROJ=$BASE/pytorch
export IPEX_PROJ=$BASE/intel-extension-for-pytorch

if [ "$CLEAN" = true ]; then
  echo "**** Cleaning $PYTORCH_PROJ and $IPEX_PROJ before build ****"
  if rm -rf $PYTORCH_PROJ $IPEX_PROJ &>/dev/null; then
    pip uninstall -y torch intel_extension_for_pytorch
  fi
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
fi

if [ "$VENV" = true ]; then
  source .venv/bin/activate
fi

check_rc() {
  if [ $? != 0 ]; then
    echo "Command failed with rc: $rc"
    exit 1
  fi
}

############################################################################
# Configure and build the pytorch project.

build_pytorch() {
  if [ ! -d "$PYTORCH_PROJ" ]; then
    echo "**** Cloning $PYTORCH_PROJ ****"
    cd $BASE
    git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/Stonepia/pytorch.git
    PYTORCH_COMMIT_ID="$(<$BASE/intel-xpu-backend-for-triton/.github/pins/pytorch.txt)"
    git checkout $PYTORCH_COMMIT_ID
    git submodule update --recursive
  fi
  echo "****** Building $PYTORCH_PROJ ******"
  cd $PYTORCH_PROJ
  if [ ! -d "$PYTORCH_PROJ/dist" ]; then
    pip install cmake ninja
    pip install mkl-static mkl-include
    pip install -r requirements.txt
    python setup.py bdist_wheel
  fi
  pip install dist/*.whl
  cd $BASE
  python -c "import torch;print(torch.__version__)"
  check_rc
}

############################################################################
# Configure and build the ipex project.

build_ipex() {
  if [ ! -d "$IPEX_PROJ" ]; then
    echo "**** Cloning $IPEX_PROJ ****"
    cd $BASE
    git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/intel/intel-extension-for-pytorch.git
    IPEX_COMMIT_ID="$(<$BASE/intel-xpu-backend-for-triton/.github/pins/ipex.txt)"
    git checkout $IPEX_COMMIT_ID
    git submodule update --recursive
  fi
  echo "****** Building $IPEX_PROJ ******"
  cd $IPEX_PROJ
  if [ ! -d "$IPEX_PROJ/dist" ]; then
    pip install -r requirements.txt
    python setup.py bdist_wheel
  fi
  pip install dist/*.whl
  cd $BASE
  python -c "import torch;import intel_extension_for_pytorch as ipex;print(ipex.__version__)"
  check_rc
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
