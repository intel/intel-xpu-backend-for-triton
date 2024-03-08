#!/usr/bin/env bash

# Select what to build.
BUILD_PYTORCH=false
BUILD_IPEX=false
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
    --clean)
      CLEAN=true
      shift
      ;;
    --venv)
      VENV=true
      shift
      ;;
    --help)
      echo "Example usage: ./compile-pytorch-ipex.sh [--pytorch | --ipex | --clean | --venv]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

SCRIPTS_DIR=$(dirname "$0")
source "$SCRIPTS_DIR"/functions.sh

set_env

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
    PYTORCH_COMMIT_ID="$(<.github/pins/pytorch.txt)" || true
    git checkout $PYTORCH_COMMIT_ID
    git submodule update --recursive
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
    IPEX_COMMIT_ID="$(<.github/pins/ipex.txt)" || true
    git checkout $IPEX_COMMIT_ID
    git submodule update --recursive
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
