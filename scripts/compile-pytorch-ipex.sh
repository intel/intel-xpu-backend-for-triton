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
      echo "Example usage: ./compile-triton.sh [--pytorch | --ipex | --clean | --venv]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

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

set +o xtrace
if [ -z "$BASE" ]; then
  echo "**** BASE is not given *****"
  BASE=$(cd $(dirname "$0")/../.. && pwd)
  echo "**** Default BASE is set to $BASE ****"
fi

export PYTORCH_PROJ=$BASE/pytorch
export IPEX_PROJ=$BASE/intel-extension-for-pytorch

if [ "$CLEAN" = true ]; then
  echo "**** Cleaning $PYTORCH_PROJ and $IPEX_PROJ before build ****"
  rm -rf $PYTORCH_PROJ $IPEX_PROJ
  pip uninstall -y torch intel_extension_for_pytorch
fi

if [ "$VENV" = true ]; then
  echo "**** Creating Python virtualenv ****"
  python3 -m venv .venv --prompt triton
  source .venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
  echo "**** Cleaning up Python virtualenv ****"
  deactivate
fi

check_rc() {
  if [ $? != 0 ]; then
    echo "Command failed with rc: $rc"
    exit 1
  fi
}

############################################################################
# Clone the PyTorch repository.

if [ ! -d "$PYTORCH_PROJ" ]; then
  echo "**** Cloning $PYTORCH_PROJ ****"
  cd $BASE
  git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/Stonepia/pytorch.git
fi

############################################################################
## Configure and build the pytorch project.

build_pytorch() {
  echo "****** Building $PYTORCH_PROJ ******"
  cd $PYTORCH_PROJ
  if [ ! -d "$PYTORCH_PROJ/dist" ]; then
    pip install -r requirements.txt
    python setup.py bdist_wheel
  fi
  pip install dist/*.whl
}

############################################################################
# Clone the IPEX repository.

if [ ! -d "$IPEX_PROJ" ]; then
  echo "**** Cloning $IPEX_PROJ ****"
  cd $BASE
  git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/intel/intel-extension-for-pytorch.git
fi

############################################################################
## Configure and build the ipex project.

build_ipex() {
  echo "****** Building $IPEX_PROJ ******"
  cd $IPEX_PROJ
  if [ ! -d "$IPEX_PROJ/dist" ]; then
    pip install -r requirements.txt
    python setup.py bdist_wheel
  fi
  pip install dist/*.whl
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
